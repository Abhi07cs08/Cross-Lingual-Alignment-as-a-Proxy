#!/usr/bin/env python3
"""MT baseline: reproduce the English-pivot of Al Ali et al. (2026), 'Is English Enough?',
and extend it to the non-English-pivot analysis.

Pipeline:
  1. p-wm sentence reps per layer (FLORES-200 devtest)
  2. CLA(src, pivot) via ANC at the max-alignment layer, for each pivot (en, plus others)
  3. 5-shot greedy translation src->tgt, scored with chrF++
  4. per-target Pearson(CLA, chrF) across source languages, averaged over targets, for each
     pivot direction, with bootstrap CIs.

What this version adds over the first run:
  - the full language set (not the 12-language slice), toward the paper's 44;
  - timeout-safe incremental writes: the JSON is rewritten after EVERY translation pair, so
    a walltime timeout keeps all completed pairs (the first MT run lost everything on a
    timeout);
  - a generalized pivot correlation, so alignment-to-{en,es,ru,zh,sw} are all scored, which
    feeds the "is English special?" analysis directly from the run instead of a separate
    rerun;
  - bootstrap CIs on the mean per-target correlation.

Target to reproduce (Al Ali Table 6): src-en Pearson ~0.63-0.88, clearly > src-tgt.
Offline-safe: run prefetch.py on the login node first. Reuses cla.py and translate.py.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
from scipy.stats import pearsonr, spearmanr

from cla import encode, cla_pair
from translate import translate_hf, chrf_pp

# FLORES-200 codes: a broad high + mid + low resource set (extend toward the paper's 44).
FLORES = {
    "en": "eng_Latn", "de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "ru": "rus_Cyrl",
    "pl": "pol_Latn", "uk": "ukr_Cyrl", "cs": "ces_Latn", "ro": "ron_Latn",
    "el": "ell_Grek", "tr": "tur_Latn", "ar": "arb_Arab", "he": "heb_Hebr",
    "fa": "pes_Arab", "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml",
    "te": "tel_Telu", "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn",
    "ko": "kor_Hang", "ja": "jpn_Jpan", "zh": "cmn_Hans", "sw": "swh_Latn",
}

# Pivots scored for the "is English special?" analysis (must be in --langs).
PIVOT_PREFERENCE = ["en", "es", "ru", "zh", "sw"]


def load_flores(codes, n):
    from datasets import load_dataset
    out = {}
    for iso, fl in codes.items():
        ds = load_dataset("openlanguagedata/flores_plus", fl, split="devtest")
        col = "text" if "text" in ds.column_names else ds.column_names[0]
        out[iso] = [r[col] for r in ds][:n]
    return out


def per_target_correlation(cla, chrf, languages, pivot="en", method="pearson"):
    """Mean per-target correlation of alignment vs chrF across source languages.

    pivot="tgt": align each source to the target language it is translating into.
    pivot=<code>: align each source to a fixed pivot language (e.g. "en", "es").
    """
    fn = pearsonr if method == "pearson" else spearmanr
    per_target = {}
    for tgt in languages:
        srcs = []
        for l in languages:
            if l == tgt or (l, tgt) not in chrf:
                continue
            if pivot != "tgt" and (l == pivot or (l, pivot) not in cla):
                continue
            srcs.append(l)
        if len(srcs) < 3:
            continue
        x = [cla[(s, tgt)] if pivot == "tgt" else cla[(s, pivot)] for s in srcs]
        y = [chrf[(s, tgt)] for s in srcs]
        per_target[tgt] = float(fn(x, y).statistic)
    vals = [v for v in per_target.values() if v == v]
    return (float(np.mean(vals)) if vals else float("nan")), per_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B-Base")
    ap.add_argument("--langs", default=",".join(FLORES))
    ap.add_argument("--targets", default="es,fr,de,ru,zh,sw,tr")
    ap.add_argument("--pivots", default="", help="comma list; default = preferred pivots present")
    ap.add_argument("--n-align", type=int, default=500)
    ap.add_argument("--n-translate", type=int, default=300)
    ap.add_argument("--measure", default="anc", choices=["anc", "abs"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="results/mt_baseline.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    langs = [c for c in args.langs.split(",") if c]
    targets = [c for c in args.targets.split(",") if c and c in langs]
    assert "en" in langs, "English ('en') must be in --langs for the pivot direction"
    non_en = [c for c in langs if c != "en"]
    if args.pivots:
        pivots = [p for p in args.pivots.split(",") if p and p in langs]
    else:
        pivots = [p for p in PIVOT_PREFERENCE if p in langs]
    if "en" not in pivots:
        pivots = ["en"] + pivots
    codes = {c: FLORES[c] for c in langs}
    dtype = getattr(torch, args.dtype)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    log(f"model={args.model} langs={langs} targets={targets} pivots={pivots} "
        f"n_align={args.n_align} n_translate={args.n_translate}")

    sents = load_flores(codes, max(args.n_align, args.n_translate + 5))
    log(f"FLORES loaded ({time.time()-t0:.0f}s)")

    tok = AutoTokenizer.from_pretrained(args.model)
    # do NOT pass output_hidden_states=True at load (it OOMs generate()); encode() asks per-call.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    log(f"model loaded ({time.time()-t0:.0f}s)")

    reps = {iso: encode(model, tok, sents[iso][:args.n_align], batch_size=args.batch_size)
            for iso in codes}
    log(f"encoded reps ({time.time()-t0:.0f}s)")

    # right-hand languages we need alignment to: every pivot and every target
    rhs = sorted(set(pivots) | set(targets))
    cla, chrf = {}, {}
    for s in non_en:
        for r in rhs:
            if s != r:
                cla[(s, r)] = cla_pair(reps[s], reps[r], args.measure)

    def summarise():
        out = {"model": args.model, "measure": args.measure, "langs": langs,
               "targets": targets, "pivots": pivots, "n_align": args.n_align,
               "n_translate": args.n_translate,
               "cla": {f"{a}|{b}": v for (a, b), v in cla.items()},
               "chrf": {f"{a}|{b}": v for (a, b), v in chrf.items()},
               "runtime_sec": round(time.time() - t0, 1)}
        for method in ("pearson", "spearman"):
            for p in pivots:
                m, per = per_target_correlation(cla, chrf, non_en, p, method)
                out[f"src_{p}_{method}"] = m
                out[f"per_target_{p}_{method}"] = per
            mt, pert = per_target_correlation(cla, chrf, non_en, "tgt", method)
            out[f"src_tgt_{method}"] = mt
            out[f"per_target_tgt_{method}"] = pert
        # bootstrap CI on the mean per-target correlation, per pivot (resample targets)
        for p in pivots:
            _, per = per_target_correlation(cla, chrf, non_en, p, "pearson")
            vals = [v for v in per.values() if v == v]
            if len(vals) >= 3:
                rng = np.random.default_rng(0)
                means = [float(np.mean(rng.choice(vals, len(vals), replace=True)))
                         for _ in range(5000)]
                lo, hi = np.percentile(means, [2.5, 97.5])
                out[f"src_{p}_pearson_ci"] = {"mean": float(np.mean(vals)),
                                              "ci_low": float(lo), "ci_high": float(hi),
                                              "n_targets": len(vals)}
        return out

    def dump():
        with open(args.out, "w") as f:
            json.dump(summarise(), f, indent=2)

    dump()  # write the alignment-only state immediately
    for t in targets:
        for s in non_en:
            if s == t:
                continue
            shots = list(zip(sents[s][:5], sents[t][:5]))
            src_eval = sents[s][5:5 + args.n_translate]
            tgt_ref = sents[t][5:5 + args.n_translate]
            hyps = translate_hf(model, tok, src_eval, shots,
                                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
            chrf[(s, t)] = chrf_pp(hyps, tgt_ref)
            dump()  # timeout-safe: rewrite after every pair
            log(f"{s}->{t}  CLA_en={cla.get((s,'en'), float('nan')):.3f}  chrF={chrf[(s,t)]:.1f}")

    out = summarise()
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print(f"RESULT  src-en(Pearson)={out['src_en_pearson']:+.3f}  src-tgt={out['src_tgt_pearson']:+.3f}")
    print(f"        src-en(Spearman)={out['src_en_spearman']:+.3f}  src-tgt={out['src_tgt_spearman']:+.3f}")
    print("  pivots (Pearson, mean per-target):")
    for p in pivots:
        print(f"    {p}: {out.get(f'src_{p}_pearson', float('nan')):+.3f}")
    print(f"  English pivot {'HOLDS' if out['src_en_pearson'] > out['src_tgt_pearson'] else 'does NOT hold'}"
          f"  (paper src-en ~0.63-0.88)")
    print(f"written to {args.out}  ({out['runtime_sec']:.0f}s)")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
