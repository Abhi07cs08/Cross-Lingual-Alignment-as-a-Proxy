#!/usr/bin/env python3
"""MT baseline: reproduce the English-pivot of Al Ali et al. (2026), 'Is English Enough?'.

Pipeline:
  1. p-wm sentence reps per layer (FLORES-200 devtest)
  2. CLA(src,en) and CLA(src,tgt) via ANC at the max-alignment layer
  3. 5-shot greedy translation src->tgt, scored with chrF++
  4. per-target Pearson(CLA, chrF) across source languages, averaged over targets,
     for the 'en' (English-pivot) and 'tgt' directions.

Target to reproduce (Al Ali Table 6): src-en Pearson ~0.63-0.88, clearly > src-tgt.
Offline-safe: run prefetch.py on the login node first. Reuses cla.py and translate.py.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
from scipy.stats import pearsonr, spearmanr

from cla import encode, cla_pair
from translate import translate_hf, chrf_pp

# FLORES-200 codes: a 12-language slice (high + low resource). Extend toward the paper's 44.
FLORES = {
    "en": "eng_Latn", "de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn",
    "ru": "rus_Cyrl", "zh": "cmn_Hans", "hi": "hin_Deva", "sw": "swh_Latn",
    "bn": "ben_Beng", "ta": "tam_Taml", "th": "tha_Thai", "tr": "tur_Latn",
}


def load_flores(codes, n):
    from datasets import load_dataset
    out = {}
    for iso, fl in codes.items():
        ds = load_dataset("openlanguagedata/flores_plus", fl, split="devtest")
        col = "text" if "text" in ds.column_names else ds.column_names[0]
        out[iso] = [r[col] for r in ds][:n]
    return out


def per_target_correlation(cla, chrf, languages, which="en", method="pearson"):
    fn = pearsonr if method == "pearson" else spearmanr
    per_target = {}
    for tgt in languages:
        srcs = [l for l in languages if l != tgt and (l, tgt) in chrf]
        x = [cla[(s, "en")] if which == "en" else cla[(s, tgt)] for s in srcs]
        y = [chrf[(s, tgt)] for s in srcs]
        if len(srcs) >= 3:
            per_target[tgt] = float(fn(x, y).statistic)
    vals = [v for v in per_target.values() if v == v]
    return (float(np.mean(vals)) if vals else float("nan")), per_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B-Base")
    ap.add_argument("--langs", default=",".join(FLORES))
    ap.add_argument("--targets", default="es,zh,sw,hi,ru")
    ap.add_argument("--n-align", type=int, default=500)
    ap.add_argument("--n-translate", type=int, default=200)
    ap.add_argument("--measure", default="anc", choices=["anc", "abs"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="results/mt_baseline.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    langs = [c for c in args.langs.split(",") if c]
    targets = [c for c in args.targets.split(",") if c]
    assert "en" in langs, "English ('en') must be in --langs for the pivot direction"
    non_en = [c for c in langs if c != "en"]
    codes = {c: FLORES[c] for c in langs}
    dtype = getattr(torch, args.dtype)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] model={args.model} langs={langs} targets={targets} "
          f"n_align={args.n_align} n_translate={args.n_translate}", flush=True)

    sents = load_flores(codes, max(args.n_align, args.n_translate + 5))
    print(f"[{time.strftime('%H:%M:%S')}] FLORES loaded ({time.time()-t0:.0f}s)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    # do NOT pass output_hidden_states=True at load (it OOMs generate()); encode() asks per-call.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] model loaded ({time.time()-t0:.0f}s)", flush=True)

    reps = {iso: encode(model, tok, sents[iso][:args.n_align], batch_size=args.batch_size)
            for iso in codes}
    print(f"[{time.strftime('%H:%M:%S')}] encoded reps ({time.time()-t0:.0f}s)", flush=True)

    cla, chrf = {}, {}
    for s in non_en:
        cla[(s, "en")] = cla_pair(reps[s], reps["en"], args.measure)
    for t in targets:
        for s in non_en:
            if s == t:
                continue
            cla[(s, t)] = cla_pair(reps[s], reps[t], args.measure)
            shots = list(zip(sents[s][:5], sents[t][:5]))
            src_eval = sents[s][5:5 + args.n_translate]
            tgt_ref = sents[t][5:5 + args.n_translate]
            hyps = translate_hf(model, tok, src_eval, shots,
                                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
            chrf[(s, t)] = chrf_pp(hyps, tgt_ref)
            print(f"  [{time.strftime('%H:%M:%S')}] {s}->{t}  "
                  f"CLA={cla[(s,t)]:.3f} chrF={chrf[(s,t)]:.1f}", flush=True)

    out = {"model": args.model, "measure": args.measure, "langs": langs, "targets": targets,
           "n_align": args.n_align, "n_translate": args.n_translate,
           "cla": {f"{a}|{b}": v for (a, b), v in cla.items()},
           "chrf": {f"{a}|{b}": v for (a, b), v in chrf.items()},
           "runtime_sec": round(time.time() - t0, 1)}
    for method in ("pearson", "spearman"):
        me, pe = per_target_correlation(cla, chrf, non_en, "en", method)
        mt, pt = per_target_correlation(cla, chrf, non_en, "tgt", method)
        out[f"src_en_{method}"] = me
        out[f"src_tgt_{method}"] = mt
        out[f"per_target_en_{method}"] = pe
        out[f"per_target_tgt_{method}"] = pt
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print(f"RESULT  src-en(Pearson)={out['src_en_pearson']:+.3f}  src-tgt={out['src_tgt_pearson']:+.3f}")
    print(f"        src-en(Spearman)={out['src_en_spearman']:+.3f}  src-tgt={out['src_tgt_spearman']:+.3f}")
    print(f"pivot {'HOLDS' if out['src_en_pearson'] > out['src_tgt_pearson'] else 'does NOT hold'}"
          f"  (paper src-en ~0.63-0.88)")
    print(f"written to {args.out}  ({out['runtime_sec']:.0f}s)")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
