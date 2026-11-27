#!/usr/bin/env python3
"""Reasoning arm: does cross-lingual alignment predict multilingual reasoning accuracy?

For each language: measure alignment (p-wm + ANC + max-layer, on FLORES) to a set of
PIVOT languages, and measure reasoning accuracy on MGSM / AfriMGSM with chain-of-thought
(8-shot native CoT, greedy). Then, across languages, correlate alignment-to-pivot against
accuracy, for each pivot, with bootstrap CIs, plus a partial correlation that controls for
a resource-level proxy.

What this version adds over the first run:
  - multi-pivot alignment (not just English), so the "is English special?" question is
    answered on reasoning too, not only translation;
  - robust answer extraction: non-Latin digits AND Chinese/Japanese numerals, plus common
    answer markers, which fixes the CJK extraction artifact (zh looked well-aligned but
    scored near zero because the extractor missed CJK numerals);
  - bootstrap confidence intervals and a partial correlation controlling for resource level
    (the main confound);
  - heavier default settings for the final run.

Robustness: results are written to the JSON after EVERY language, so a walltime timeout
loses only the language in progress. Any language whose data fails to load is skipped with
a logged note, not crashed on. Offline-safe: run prefetch.py on the login node first.
"""
from __future__ import annotations
import argparse, json, os, re, time, unicodedata
import numpy as np
from cla import encode, cla_pair
from stats import corr_with_ci, partial_spearman_with_ci
from resources import resource_of

# iso -> (dataset config name, FLORES+ config for alignment)
MGSM = {
    "en": ("en", "eng_Latn"), "de": ("de", "deu_Latn"), "es": ("es", "spa_Latn"),
    "fr": ("fr", "fra_Latn"), "ru": ("ru", "rus_Cyrl"), "zh": ("zh", "cmn_Hans"),
    "ja": ("ja", "jpn_Jpan"), "th": ("th", "tha_Thai"), "sw": ("sw", "swh_Latn"),
    "bn": ("bn", "ben_Beng"), "te": ("te", "tel_Telu"),
}
AFRIMGSM = {
    "en": ("eng", "eng_Latn"), "fr": ("fra", "fra_Latn"), "amh": ("amh", "amh_Ethi"),
    "swa": ("swa", "swh_Latn"), "hau": ("hau", "hau_Latn"), "ibo": ("ibo", "ibo_Latn"),
    "kin": ("kin", "kin_Latn"), "lin": ("lin", "lin_Latn"), "lug": ("lug", "lug_Latn"),
    "orm": ("orm", "gaz_Latn"), "sna": ("sna", "sna_Latn"), "sot": ("sot", "sot_Latn"),
    "twi": ("twi", "twi_Latn_akua1239"), "ewe": ("ewe", "ewe_Latn"), "wol": ("wol", "wol_Latn"),
    "xho": ("xho", "xho_Latn"), "yor": ("yor", "yor_Latn"), "zul": ("zul", "zul_Latn"),
}
DATASETS = {"mgsm": ("juletxara/mgsm", MGSM), "afrimgsm": ("masakhane/afrimgsm", AFRIMGSM)}

# Preferred pivots, in order; only those actually present in the run's languages are used.
# English is always a pivot/anchor; the others let us test whether English is special.
PIVOT_PREFERENCE = ["en", "es", "ru", "zh", "fr"]


# ---------- answer extraction ----------
_CJK_NUM = {"〇": 0, "零": 0, "一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100,
            "千": 1000, "萬": 10000, "万": 10000, "億": 10 ** 8, "亿": 10 ** 8}
_CJK_RE = re.compile("[" + "".join(_CJK_NUM) + "]+")


def _cjk_run_to_int(run: str):
    """Convert a run of CJK numerals (e.g. 二十三) to an int. None if it does not parse."""
    total = 0      # accumulated at 10^4 boundaries
    section = 0    # accumulated within the current 10^4 block
    number = 0     # the pending single digit
    for ch in run:
        v = _CJK_NUM[ch]
        if v >= 10000:
            section = (section + number) * v
            total += section
            section = 0
            number = 0
        elif v >= 10:
            if number == 0:
                number = 1            # bare 十 == 10
            section += number * v
            number = 0
        else:
            number = v
    val = total + section + number
    return val


def _normalise_numbers(text: str) -> str:
    # per-character digit normalisation: Devanagari/Bengali/Thai/fullwidth -> ASCII
    out = []
    for ch in text:
        if ch.isdigit():
            d = unicodedata.digit(ch, None)
            out.append(str(d) if d is not None else ch)
        else:
            out.append(ch)
    s = "".join(out)
    # CJK ideographic numerals -> ASCII integers
    s = _CJK_RE.sub(lambda m: str(_cjk_run_to_int(m.group(0))), s)
    # drop thousands separators sitting between digits
    s = re.sub(r"(?<=\d)[, \s](?=\d)", "", s)
    return s


def extract_answer(text: str):
    """Best-effort final integer from a chain-of-thought generation.

    Order: normalise all numerals, then prefer an explicit answer marker
    (GSM8k '####', or 'answer is ...', '答案', '=' near the end); otherwise
    fall back to the last integer in the text.
    """
    t = _normalise_numbers(text)
    # explicit markers, strongest first
    m = re.findall(r"####\s*(-?\d+)", t)
    if m:
        return _to_int(m[-1])
    m = re.findall(r"(?:answer|ans|result|total)\s*(?:is|:|=)?\s*(-?\d+)", t, re.I)
    if m:
        return _to_int(m[-1])
    nums = re.findall(r"-?\d+", t)
    return _to_int(nums[-1]) if nums else None


def _to_int(s):
    try:
        return int(s)
    except Exception:
        return None


# ---------- data ----------
def flores_sents(fl, n):
    from datasets import load_dataset
    ds = load_dataset("openlanguagedata/flores_plus", fl, split="devtest")
    col = "text" if "text" in ds.column_names else ds.column_names[0]
    return [r[col] for r in ds][:n]


def load_split(ds_id, cfg, split):
    from datasets import load_dataset
    return list(load_dataset(ds_id, cfg, split=split))


def build_prompt(shots, question):
    p = ""
    for ex in shots:
        p += ex["question"].strip() + "\n" + (ex["answer"] or "").strip() + "\n\n"
    return p + question.strip() + "\n"


def generate_cot(model, tok, prompts, max_new_tokens=512, batch_size=8):
    import torch
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=4096).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for txt in tok.batch_decode(new, skip_special_tokens=True):
                cut = re.split(r"\n\s*(?:Question|Q)\s*[:\.]", txt)[0]
                out.append(cut)
    return out


# ---------- stats assembly ----------
def compute_stats(per_lang, pivots):
    """Correlations of alignment-to-pivot vs accuracy, across languages, per pivot.

    Also a partial Spearman controlling for the resource-level proxy, using the
    English pivot (the headline direction).
    """
    out = {"pivots": {}}
    for p in pivots:
        rows = {l: v for l, v in per_lang.items()
                if l != p and "accuracy" in v and v.get("cla", {}).get(p) is not None}
        if len(rows) < 3:
            continue
        x = [v["cla"][p] for v in rows.values()]
        y = [v["accuracy"] for v in rows.values()]
        out["pivots"][p] = {
            "pearson": corr_with_ci(x, y, "pearson"),
            "spearman": corr_with_ci(x, y, "spearman"),
            "langs": list(rows),
        }
    # headline English-pivot keys kept for backward compatibility
    if "en" in out["pivots"]:
        out["src_en_vs_accuracy_pearson"] = out["pivots"]["en"]["pearson"]["r"]
        out["src_en_vs_accuracy_spearman"] = out["pivots"]["en"]["spearman"]["r"]
        out["n_langs_scored"] = out["pivots"]["en"]["pearson"]["n"]
    # partial correlation controlling for resource level (English pivot)
    rows = {l: v for l, v in per_lang.items()
            if l != "en" and "accuracy" in v and v.get("cla", {}).get("en") is not None
            and resource_of(l) is not None}
    if len(rows) >= 5:
        x = [v["cla"]["en"] for v in rows.values()]
        y = [v["accuracy"] for v in rows.values()]
        z = [resource_of(l) for l in rows]
        out["partial_spearman_en_controlling_resource"] = partial_spearman_with_ci(x, y, z)
        out["partial_resource_langs"] = list(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mgsm", "afrimgsm"], default="mgsm")
    ap.add_argument("--model", default="Qwen/Qwen3-14B-Base")
    ap.add_argument("--langs", default="")
    ap.add_argument("--pivots", default="", help="comma list; default = preferred pivots present")
    ap.add_argument("--n-align", type=int, default=800)
    ap.add_argument("--n-reason", type=int, default=250)
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="results/reason.json")
    args = ap.parse_args()

    ds_id, LMAP = DATASETS[args.dataset]
    langs = [l for l in args.langs.split(",") if l] or list(LMAP)
    assert "en" in langs, "keep 'en' in --langs (English is the anchor pivot)"
    if args.pivots:
        pivots = [p for p in args.pivots.split(",") if p and p in langs]
    else:
        pivots = [p for p in PIVOT_PREFERENCE if p in langs]
    if "en" not in pivots:
        pivots = ["en"] + pivots

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    log(f"dataset={args.dataset} model={args.model} langs={langs} pivots={pivots} "
        f"n_align={args.n_align} n_reason={args.n_reason}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    log(f"model loaded ({time.time()-t0:.0f}s)")

    # cache pivot representations once (English + any other pivots present)
    pivot_reps = {}
    for p in pivots:
        pivot_reps[p] = encode(model, tok, flores_sents(LMAP[p][1], args.n_align), batch_size=16)
        log(f"cached pivot reps: {p}")

    results = {"dataset": args.dataset, "model": args.model, "measure": "anc",
               "n_align": args.n_align, "n_reason": args.n_reason, "pivots": pivots,
               "per_lang": {}}

    def dump():
        results.update({k: v for k, v in compute_stats(results["per_lang"], pivots).items()})
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    for lang in langs:
        cfg, fl = LMAP[lang]
        rec = {"cla": {}}
        try:
            reps = pivot_reps.get(lang)
            if reps is None:
                reps = encode(model, tok, flores_sents(fl, args.n_align), batch_size=16)
            for p in pivots:
                rec["cla"][p] = 1.0 if p == lang else cla_pair(reps, pivot_reps[p], "anc")
            rec["cla_en"] = rec["cla"].get("en")
        except Exception as e:
            rec["align_error"] = str(e)[:200]
        try:
            shots = load_split(ds_id, cfg, "train")[:args.shots]
            test = load_split(ds_id, cfg, "test")[:args.n_reason]
            prompts = [build_prompt(shots, ex["question"]) for ex in test]
            gens = generate_cot(model, tok, prompts, args.max_new_tokens, args.batch_size)
            correct = sum(extract_answer(g) == int(ex["answer_number"])
                          for g, ex in zip(gens, test))
            rec["accuracy"] = correct / len(test)
            rec["n"] = len(test)
        except Exception as e:
            rec["reason_error"] = str(e)[:200]
        results["per_lang"][lang] = rec
        dump()
        log(f"{lang:4s}  CLA_en={rec.get('cla_en', float('nan')) or float('nan'):.3f}  "
            f"acc={rec.get('accuracy', float('nan')):.3f}  n={rec.get('n','-')}"
            + ("  [ALIGN FAIL]" if "align_error" in rec else "")
            + ("  [REASON FAIL]" if "reason_error" in rec else ""))

    dump()
    print("\n" + "=" * 70)
    pe = results.get("src_en_vs_accuracy_pearson", float("nan"))
    sp = results.get("src_en_vs_accuracy_spearman", float("nan"))
    print(f"RESULT [{args.dataset}]  English-pivot alignment vs reasoning accuracy:  "
          f"Pearson={pe:+.3f}  Spearman={sp:+.3f}  (n={results.get('n_langs_scored','?')} langs)")
    part = results.get("partial_spearman_en_controlling_resource")
    if part:
        print(f"  partial Spearman (controlling resource level): {part['r']:+.3f} "
              f"[{part['ci_low']:+.3f}, {part['ci_high']:+.3f}]")
    for p, s in results.get("pivots", {}).items():
        pr = s["pearson"]
        print(f"  pivot {p}: Pearson {pr['r']:+.3f} [{pr['ci_low']:+.3f}, {pr['ci_high']:+.3f}]  (n={pr['n']})")
    print(f"written to {args.out}   ({time.time()-t0:.0f}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
