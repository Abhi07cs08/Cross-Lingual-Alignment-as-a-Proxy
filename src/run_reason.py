#!/usr/bin/env python3
"""Reasoning arm: does cross-lingual alignment predict multilingual reasoning accuracy?

For each language: measure alignment to English (same p-wm + ANC + max-layer measure as
the MT baseline, on FLORES), and measure reasoning accuracy on MGSM / AfriMGSM with
chain-of-thought (8-shot native CoT, greedy). Then correlate alignment-to-English vs
accuracy across languages (Pearson + Spearman).

Robustness: results are written to the JSON after EVERY language, so a walltime timeout
loses only the language in progress, never the whole run. Any language whose data fails
to load is skipped with a logged note, not crashed on.

Offline-safe: run prefetch_reason.py on the login node first. Reuses cla.py.
"""
from __future__ import annotations
import argparse, json, os, re, time, unicodedata
import numpy as np
from scipy.stats import pearsonr, spearmanr
from cla import encode, cla_pair

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


def flores_sents(fl, n):
    from datasets import load_dataset
    ds = load_dataset("openlanguagedata/flores_plus", fl, split="devtest")
    col = "text" if "text" in ds.column_names else ds.column_names[0]
    return [r[col] for r in ds][:n]


def load_split(ds_id, cfg, split):
    from datasets import load_dataset
    return list(load_dataset(ds_id, cfg, split=split))


def to_ascii_digits(s):
    out = []
    for ch in s:
        if ch.isdigit():
            d = unicodedata.digit(ch, None)
            out.append(str(d) if d is not None else ch)
        else:
            out.append(ch)
    return "".join(out)


def extract_answer(text):
    """Last integer in the generation, after normalising non-Latin digits and separators."""
    t = to_ascii_digits(text)
    t = re.sub(r"(?<=\d)[,\s](?=\d)", "", t)
    nums = re.findall(r"-?\d+", t)
    try:
        return int(nums[-1]) if nums else None
    except Exception:
        return None


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
            enc = tok(prompts[i:i+batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=4096).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for txt in tok.batch_decode(new, skip_special_tokens=True):
                cut = re.split(r"\n\s*(?:Question|Q)\s*[:\.]", txt)[0]
                out.append(cut)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mgsm", "afrimgsm"], default="mgsm")
    ap.add_argument("--model", default="Qwen/Qwen3-14B-Base")
    ap.add_argument("--langs", default="")
    ap.add_argument("--n-align", type=int, default=300)
    ap.add_argument("--n-reason", type=int, default=250)
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="results/reason.json")
    args = ap.parse_args()

    ds_id, LMAP = DATASETS[args.dataset]
    langs = [l for l in args.langs.split(",") if l] or list(LMAP)
    assert "en" in langs, "keep 'en' in --langs (English is the pivot and the FLORES anchor)"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    log(f"dataset={args.dataset} model={args.model} langs={langs} "
        f"n_align={args.n_align} n_reason={args.n_reason}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    log(f"model loaded ({time.time()-t0:.0f}s)")

    en_reps = encode(model, tok, flores_sents("eng_Latn", args.n_align), batch_size=16)

    results = {"dataset": args.dataset, "model": args.model, "measure": "anc",
               "n_align": args.n_align, "n_reason": args.n_reason, "per_lang": {}}

    def dump():
        rows = {l: v for l, v in results["per_lang"].items()
                if "cla_en" in v and "accuracy" in v and l != "en"}
        if len(rows) >= 3:
            x = [v["cla_en"] for v in rows.values()]
            y = [v["accuracy"] for v in rows.values()]
            results["src_en_vs_accuracy_pearson"] = float(pearsonr(x, y).statistic)
            results["src_en_vs_accuracy_spearman"] = float(spearmanr(x, y).statistic)
            results["n_langs_scored"] = len(rows)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    for lang in langs:
        cfg, fl = LMAP[lang]
        rec = {}
        try:
            reps = en_reps if lang == "en" else encode(model, tok, flores_sents(fl, args.n_align), batch_size=16)
            rec["cla_en"] = 1.0 if lang == "en" else cla_pair(reps, en_reps, "anc")
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
        log(f"{lang:4s}  CLA_en={rec.get('cla_en', float('nan')):.3f}  "
            f"acc={rec.get('accuracy', float('nan')):.3f}  n={rec.get('n','-')}"
            + ("  [ALIGN FAIL]" if "align_error" in rec else "")
            + ("  [REASON FAIL]" if "reason_error" in rec else ""))

    dump()
    print("\n" + "=" * 60)
    pe = results.get("src_en_vs_accuracy_pearson", float("nan"))
    sp = results.get("src_en_vs_accuracy_spearman", float("nan"))
    print(f"RESULT [{args.dataset}]  alignment-to-English vs reasoning accuracy:  "
          f"Pearson={pe:+.3f}  Spearman={sp:+.3f}  (n={results.get('n_langs_scored','?')} langs)")
    print(f"written to {args.out}   ({time.time()-t0:.0f}s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
