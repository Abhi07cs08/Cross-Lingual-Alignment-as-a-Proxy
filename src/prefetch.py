#!/usr/bin/env python3
"""Run ONCE on the LOGIN NODE (internet) before any job. Caches the model weights, the
FLORES+ configs, and the MGSM / AfriMGSM datasets into $HF_HOME so the compute nodes run
offline.

  python prefetch.py --task all                                  # MT + both reasoning sets
  python prefetch.py --task mgsm                                 # one task's data only
  python prefetch.py --models Qwen/Qwen3-14B-Base,google/gemma-3-12b-pt
  python prefetch.py --task all --skip-model                     # data only, weights cached

FLORES+ is gated on HuggingFace: export HF_TOKEN and accept the dataset terms first.
Gated models (gemma, Ministral) also need the terms accepted on their model page and a
valid HF_TOKEN. A few FAIL lines on obscure FLORES configs are fine; those langs are
skipped by the run too.
"""
import argparse, os
from run_repro import FLORES as MT_FLORES
from run_reason import MGSM, AFRIMGSM, DATASETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["all", "mt", "mgsm", "afrimgsm"], default="all")
    ap.add_argument("--models", default="Qwen/Qwen3-14B-Base",
                    help="comma-separated HF model ids to cache")
    ap.add_argument("--skip-model", action="store_true", help="skip all weight downloads")
    args = ap.parse_args()
    print("HF_HOME =", os.environ.get("HF_HOME", "(UNSET - export it to a scratch path!)"))

    from datasets import load_dataset

    flores = set()
    datasets_to_get = []
    if args.task in ("all", "mt"):
        flores |= set(MT_FLORES.values())
    if args.task in ("all", "mgsm"):
        flores |= {fl for (_, fl) in MGSM.values()}
        datasets_to_get.append(("mgsm", DATASETS["mgsm"]))
    if args.task in ("all", "afrimgsm"):
        flores |= {fl for (_, fl) in AFRIMGSM.values()}
        datasets_to_get.append(("afrimgsm", DATASETS["afrimgsm"]))

    for fl in sorted(flores):
        try:
            load_dataset("openlanguagedata/flores_plus", fl, split="devtest")
            print("  FLORES ok ", fl, flush=True)
        except Exception as e:
            print("  FLORES FAIL", fl, str(e)[:120], flush=True)

    for name, (ds_id, lmap) in datasets_to_get:
        for iso, (cfg, _) in lmap.items():
            try:
                load_dataset(ds_id, cfg, split="train")
                load_dataset(ds_id, cfg, split="test")
                print(f"  {name} ok  {cfg}", flush=True)
            except Exception as e:
                print(f"  {name} FAIL {cfg} {str(e)[:120]}", flush=True)

    if not args.skip_model:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        for m in [x for x in args.models.split(",") if x]:
            try:
                AutoTokenizer.from_pretrained(m)
                AutoModelForCausalLM.from_pretrained(m)
                print("  model cached", m, flush=True)
            except Exception as e:
                print("  model FAIL ", m, str(e)[:160], flush=True)
    print("prefetch done.")


if __name__ == "__main__":
    main()
