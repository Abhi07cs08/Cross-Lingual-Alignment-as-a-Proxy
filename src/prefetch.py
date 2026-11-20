#!/usr/bin/env python3
"""Run ONCE on the LOGIN NODE (internet) before any job. Caches the model, the FLORES+
configs, and the MGSM / AfriMGSM datasets into $HF_HOME so the compute nodes run offline.

  python prefetch.py --task all          # everything (MT + both reasoning sets)
  python prefetch.py --task mt           # MT baseline only
  python prefetch.py --task mgsm         # reasoning: MGSM only
  python prefetch.py --task afrimgsm     # reasoning: AfriMGSM only

FLORES+ is gated on HuggingFace: export HF_TOKEN and accept the dataset terms first.
"""
import argparse, os
from run_repro import FLORES as MT_FLORES
from run_reason import MGSM, AFRIMGSM, DATASETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["all", "mt", "mgsm", "afrimgsm"], default="all")
    ap.add_argument("--model", default="Qwen/Qwen3-14B-Base")
    ap.add_argument("--skip-model", action="store_true", help="if the weights are already cached")
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
        AutoTokenizer.from_pretrained(args.model)
        AutoModelForCausalLM.from_pretrained(args.model)
        print("  model cached", args.model, flush=True)
    print("prefetch done.")


if __name__ == "__main__":
    main()
