#!/usr/bin/env python3
"""Aggregate every results/*.json from the run into one table for the write-up.

Reads whatever is in results/ (any model, any task), pulls the headline numbers, and
writes a Markdown summary plus a flat CSV. Safe to run mid-run: it just reports whatever
has been written so far. No GPU, no network.

  python src/analyze.py                      # reads ./results, writes results/summary.md + .csv
  python src/analyze.py --results-dir path   # custom dir
"""
from __future__ import annotations
import argparse, csv, glob, json, os


def _fmt(x, nd=3):
    try:
        return f"{float(x):+.{nd}f}"
    except Exception:
        return "  .  "


def _ci(d):
    if not isinstance(d, dict):
        return ""
    lo, hi = d.get("ci_low"), d.get("ci_high")
    if lo is None or hi is None or lo != lo or hi != hi:
        return ""
    return f" [{float(lo):+.2f}, {float(hi):+.2f}]"


def load(results_dir):
    reason, mt = [], []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get("dataset") in ("mgsm", "afrimgsm"):
            reason.append((os.path.basename(path), d))
        elif "src_en_pearson" in d or "src_tgt_pearson" in d:
            mt.append((os.path.basename(path), d))
    return reason, mt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    reason, mt = load(args.results_dir)

    lines = ["# Run summary", ""]
    rows = []  # for CSV

    if reason:
        lines += ["## Reasoning arm (alignment to English vs accuracy)", "",
                  "| model | dataset | n langs | Pearson | Spearman | partial Spearman (resource-controlled) |",
                  "|---|---|---|---|---|---|"]
        for name, d in reason:
            part = d.get("partial_spearman_en_controlling_resource", {})
            lines.append(f"| {d.get('model','?')} | {d['dataset']} | {d.get('n_langs_scored','?')} | "
                         f"{_fmt(d.get('src_en_vs_accuracy_pearson'))} | "
                         f"{_fmt(d.get('src_en_vs_accuracy_spearman'))} | "
                         f"{_fmt(part.get('r'))}{_ci(part)} |")
            rows.append({"file": name, "model": d.get("model"), "task": d["dataset"],
                         "pearson": d.get("src_en_vs_accuracy_pearson"),
                         "spearman": d.get("src_en_vs_accuracy_spearman"),
                         "partial_spearman_resource": part.get("r")})
        # per-pivot block (is English special?)
        lines += ["", "### Alignment-to-pivot vs accuracy, by pivot (Pearson [95% CI])", ""]
        for name, d in reason:
            piv = d.get("pivots", {})
            if not piv:
                continue
            cells = "  ".join(f"{p}:{_fmt(s['pearson']['r'],2)}{_ci(s['pearson'])}"
                              for p, s in piv.items())
            lines.append(f"- **{d.get('model','?')} / {d['dataset']}**: {cells}")
        lines.append("")

    if mt:
        lines += ["## MT arm (English pivot vs target pivot)", "",
                  "| model | src-en Pearson | src-tgt Pearson | pivots (Pearson) |",
                  "|---|---|---|---|"]
        for name, d in mt:
            pivs = d.get("pivots", [])
            cells = "  ".join(f"{p}:{_fmt(d.get(f'src_{p}_pearson'),2)}" for p in pivs)
            lines.append(f"| {d.get('model','?')} | {_fmt(d.get('src_en_pearson'))} | "
                         f"{_fmt(d.get('src_tgt_pearson'))} | {cells} |")
            rows.append({"file": name, "model": d.get("model"), "task": "mt",
                         "src_en_pearson": d.get("src_en_pearson"),
                         "src_tgt_pearson": d.get("src_tgt_pearson")})

    if not reason and not mt:
        lines.append("_No result JSONs found yet._")

    md = "\n".join(lines) + "\n"
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "summary.md"), "w") as f:
        f.write(md)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(args.results_dir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(md)
    print(f"[written] {args.results_dir}/summary.md" + (f" and summary.csv" if rows else ""))


if __name__ == "__main__":
    main()
