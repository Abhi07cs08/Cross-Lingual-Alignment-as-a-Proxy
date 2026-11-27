#!/bin/bash
# Dispatch the full matrix: every model x every task, one SLURM job each.
# Run from the repo root on the LOGIN NODE, after setup_env.sh and prefetch.py.
#
#   bash slurm/dispatch.sh                 # defaults below
#   MODELS="Qwen/Qwen3-14B-Base google/gemma-3-12b-pt" TASKS="mgsm afrimgsm" bash slurm/dispatch.sh
#
# Cluster defaults are Narval (A100, def-annielee). For Nibi (H100) set:
#   ACCOUNT=rrg-annielee GPU=h100:1 bash slurm/dispatch.sh
set -e
cd "$(dirname "$0")/.."

: "${ACCOUNT:=def-annielee}"
: "${GPU:=a100:1}"
: "${TIME:=24:00:00}"
# MODELS: space-separated HF ids. Default is the one ungated, already-cached model.
# Add gated ones (gemma, Ministral) only after accepting their terms + caching via prefetch.
: "${MODELS:=Qwen/Qwen3-14B-Base}"
# If slurm/models.txt exists and MODELS was not overridden, read models from it (one per line, # comments ok).
if [ "${MODELS}" = "Qwen/Qwen3-14B-Base" ] && [ -f slurm/models.txt ]; then
  FROMFILE=$(grep -vE '^\s*(#|$)' slurm/models.txt | tr '\n' ' ')
  [ -n "$FROMFILE" ] && MODELS="$FROMFILE"
fi
: "${TASKS:=mgsm afrimgsm mt}"

echo "ACCOUNT=$ACCOUNT GPU=$GPU TIME=$TIME"
echo "MODELS=$MODELS"
echo "TASKS=$TASKS"
for m in $MODELS; do
  for t in $TASKS; do
    echo ">> submitting  $m  /  $t"
    sbatch --account="$ACCOUNT" --gpus-per-node="$GPU" --time="$TIME" \
           --job-name="cla-$t" slurm/submit_run.sh "$m" "$t"
  done
done
echo "---"
squeue -u "$USER"
