#!/bin/bash
# Generic one-job runner: one (model, task) per job. Normally launched by dispatch.sh,
# which sets --account / --gpus-per-node / --time per cluster. Can also be run directly:
#   sbatch --account=def-annielee --gpus-per-node=a100:1 --time=24:00:00 \
#          slurm/submit_run.sh Qwen/Qwen3-14B-Base mgsm
#SBATCH --job-name=cla
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=results/slurm-%x-%j.log
#SBATCH --mail-user=abhinnkaushik@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
set -e
cd "$SLURM_SUBMIT_DIR"
module load python/3.11 gcc arrow
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hfcache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

MODEL="$1"; TASK="$2"
if [ -z "$MODEL" ] || [ -z "$TASK" ]; then echo "usage: submit_run.sh <model> <mt|mgsm|afrimgsm>"; exit 2; fi
slug=$(echo "$MODEL" | tr '/:' '__' | tr '[:upper:]' '[:lower:]')
echo "model=$MODEL task=$TASK slug=$slug"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd src
case "$TASK" in
  mt)        python run_repro.py  --model "$MODEL" --out "../results/mt_${slug}.json" ;;
  mgsm)      python run_reason.py --dataset mgsm     --model "$MODEL" --out "../results/reason_mgsm_${slug}.json" ;;
  afrimgsm)  python run_reason.py --dataset afrimgsm --model "$MODEL" --out "../results/reason_afrimgsm_${slug}.json" ;;
  *) echo "unknown task $TASK"; exit 2 ;;
esac
