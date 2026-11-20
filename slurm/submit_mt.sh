#!/bin/bash
#SBATCH --job-name=cla-mt
#SBATCH --account=rrg-annielee
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-mt-%j.log
#SBATCH --mail-user=abhinnkaushik@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
# run from the repo root:  sbatch slurm/submit_mt.sh
set -e
cd "$SLURM_SUBMIT_DIR"
module load python/3.11 gcc arrow
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hfcache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd src
python run_repro.py --model Qwen/Qwen3-14B-Base \
  --langs en,de,es,fr,ru,zh,hi,sw,bn,ta,th,tr --targets es,zh,sw,hi,ru \
  --n-align 500 --n-translate 200 --batch-size 16 \
  --out ../results/mt_baseline_qwen3-14b-base.json
