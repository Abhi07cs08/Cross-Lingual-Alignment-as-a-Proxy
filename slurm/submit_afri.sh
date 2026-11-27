#!/bin/bash
#SBATCH --job-name=cla-afri
#SBATCH --account=def-annielee
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm-afrimgsm-%j.log
#SBATCH --mail-user=abhinnkaushik@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
# run from the repo root:  sbatch slurm/submit_afri.sh
set -e
cd "$SLURM_SUBMIT_DIR"
module load python/3.11 gcc arrow
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hfcache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd src
python run_reason.py --dataset afrimgsm --model Qwen/Qwen3-14B-Base \
  --batch-size 8 \
  --out ../results/reason_afrimgsm_qwen3-14b-base.json
