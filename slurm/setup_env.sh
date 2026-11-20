#!/bin/bash
# Run ONCE on the LOGIN NODE, from the repo root:  bash slurm/setup_env.sh
set -e
module load python/3.11 gcc arrow
virtualenv --no-download "$PWD/.venv"
source "$PWD/.venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch numpy scipy pandas
pip install transformers accelerate datasets sacrebleu   # recent, from PyPI (login node has internet)
python -c "import torch,transformers,datasets,sacrebleu,scipy; \
print('ok | torch',torch.__version__,'| transformers',transformers.__version__)"
echo "env ready at $PWD/.venv"
