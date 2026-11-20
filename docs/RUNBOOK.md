# Cluster runbook (DRAC / Nibi)

Everything runs on the Digital Research Alliance of Canada. Account `rrg-annielee` (Lee
Lab, Nibi). Username `abhinn`. One full H100 per job.

## 0. One-time account setup (already done)
- SSH key added at ccdb.alliancecan.ca -> My Account -> Manage SSH Keys.
- MFA (Duo) enrolled.
- Per-cluster access activated at ccdb.alliancecan.ca/me/access_systems (accept the
  agreements for Nibi / Narval). Login fails with "Permission denied" until this is done,
  even with a good key.

## 1. Get the repo onto the cluster
```
scp -i ~/.ssh/ccdb -r cla-reasoning abhinn@nibi.alliancecan.ca:scratch/
ssh -i ~/.ssh/ccdb abhinn@nibi.alliancecan.ca
cd ~/scratch/cla-reasoning
```

## 2. Build the environment (LOGIN NODE)
```
bash slurm/setup_env.sh
```
Makes `./.venv`: torch from the Alliance wheelhouse (CUDA-matched), recent
transformers/datasets/sacrebleu from PyPI. A couple of minutes.

## 3. Download model + data (LOGIN NODE - the only networked step)
```
export HF_HOME=~/scratch/iee/hf     # reuse the already-cached Qwen3-14B-Base; or pick a new path
export HF_TOKEN=hf_yourtoken        # FLORES+ is a gated dataset
source .venv/bin/activate
python src/prefetch.py --task all --skip-model
```
Drop `--skip-model` if the weights are not already cached at `$HF_HOME` (that pulls ~28 GB).
A few `FAIL` lines on obscure FLORES configs are fine; those languages are skipped.

## 4. Dispatch jobs (from the repo root)
```
sbatch slurm/submit_mgsm.sh
sbatch slurm/submit_afri.sh
sbatch slurm/submit_mt.sh      # optional: re-run the MT baseline
squeue -u $USER
```
`PD` = queued, `R` = running. Emails fire on begin/end/fail.

## 5. Watch
```
tail -f results/slurm-mgsm-*.log
```
Each language prints `CLA_en=... acc=...` as it finishes. The RESULT line at the end gives
the alignment-vs-accuracy correlation.

## 6. Collect
```
cat results/reason_mgsm_qwen3-14b-base.json
cat results/reason_afrimgsm_qwen3-14b-base.json
```
Copy these back and commit them to `results/`.

## Gotchas (all already hit and solved once)
- Nibi needs an explicit GPU type: `--gpus-per-node=h100:1`, not `=1`.
- FLORES Chinese config is `cmn_Hans`, not `zho_Hans`.
- Load `module load python/3.11 gcc arrow` AND `source .venv/bin/activate` in every
  interactive shell before running python; `arrow` provides pyarrow for `datasets`.
- The reasoning jobs write results per-language, so a timeout never costs the whole run.
- Lab support: Armin Sobhani, asobhani@sharcnet.ca.
