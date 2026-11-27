# Cluster runbook (DRAC)

The final robust run: every model x every task, full language sets, timeout-safe, with
bootstrap CIs and a resource-controlled partial correlation. Default cluster is **Narval**
(A100, `def-annielee`); Nibi (H100, `rrg-annielee`) overrides are noted inline.

## 0. One-time account setup (already done)
- SSH key added at ccdb.alliancecan.ca -> Manage SSH Keys; MFA (Duo) enrolled.
- Per-cluster access activated at ccdb.alliancecan.ca/me/access_systems (accept the
  agreements for Narval / Nibi). Login fails with "Permission denied" until this is done.

## 1. Get the repo onto the cluster (git, not scp)
Once the GitHub remote is set up (see `docs/SYNC.md`), the cluster just pulls:
```
# first time
cd ~/scratch && git clone <your repo url> cla-reasoning && cd cla-reasoning
# every later update
cd ~/scratch/cla-reasoning && git pull
```
No more re-uploading folders: edit anywhere, push, `git pull` on the cluster.

## 2. Build the environment (LOGIN NODE, once)
```
bash slurm/setup_env.sh
```
Makes `./.venv`: torch from the Alliance wheelhouse, recent transformers/datasets/sacrebleu
from PyPI. A couple of minutes.

## 3. Download models + data (LOGIN NODE, the only networked step)
```
export HF_HOME=~/scratch/hfcache        # one cache, reused by every job
export HF_TOKEN=hf_yourtoken            # FLORES+ is gated; gated models need it too
source .venv/bin/activate
python src/prefetch.py --task all --models Qwen/Qwen3-14B-Base
```
Add gated models to `--models` (comma-separated) only after accepting their terms on the HF
model page while logged in. A few `FAIL` lines on obscure FLORES configs are fine.

## 4. Dispatch the full matrix (LOGIN NODE)
```
export HF_HOME=~/scratch/hfcache
bash slurm/dispatch.sh
```
Defaults: Narval, `def-annielee`, `a100:1`, 24 h, model list from `slurm/models.txt`,
tasks `mgsm afrimgsm mt`. Override anything via env:
```
MODELS="Qwen/Qwen3-14B-Base google/gemma-3-12b-pt" TASKS="mgsm afrimgsm" bash slurm/dispatch.sh
# Nibi instead of Narval:
ACCOUNT=rrg-annielee GPU=h100:1 bash slurm/dispatch.sh
```
Each (model, task) is its own job, so one failure never blocks the rest. `squeue -u $USER`
to watch; `PD` queued, `R` running; emails fire on begin/end/fail.

A single quick re-run without the matrix:
```
sbatch slurm/submit_mgsm.sh        # or submit_afri.sh / submit_mt.sh
```

## 5. Watch
```
tail -f results/slurm-cla-mgsm-*.log
```
Each language prints `CLA_en=... acc=...`. The reasoning jobs write the JSON after every
language and the MT job after every translation pair, so a timeout keeps all finished work.

## 6. Collect + summarize (any node, no GPU)
```
python src/analyze.py            # reads results/, writes results/summary.md + summary.csv
```
That table (per model/task: Pearson, Spearman, the resource-controlled partial correlation,
and the per-pivot correlations) is the input for the write-up. Commit `results/` and push.

## What "robust / max settings" means here
- Reasoning: all MGSM (11) and AfriMGSM (17) languages, 8-shot native CoT, 250 problems/lang,
  `n_align=800` FLORES sentences for alignment, answer extraction that handles non-Latin AND
  CJK numerals (fixes the Chinese artifact).
- MT: full 28-language set, 7 targets, `n_align=500`, 300 sentences/pair.
- Stats: bootstrap 95% CIs on every correlation; partial Spearman controlling for a
  resource-level proxy (the main confound); all five pivots scored (en, es, ru, zh, sw) so
  "is English special?" comes straight out of the run.

## Gotchas (all hit and solved once)
- Narval SSH drops on long downloads: run prefetch inside `tmux` so it survives disconnects.
- `module load python/3.11 gcc arrow` AND `source .venv/bin/activate` in every interactive
  shell; `arrow` provides pyarrow for `datasets`.
- `HF_HOME` must be exported the same in prefetch and in jobs (submit scripts default it to
  `$SLURM_SUBMIT_DIR/.hfcache`; set it explicitly if your cache lives elsewhere).
- Nibi needs an explicit GPU type (`a100:1` / `h100:1`, not `=1`); FLORES Chinese config is
  `cmn_Hans`, not `zho_Hans`.
- Lab support: Armin Sobhani, asobhani@sharcnet.ca.
