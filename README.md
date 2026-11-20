# Cross-lingual alignment as a label-free proxy for multilingual reasoning

Does a cheap, label-free measure of a model's internal **cross-lingual alignment** predict
how well it performs on multilingual tasks, especially reasoning, and especially in
low-resource languages? Framed as a ProxyLM-style performance-prediction study.

The alignment score is computed with forward passes only (no labels, no training): take
parallel sentences (FLORES-200), pool each into one vector per layer with a
position-weighted mean, and measure how aligned two languages' representations are with
the **ANC** measure (average absolute per-neuron correlation) read at the max-alignment
layer. This follows Al Ali et al. 2026 ("Is English Enough?", arXiv:2608.03446) and
MEXA (Kargaran et al. 2025, arXiv:2410.05873).

## The two arms

**MT baseline (`src/run_repro.py`)** - validation. Reproduces the paper's English-pivot:
alignment to English predicts translation quality (chrF++) better than alignment to the
target. This proves the alignment measure is implemented correctly before it is used to
predict anything. *Status: reproduced on Qwen3-14B-Base, src-en Pearson 0.75 vs src-tgt
0.69 (excluding the Hindi target, which failed to generate), inside the paper's 0.63-0.88
band. See `figures/pivot_result.png`.*

**Reasoning arm (`src/run_reason.py`)** - the contribution. For each language it measures
alignment to English and reasoning accuracy on MGSM (`juletxara/mgsm`, 11 langs) or
AfriMGSM (`masakhane/afrimgsm`, 17 African langs) with 8-shot chain-of-thought, then
correlates alignment-to-English against accuracy across languages. Tests whether the
English-pivot that holds for translation also holds for reasoning, and whether it holds
within low-resource languages. *Status: running.*

## Layout

```
src/        cla.py          shared alignment core (pooling + ANC + retrieval + max-layer)
            translate.py    5-shot translation + chrF++
            run_repro.py    MT baseline entry point
            run_reason.py   reasoning arm entry point
            prefetch.py     login-node download of model + FLORES + MGSM/AfriMGSM
slurm/      setup_env.sh    build the venv (login node)
            submit_mt.sh    MT baseline job
            submit_mgsm.sh  reasoning on MGSM
            submit_afri.sh  reasoning on AfriMGSM
results/    JSON outputs (copy run outputs here)
figures/    pivot_result.png + its generator
docs/       RUNBOOK.md       full step-by-step cluster guide
```

## Quick start (DRAC / Nibi)

From the repo root on the **login node**:

```
bash slurm/setup_env.sh                       # build ./.venv (once)
export HF_HOME=~/scratch/iee/hf               # reuse the cached model; else it re-downloads
export HF_TOKEN=hf_...                         # FLORES+ is gated
source .venv/bin/activate
python src/prefetch.py --task all --skip-model # download data (model already cached)
```

Then dispatch jobs (still from the repo root):

```
sbatch slurm/submit_mgsm.sh     # reasoning on MGSM
sbatch slurm/submit_afri.sh     # reasoning on AfriMGSM
sbatch slurm/submit_mt.sh       # re-run / extend the MT baseline
```

Each writes a JSON to `results/` and logs to `results/slurm-*.log`. The reasoning jobs
write after every language, so a walltime timeout keeps all completed languages.

See `docs/RUNBOOK.md` for the full walkthrough, including SSH/MFA and the gotchas.
