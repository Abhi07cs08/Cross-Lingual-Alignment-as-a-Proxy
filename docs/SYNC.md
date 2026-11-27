# Keeping the cluster in sync with GitHub

Goal: stop re-uploading folders. Edit the code anywhere, push to GitHub once, and `git pull`
on Narval. GitHub is the single source of truth; the cluster is just a checkout.

## One-time setup

### 1. The GitHub repo
Private repo under your account (e.g. `cla-reasoning`). The working copy here already has a
clean commit history and a `.gitignore` that excludes caches, weights, logs, and the
`ref/` clone, so nothing heavy or secret gets pushed.

### 2. Push the code up (from wherever the repo lives)
```
git remote add origin git@github.com:<you>/cla-reasoning.git   # or the https url
git push -u origin main
```

### 3. Let Narval read the private repo
Pick one:
- **Deploy key (cleanest, read-only):** on Narval, `ssh-keygen -t ed25519 -f ~/.ssh/gh_cla`,
  then add `~/.ssh/gh_cla.pub` as a *deploy key* on the repo (Settings -> Deploy keys).
  Clone with that key. Read-only, scoped to this one repo, nothing else exposed.
- **Fine-grained PAT:** a token with read-only "Contents" on just this repo, used in the
  https clone URL. Easy, but it is a secret on the cluster; a deploy key is tidier.

### 4. Clone on Narval
```
cd ~/scratch
git clone git@github.com:<you>/cla-reasoning.git
cd cla-reasoning
```

## Day-to-day loop
```
# edit + commit + push from here (or your laptop)
git add -A && git commit -m "…" && git push

# on Narval, before a run
cd ~/scratch/cla-reasoning && git pull
bash slurm/dispatch.sh
```

## Getting results back
After jobs finish, commit the JSONs from the cluster and pull them anywhere:
```
# on Narval
git add results/*.json && git commit -m "run results" && git push
```
`results/*.json` is intentionally tracked; weights, caches, `.venv`, and logs are ignored.

## What stays out of git
See `.gitignore`: `.venv/`, `.hfcache/`, `*.log`, model weights, `ref/`, and any HF token.
Never commit `HF_TOKEN`. If a token was ever pasted into a shell, rotate it.
