"""Cross-lingual alignment (CLA) measures, following Al Ali et al. (2026),
"Is English Enough?" (arXiv:2608.03446), code: github.com/KathyHaem/cla-metrics.

Two measures, both read at the max-alignment layer (selected on the same FLORES data,
as in the paper, following Kargaran et al. 2025 / MEXA):
  - ANC:  average neuron-wise Pearson correlation over parallel sentences (system-level).
  - abs:  xSIM-style cosine retrieval accuracy (the "abs" variant).

Sentence embeddings use the position-weighted mean (p-wm): weight token t by its
position index (Muennighoff, 2022), so later tokens count more.

This is a clean-room reimplementation written from the paper definitions, unit-tested
against scipy, NOT the upstream repo imported.
"""
from __future__ import annotations

from typing import List
import numpy as np


# ---------- pooling ----------
def pwm_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Position-weighted mean of token states. hidden [b,s,d], mask [b,s] -> [b,d]."""
    hidden = np.asarray(hidden, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    b, s, _ = hidden.shape
    w = mask * np.arange(1, s + 1, dtype=np.float64)[None, :]   # weight = position index
    denom = np.clip(w.sum(1, keepdims=True), 1e-9, None)
    return (hidden * w[:, :, None]).sum(1) / denom


def encode(model, tokenizer, sentences: List[str], batch_size: int = 16,
           max_length: int = 128, device: str | None = None) -> np.ndarray:
    """Per-layer p-wm sentence reps: [n_layers+1, n_sentences, dim]. Imports torch lazily."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    chunks = None
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            enc = tokenizer(sentences[i:i + batch_size], return_tensors="pt",
                            padding=True, truncation=True, max_length=max_length).to(device)
            hs = model(**enc, output_hidden_states=True).hidden_states
            mask = enc["attention_mask"].cpu().numpy()
            pooled = [pwm_pool(h.float().cpu().numpy(), mask) for h in hs]
            if chunks is None:
                chunks = [[] for _ in pooled]
            for li, p in enumerate(pooled):
                chunks[li].append(p)
    return np.stack([np.concatenate(c, 0) for c in chunks], 0).astype(np.float32)


# ---------- measures ----------
def _cos_matrix(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return a @ b.T


def retrieval_abs(a: np.ndarray, b: np.ndarray) -> float:
    """xSIM-style two-directional P@1: the parallel pair is each other's nearest neighbour."""
    C = _cos_matrix(np.asarray(a, float), np.asarray(b, float))
    diag = np.diag(C).copy()
    row = C.copy(); np.fill_diagonal(row, -np.inf)
    col = C.T.copy(); np.fill_diagonal(col, -np.inf)
    hit = (diag > row.max(1)) & (diag > col.max(1))
    return float(hit.mean())


def anc(a: np.ndarray, b: np.ndarray) -> float:
    """Average neuron-wise correlation (Del & Fishel 2022), as Al Ali's code computes it.

    Their anc_scoring.py: center each column, take the per-neuron cosine over the sentence
    axis (which equals Pearson r after centering), then return the mean of the ABSOLUTE
    values: `score.abs().mean()`. So this is mean |r| over neurons, not signed mean.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    ac = a - a.mean(0, keepdims=True); bc = b - b.mean(0, keepdims=True)
    sa = np.sqrt((ac ** 2).sum(0)); sb = np.sqrt((bc ** 2).sum(0))
    ok = (sa > 0) & (sb > 0)
    if not ok.any():
        return float("nan")
    r = (ac[:, ok] * bc[:, ok]).sum(0) / (sa[ok] * sb[ok])
    return float(np.abs(r).mean())   # abs per neuron, matching their code


_MEASURES = {"anc": anc, "abs": retrieval_abs}


def cla_pair(reps_a: np.ndarray, reps_b: np.ndarray, measure: str = "anc",
             skip_embedding_layer: bool = False) -> float:
    """CLA between two languages: the measure at its max-alignment layer ("HIGHEST").

    reps_a, reps_b: [n_layers+1, n, dim], row-aligned (parallel sentences).
    Matches their get_alignment(..., layer="HIGHEST") = max over ALL stored layers
    (their save_embeds keeps every hidden_states layer incl. the embedding layer, so
    the default here does not skip it).
    """
    fn = _MEASURES[measure]
    start = 1 if (skip_embedding_layer and reps_a.shape[0] > 1) else 0
    vals = [fn(reps_a[L], reps_b[L]) for L in range(start, reps_a.shape[0])]
    return float(np.nanmax(vals))
