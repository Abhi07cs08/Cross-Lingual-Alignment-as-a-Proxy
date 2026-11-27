"""Correlation statistics for the alignment-vs-performance analysis.

Everything here resamples or residualises at the LANGUAGE level, because a
language is the unit of observation (one alignment value, one accuracy per
language). Two things the baseline run did not have:

  - bootstrap confidence intervals, so a correlation comes with an interval
    instead of a bare point estimate (the language samples are small);
  - partial correlation controlling for a resource-level proxy, which is the
    main confound: alignment and accuracy both rise with how much data a
    language has, so a raw correlation cannot separate the two.

No new dependencies: numpy + scipy only.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata


def _stat(x: np.ndarray, y: np.ndarray, method: str) -> float:
    fn = pearsonr if method == "pearson" else spearmanr
    return float(fn(x, y).statistic)


def corr_with_ci(x, y, method: str = "pearson", n_boot: int = 5000,
                 seed: int = 0, ci: float = 95.0) -> dict:
    """Point estimate + percentile bootstrap CI, resampling language points.

    Returns nan bounds (not an error) when there are too few points or no
    variance in a resample, so a thin language set never crashes the run.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    out = {"method": method, "n": int(n), "r": float("nan"),
           "ci_low": float("nan"), "ci_high": float("nan")}
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        if n >= 2 and np.std(x) > 0 and np.std(y) > 0:
            out["r"] = _stat(x, y, method)
        return out
    out["r"] = _stat(x, y, method)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        boots.append(_stat(xb, yb, method))
    if boots:
        lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
        out["ci_low"], out["ci_high"] = float(lo), float(hi)
        out["n_boot"] = len(boots)
    return out


def _residualise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Residual of a after a linear fit on b (b is 1-D). Used on ranks."""
    design = np.c_[np.ones_like(b, dtype=float), b.astype(float)]
    coef, *_ = np.linalg.lstsq(design, a.astype(float), rcond=None)
    return a - design @ coef


def partial_spearman(x, y, z) -> float:
    """Spearman partial correlation of x and y controlling for z.

    Done the standard way: rank all three, linearly remove z's rank from the
    ranks of x and y, then correlate the residuals. This holds resource level
    (z) fixed so what is left is alignment's association with accuracy beyond
    what resource level already explains.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    if len(x) < 4:
        return float("nan")
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex, ey = _residualise(rx, rz), _residualise(ry, rz)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return float("nan")
    return float(pearsonr(ex, ey).statistic)


def partial_spearman_with_ci(x, y, z, n_boot: int = 5000, seed: int = 0,
                             ci: float = 95.0) -> dict:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    n = len(x)
    out = {"n": int(n), "r": partial_spearman(x, y, z),
           "ci_low": float("nan"), "ci_high": float("nan")}
    if n < 5:
        return out
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = partial_spearman(x[idx], y[idx], z[idx])
        if r == r:  # not nan
            boots.append(r)
    if boots:
        lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
        out["ci_low"], out["ci_high"] = float(lo), float(hi)
        out["n_boot"] = len(boots)
    return out
