"""Spatial smoothing operators applied to leave-one-out (LOO) evidence maps.

`ppr_smooth` is the SpatialShap operator. It is personalised PageRank on the
4-connected patch grid, solved in closed form:

    p = (1 - a) p0 + a W^T p,   W_ij = 1/deg(i) for j in N(i)

W is row-stochastic, so W^T preserves total mass, and summing both sides
gives sum(p) = (1 - a) sum(p0) + a sum(p), i.e. sum(p) = sum(p0). This is the
mass-preservation property stated in the paper; see tests/test_smoothing.py.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter


@lru_cache(maxsize=16)
def grid_transition(g: int) -> np.ndarray:
    """Row-stochastic random-walk matrix W of the g x g 4-connected grid."""
    n = g * g
    w = np.zeros((n, n))
    for r in range(g):
        for c in range(g):
            i = r * g + c
            nbrs = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 0 <= r + dr < g and 0 <= c + dc < g]
            for rr, cc in nbrs:
                w[i, rr * g + cc] = 1.0 / len(nbrs)
    return w


@lru_cache(maxsize=64)
def _ppr_operator(g: int, alpha: float) -> np.ndarray:
    w = grid_transition(g)
    return (1.0 - alpha) * np.linalg.inv(np.eye(g * g) - alpha * w.T)


def ppr_smooth(x: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """SpatialShap: exact personalised-PageRank smoothing. Preserves sum(x)."""
    x = np.asarray(x, dtype=np.float64)
    g = x.shape[0]
    return (_ppr_operator(g, float(alpha)) @ x.ravel()).reshape(x.shape)


def ppr_smooth_iter(x: np.ndarray, alpha: float = 0.85, steps: int = 20) -> np.ndarray:
    """The truncated iteration used by the original eval_evidgeo.py (for comparison)."""
    x = np.asarray(x, dtype=np.float64)
    g = x.shape[0]
    wt = grid_transition(g).T
    p0 = x.ravel()
    p = p0.copy()
    for _ in range(steps):
        p = (1.0 - alpha) * p0 + alpha * (wt @ p)
    return p.reshape(x.shape)


def gaussian_smooth(x: np.ndarray, sigma: float, renormalize: bool = True) -> np.ndarray:
    """Gaussian blur baseline on the patch grid.

    With zero padding, mass that would land outside the grid is lost. When
    `renormalize` is True the output is rescaled to the input's total, which
    is the fair comparison; the raw version is kept to quantify the leak.
    """
    x = np.asarray(x, dtype=np.float64)
    y = gaussian_filter(x, sigma=sigma, mode="constant", cval=0.0)
    if renormalize and y.sum() > 0:
        y *= x.sum() / y.sum()
    return y


def mass_leak(x: np.ndarray, sigma: float) -> float:
    """Fraction of mass a zero-padded Gaussian blur pushes off the grid."""
    x = np.asarray(x, dtype=np.float64)
    if x.sum() <= 0:
        return 0.0
    y = gaussian_filter(x, sigma=sigma, mode="constant", cval=0.0)
    return float(1.0 - y.sum() / x.sum())
