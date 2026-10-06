"""Concentration and divergence measures for evidence maps.

Every function accepts a non-negative map of any shape and treats it as a
flat distribution over patches.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def normalize(x: np.ndarray) -> np.ndarray:
    """Clip negatives and l1-normalise. An all-zero map becomes uniform."""
    x = np.clip(np.asarray(x, dtype=np.float64).ravel(), 0.0, None)
    s = x.sum()
    if s <= EPS:
        return np.full_like(x, 1.0 / x.size)
    return x / s


def gini(x: np.ndarray) -> float:
    """Gini coefficient in [0, 1 - 1/n]; 0 = uniform, larger = more concentrated."""
    x = np.sort(normalize(x))
    n = x.size
    idx = np.arange(1, n + 1)
    return float(2.0 * np.sum(idx * x) / n - (n + 1.0) / n)


def entropy(x: np.ndarray) -> float:
    """Shannon entropy in nats."""
    p = normalize(x)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def norm_entropy(x: np.ndarray) -> float:
    """Entropy divided by log(n), so grids of different size are comparable."""
    n = np.asarray(x).size
    return entropy(x) / np.log(n)


def top_mass(x: np.ndarray, frac: float = 0.10) -> float:
    """Share of total mass held by the top `frac` of patches (at least one patch)."""
    p = np.sort(normalize(x))[::-1]
    k = max(1, int(round(frac * p.size)))
    return float(p[:k].sum())


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats (bounded by log 2)."""
    p, q = normalize(p), normalize(q)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def concentration(x: np.ndarray, prefix: str) -> dict:
    """All per-map scalars under a common prefix, e.g. {'pec_gini': ...}."""
    return {
        f"{prefix}_gini": gini(x),
        f"{prefix}_entropy": norm_entropy(x),
        f"{prefix}_top10": top_mass(x, 0.10),
    }


def mass_in_mask(x: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of evidence mass falling inside a binary patch mask (same shape)."""
    p = normalize(x)
    return float(p[np.asarray(mask, dtype=bool).ravel()].sum())


def peak_hit(x: np.ndarray, mask: np.ndarray) -> bool:
    """Whether the single highest-evidence patch lies inside the mask."""
    return bool(np.asarray(mask, dtype=bool).ravel()[int(np.argmax(np.asarray(x).ravel()))])
