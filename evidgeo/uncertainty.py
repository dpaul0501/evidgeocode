"""Uncertainty of the generation process, read off evidence maps.

Three sources, kept separate so they are never conflated:

  seed        K samples of the same (model, prompt, guidance) differing only
              in the initial noise -> aleatoric / sampling uncertainty.
  trajectory  evidence maps of the predicted clean image x0 at intermediate
              denoising steps -> when does the sampler commit to a layout?
  model       same prompt AND same seed across models -> epistemic proxy.

and a set of scores for checking whether an uncertainty predicts failure
(AUROC, accuracy-coverage / AURC), which is what makes it useful.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from . import metrics as M


# ------------------------------------------------------------ seed spread
def seed_uncertainty(maps: np.ndarray, embs: np.ndarray | None = None) -> dict:
    """maps: (K, G, G) evidence maps of K seeds; embs: optional (K, D) image embeddings.

    Returns scalar scores (larger = more uncertain) and the per-patch maps.
    """
    k, g, _ = maps.shape
    p = np.stack([M.normalize(m).reshape(g, g) for m in maps])
    mean = p.mean(0)
    var = p.var(0, ddof=1) if k > 1 else np.zeros_like(mean)
    # peak location spread, in patch units
    peaks = np.array([np.unravel_index(np.argmax(m), (g, g)) for m in p], dtype=float)
    # evidence centroid spread
    rr, cc = np.mgrid[0:g, 0:g]
    cent = np.array([[(m * rr).sum(), (m * cc).sum()] for m in p])
    out = {
        "seed_jsd": float(np.mean([M.jsd(a, b) for a, b in combinations(p, 2)])) if k > 1 else 0.0,
        # JSD of the mixture = mutual information between seed and patch
        "seed_mi": float(M.entropy(mean) - np.mean([M.entropy(m) for m in p])),
        "seed_patch_var": float(var.sum()),
        "seed_peak_spread": float(np.sqrt(((peaks - peaks.mean(0)) ** 2).sum(1).mean())),
        "seed_centroid_spread": float(np.sqrt(((cent - cent.mean(0)) ** 2).sum(1).mean())),
        "seed_gini_sd": float(np.std([M.gini(m) for m in p], ddof=1)) if k > 1 else 0.0,
        "mean_map_gini": M.gini(mean),
        "k": k,
    }
    if embs is not None and len(embs) > 1:
        e = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        out["seed_sem_var"] = float(((e - e.mean(0)) ** 2).sum(1).mean())
    return {**out, "mean_map": mean, "var_map": var}


def seed_mi_is_bounded(maps: np.ndarray) -> bool:
    """Sanity: 0 <= MI <= log K (used in tests)."""
    mi = seed_uncertainty(maps)["seed_mi"]
    return -1e-9 <= mi <= np.log(len(maps)) + 1e-9


# ------------------------------------------------------------- trajectory
def trajectory_uncertainty(maps: np.ndarray, steps: np.ndarray, tau: float = 0.05) -> dict:
    """maps: (T, G, G) evidence maps of x0-predictions at denoising `steps`
    (increasing; last = final image). tau: JSD threshold for 'settled'.

    commit_frac: fraction of the trajectory elapsed before the map stays
    within tau of the final map for good (later commitment = more uncertain).
    """
    steps = np.asarray(steps, dtype=float)
    final = maps[-1]
    d = np.array([M.jsd(m, final) for m in maps])
    settled = d <= tau
    # first index after which every later map is settled
    idx = len(d) - 1
    while idx > 0 and settled[idx - 1]:
        idx -= 1
    span = steps[-1] - steps[0] if steps[-1] > steps[0] else 1.0
    frac = (steps - steps[0]) / span
    return {
        "traj_commit_frac": float(frac[idx]),
        "traj_jsd_auc": float(np.trapezoid(d, frac)),
        "traj_gini_final": M.gini(final),
        "traj_gini_first": M.gini(maps[0]),
        "traj_jsd_curve": d,
    }


# ---------------------------------------------------------- model spread
def model_disagreement(maps_by_model: dict[str, np.ndarray]) -> dict:
    """Same prompt and seed, different models: mean pairwise JSD."""
    vals = list(maps_by_model.values())
    pairs = [M.jsd(a, b) for a, b in combinations(vals, 2)]
    return {"model_jsd": float(np.mean(pairs)) if pairs else np.nan, "n_models": len(vals)}


# ---------------------------------------------- does it predict failure?
def auroc(score: np.ndarray, failed: np.ndarray) -> float:
    """P(score of a failure > score of a success); ties count 1/2."""
    score, failed = np.asarray(score, float), np.asarray(failed, bool)
    ok = ~np.isnan(score)
    score, failed = score[ok], failed[ok]
    pos, neg = score[failed], score[~failed]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def coverage_curve(score: np.ndarray, correct: np.ndarray, n: int = 20):
    """Keep the least-uncertain fraction c of samples; accuracy among them.

    Returns (coverages, accuracies, AURC) where AURC is the area under the
    risk (1 - accuracy) vs coverage curve: lower = better uncertainty.
    """
    score, correct = np.asarray(score, float), np.asarray(correct, float)
    ok = ~np.isnan(score)
    order = np.argsort(score[ok], kind="mergesort")
    c_sorted = correct[ok][order]
    covs = np.linspace(1 / n, 1, n)
    accs = np.array([c_sorted[: max(1, int(round(c * len(c_sorted))))].mean() for c in covs])
    return covs, accs, float(np.trapezoid(1 - accs, covs))
