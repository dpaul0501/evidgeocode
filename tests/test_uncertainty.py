import numpy as np
import pytest

from evidgeo import uncertainty as U


def onehot(g, r, c):
    m = np.full((g, g), 1e-9)
    m[r, c] = 1
    return m


def test_identical_seeds_have_zero_uncertainty():
    m = np.random.default_rng(0).random((7, 7))
    u = U.seed_uncertainty(np.stack([m, m, m]))
    assert u["seed_jsd"] == pytest.approx(0, abs=1e-12)
    assert u["seed_mi"] == pytest.approx(0, abs=1e-9)
    assert u["seed_peak_spread"] == 0


def test_disjoint_peaks_are_maximally_uncertain():
    maps = np.stack([onehot(7, 0, 0), onehot(7, 6, 6), onehot(7, 0, 6), onehot(7, 6, 0)])
    u = U.seed_uncertainty(maps)
    assert u["seed_mi"] == pytest.approx(np.log(4), rel=1e-6)
    assert u["seed_peak_spread"] > 4
    assert U.seed_mi_is_bounded(maps)


def test_seed_uncertainty_orders_as_expected():
    rng = np.random.default_rng(1)
    base = rng.random((7, 7))
    low = np.stack([base + 0.01 * rng.random((7, 7)) for _ in range(4)])
    high = rng.random((4, 7, 7)) ** 4
    assert U.seed_uncertainty(low)["seed_jsd"] < U.seed_uncertainty(high)["seed_jsd"]


def test_trajectory_commit():
    final = onehot(7, 3, 3)
    early = np.stack([onehot(7, 0, 0)] * 2 + [final] * 8)
    late = np.stack([onehot(7, 0, 0)] * 8 + [final] * 2)
    steps = np.arange(10)
    assert U.trajectory_uncertainty(early, steps)["traj_commit_frac"] < \
        U.trajectory_uncertainty(late, steps)["traj_commit_frac"]


def test_auroc_and_coverage():
    s = np.array([0.1, 0.2, 0.8, 0.9])
    failed = np.array([0, 0, 1, 1])
    assert U.auroc(s, failed) == 1.0
    assert U.auroc(-s, failed) == 0.0
    assert U.auroc(np.ones(4), failed) == 0.5
    covs, accs, aurc = U.coverage_curve(s, 1 - failed, n=4)
    assert accs[0] == 1 and accs[-1] == 0.5
    _, _, aurc_bad = U.coverage_curve(-s, 1 - failed, n=4)
    assert aurc < aurc_bad
