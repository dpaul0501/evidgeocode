import numpy as np
import pytest

from evidgeo import metrics as M
from evidgeo import smoothing as S


@pytest.mark.parametrize("g", [5, 7, 9, 14])
@pytest.mark.parametrize("alpha", [0.5, 0.7, 0.85, 0.95])
def test_ppr_preserves_mass(g, alpha):
    rng = np.random.default_rng(g)
    x = rng.random((g, g))
    y = S.ppr_smooth(x, alpha)
    assert y.min() >= 0
    assert y.sum() == pytest.approx(x.sum(), rel=1e-10)


def test_ppr_closed_form_matches_iteration():
    x = np.random.default_rng(0).random((7, 7))
    exact = S.ppr_smooth(x, 0.85)
    approx = S.ppr_smooth_iter(x, 0.85, steps=300)
    assert np.allclose(exact, approx, atol=1e-12)


def test_ppr_iteration_preserves_mass_at_every_step():
    x = np.random.default_rng(1).random((7, 7))
    for k in (1, 5, 20):
        assert S.ppr_smooth_iter(x, 0.85, k).sum() == pytest.approx(x.sum(), rel=1e-12)


def test_gini_bounds():
    n = 49
    assert M.gini(np.ones(n)) == pytest.approx(0.0, abs=1e-12)
    one_hot = np.zeros(n)
    one_hot[3] = 1
    assert M.gini(one_hot) == pytest.approx(1 - 1 / n)


def test_smoothing_lowers_gini():
    x = np.zeros((7, 7))
    x[3, 3] = 1
    assert M.gini(S.ppr_smooth(x)) < M.gini(x)


def test_gaussian_renormalized_keeps_mass_and_raw_leaks():
    x = np.zeros((7, 7))
    x[0, 0] = 1
    assert S.gaussian_smooth(x, 1.0).sum() == pytest.approx(1.0)
    assert S.mass_leak(x, 1.0) > 0


def test_jsd_properties():
    p = np.random.default_rng(2).random(49)
    assert M.jsd(p, p) == pytest.approx(0.0, abs=1e-12)
    a, b = np.eye(49)[0], np.eye(49)[1]
    assert M.jsd(a, b) == pytest.approx(np.log(2))


def test_mask_helpers():
    x = np.zeros((7, 7))
    x[2, 2] = 1
    mask = np.zeros((7, 7), bool)
    mask[2, 2] = True
    assert M.mass_in_mask(x, mask) == pytest.approx(1.0)
    assert M.peak_hit(x, mask)
