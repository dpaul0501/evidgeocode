"""fe_fit: within-transformation fixed effects with clustered errors."""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import analyze as A  # noqa: E402


def _data(nested, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for gid in range(60):
        for x in (0.0, 0.7, 1.4):
            unit = gid // 3 if nested else rng.integers(0, 20)  # nested: 3 captions per photo
            rows.append(dict(gid=gid, unit=int(unit), x=x, y=0.05 * x + 0.1 * gid % 7 + rng.normal(0, 0.1)))
    return pd.DataFrame(rows)


def test_not_nested_equals_dummy_regression():
    d = _data(nested=False)
    f = smf.ols("y ~ x + C(gid)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.unit})
    b, _, _, p = A.fe_fit(d, "y", ["x"], "x")
    assert abs(b - f.params["x"]) < 1e-10 and abs(p - f.pvalues["x"]) < 1e-8


def test_nested_does_not_count_absorbed_effects():
    d = _data(nested=True)
    dm = d[["y", "x"]] - d.groupby("gid")[["y", "x"]].transform("mean")
    f = sm.OLS(dm.y, dm[["x"]]).fit(cov_type="cluster", cov_kwds={"groups": d.unit})
    b, _, _, p = A.fe_fit(d, "y", ["x"], "x")
    assert abs(b - f.params["x"]) < 1e-10 and abs(p - f.pvalues["x"]) < 1e-8


def test_constant_measure_gives_nan():
    d = _data(nested=True).assign(y=1.0)
    assert np.isnan(A.fe_fit(d, "y", ["x"], "x")[3])
