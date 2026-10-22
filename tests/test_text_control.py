"""text_control: real images whose evidence sits on overlaid text vs generations without text."""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import analyze as A  # noqa: E402


def test_text_control_detects_text_evidence(tmp_path):
    g, rng = 7, np.random.default_rng(0)
    m = np.zeros((g * 32, g * 32), np.uint8)
    m[: 2 * 32] = 255  # top two rows of cells are text
    rows, store = [], {}
    for gid in range(8):
        Image.fromarray(m).save(tmp_path / f"{gid:05d}_text.png")
        real = rng.random((g, g)) * 0.1
        real[:2] += 1.0  # real evidence on the text
        key = f"external/memes_real/real/real/{gid}"
        store[key] = {f"g{g}_pec": real}
        rows.append(dict(corpus="external", dataset="memes_real", model="real", mode="real", gid=gid,
                         key=key, pec_gini=0.5))
        for seed in range(2):
            key = f"guidance/memes/flux/cfg4/{gid}/{seed}"
            store[key] = {f"g{g}_pec": rng.random((g, g)) + 1.0}
            rows.append(dict(corpus="guidance", dataset="memes", model="flux", mode="cfg4", gid=gid,
                             key=key, pec_gini=0.2))
    A.NUM.clear()
    A.text_control(pd.DataFrame(rows), store, g, [("memes_real", "memes", str(tmp_path))], str(tmp_path))
    t = pd.read_csv(tmp_path / "text_control.csv")
    r = t[t.signal == "pec"].iloc[0]
    assert r.n == 8 and abs(r.text_area - 2 / 7) < 1e-6
    assert r.real_text > 0.8 and 0.2 < r.gen_text < 0.4  # generations get ~area share by chance
    assert r.real_gini > r.gen_gini  # raw comparison is driven by text


def test_external_pairs_averages_seeds(tmp_path):
    rows = []
    for gid in range(6):
        rows.append(dict(corpus="external", dataset="faces_real", model="real", mode="real", gid=gid, pec_gini=0.5))
        for seed, v in enumerate((0.2, 0.4)):  # seed mean 0.3, always below the real 0.5
            rows.append(dict(corpus="guidance", dataset="faces", model="flux", mode="cfg4", gid=gid, seed=seed,
                             pec_gini=v + 0.01 * gid))
    A.NUM.clear()
    A.external_pairs(pd.DataFrame(rows), [("faces_real", "faces")], str(tmp_path))
    r = pd.read_csv(tmp_path / "external_pairs.csv").iloc[0]
    assert r.n == 6 and abs(r.mean_b - 0.5) < 1e-9 and abs(r.mean_a - 0.325) < 1e-9
