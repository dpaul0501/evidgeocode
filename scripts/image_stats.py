#!/usr/bin/env python
"""Prompt-free image statistics for every image run_eval.py would see (CPU only).

Used to test whether "guidance concentrates evidence" is just guidance making
images simpler (more contrast, a centred subject, less clutter). Same discovery
arguments as run_eval.py; writes <out>/imstats/<corpus>__<dataset>.parquet keyed
by Record.key().

  edge_gini/top10/entropy  Sobel edge magnitude pooled on the same G x G grid
  edge_centre              share of edge mass in the central (G//2+1)^2 cells
  rms_contrast             std of luminance in [0, 1]
  colourfulness            Hasler & Suesstrunk (2003)
  hf_energy                share of spectral power above 1/4 of Nyquist
  saturation               mean HSV saturation
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidgeo import data as D  # noqa: E402
from evidgeo import metrics as M  # noqa: E402

G = 7
SIZE = 224


def stats_for(path: str) -> dict:
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    rgb = np.asarray(im, dtype=np.float64) / 255.0
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    mag = np.hypot(ndimage.sobel(lum, 0), ndimage.sobel(lum, 1))
    grid = mag.reshape(G, SIZE // G, G, SIZE // G).mean(axis=(1, 3))
    c0, c1 = G // 2 - 1, G // 2 + 2
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    f = np.abs(np.fft.fftshift(np.fft.fft2(lum - lum.mean()))) ** 2
    yy, xx = np.mgrid[-SIZE // 2:SIZE // 2, -SIZE // 2:SIZE // 2]
    hf = f[np.hypot(yy, xx) > SIZE / 8].sum() / max(f.sum(), 1e-12)
    hsv = np.asarray(im.convert("HSV"), dtype=np.float64) / 255.0
    return {
        **M.concentration(grid, "edge"),
        "edge_centre": float(grid[c0:c1, c0:c1].sum() / max(grid.sum(), 1e-12)),
        "rms_contrast": float(lum.std()),
        "colourfulness": float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())),
        "hf_energy": float(hf),
        "saturation": float(hsv[..., 1].mean()),
    }


def _one(item):
    key, path = item
    try:
        return {"key": key, **stats_for(path)}
    except Exception as e:  # unreadable file: keep going, record it
        return {"key": key, "error": str(e)[:200]}


def main():
    import run_eval as R  # same CLI and corpus discovery as the evaluation

    args = R.parse_args()
    recs = R.collect(args)
    os.makedirs(os.path.join(args.out, "imstats"), exist_ok=True)
    for (corpus, dataset), grp in pd.DataFrame(
            [(r.corpus, r.dataset, r.key(), r.path) for r in recs],
            columns=["corpus", "dataset", "key", "path"]).groupby(["corpus", "dataset"]):
        fp = os.path.join(args.out, "imstats", f"{corpus}__{dataset}.parquet")
        if os.path.exists(fp):
            continue
        with ProcessPoolExecutor() as ex:
            rows = list(ex.map(_one, zip(grp.key, grp.path), chunksize=64))
        pd.DataFrame(rows).to_parquet(fp, index=False)
        print(f"{fp}: {len(rows)} images", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
