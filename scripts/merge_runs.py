#!/usr/bin/env python
"""Merge per-part evaluation outputs (features/, maps/, imstats/) into one run directory.

    python scripts/merge_runs.py --out kaggle_out/memecap_run kaggle_out/memecap/*_results.tgz

Each input is a results tarball or an unpacked run directory. Chunk files are
prefixed with the part's index so parts never overwrite each other; imstats
parquets with the same name are concatenated and de-duplicated by key.
"""
import argparse
import glob
import os
import shutil
import tarfile
import tempfile

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("inputs", nargs="+")
a = ap.parse_args()

if os.path.exists(a.out):
    raise SystemExit(f"{a.out} exists; remove it first")
tmp = tempfile.mkdtemp()
imstats: dict[str, list[pd.DataFrame]] = {}
n_files = 0
for i, src in enumerate(sorted(a.inputs)):
    d = src
    if src.endswith((".tgz", ".tar.gz")):
        d = os.path.join(tmp, str(i))
        with tarfile.open(src) as t:
            t.extractall(d, filter="data")
    for kind in ("features", "maps"):
        for p in glob.glob(os.path.join(d, kind, "**", "*.*"), recursive=True):
            rel = os.path.relpath(p, os.path.join(d, kind))
            dst = os.path.join(a.out, kind, os.path.dirname(rel), f"p{i:02d}_{os.path.basename(rel)}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(p, dst)
            n_files += 1
    for p in glob.glob(os.path.join(d, "imstats", "*.parquet")):
        imstats.setdefault(os.path.basename(p), []).append(pd.read_parquet(p))
    for p in glob.glob(os.path.join(d, "text_masks", "*")):  # OCR masks travel with the run
        os.makedirs(os.path.join(a.out, "text_masks"), exist_ok=True)
        shutil.copy(p, os.path.join(a.out, "text_masks", os.path.basename(p)))
os.makedirs(os.path.join(a.out, "imstats"), exist_ok=True)
for name, parts in imstats.items():
    pd.concat(parts, ignore_index=True).drop_duplicates("key").to_parquet(
        os.path.join(a.out, "imstats", name), index=False)
shutil.rmtree(tmp)
print(f"{len(a.inputs)} inputs, {n_files} feature/map files, {len(imstats)} imstats tables -> {a.out}")
