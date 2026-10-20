#!/usr/bin/env python
"""OCR text masks for captioned real images (e.g. memes), for text-region controls.

    python scripts/text_masks.py --jsonl data_ext/memecap/memes_real.jsonl --out data_ext/memecap/text_masks

Writes <out>/<group:05d>_text.png (white = text, boxes padded by --pad of the
short side) and <out>/summary.csv with the text-covered area per image.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pad", type=float, default=0.01)
ap.add_argument("--min-conf", type=float, default=0.3)
a = ap.parse_args()

import easyocr  # noqa: E402

reader = easyocr.Reader(["en"], gpu=False, verbose=False)
os.makedirs(a.out, exist_ok=True)
rows = []
for line in open(a.jsonl):
    r = json.loads(line)
    im = Image.open(r["image"]).convert("RGB")
    w, h = im.size
    pad = a.pad * min(w, h)
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    n = 0
    for box, _, conf in reader.readtext(np.asarray(im)):
        if conf < a.min_conf:
            continue
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        d.rectangle([min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad], fill=255)
        n += 1
    m.save(os.path.join(a.out, f"{int(r['group']):05d}_text.png"))
    rows.append(dict(group=int(r["group"]), n_boxes=n, text_area=float(np.asarray(m).mean() / 255)))
s = pd.DataFrame(rows)
s.to_csv(os.path.join(a.out, "summary.csv"), index=False)
print(f"{len(s)} images; with text {int((s.n_boxes > 0).sum())}; median text area {s.text_area.median():.3f}")
