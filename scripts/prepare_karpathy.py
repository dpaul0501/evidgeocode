#!/usr/bin/env python
"""COCO Karpathy test split (5,000 val2014 photos) as an EvidGeo corpus.

    python scripts/prepare_karpathy.py --out /content/in/karpathy

global_id = position in the split (HF yerevann/coco-karpathy, test), caption = the
first of the five reference captions, one caption per photograph. Writes
  prompts.jsonl  {global_id, caption, cocoid}          for generate_modern --prompts-jsonl
  init.jsonl     {global_id, image}                    for generate_modern --init-jsonl
  real.jsonl     {image, text, group, ...}             for run_eval --external
and downloads each photo from images.cocodataset.org, resized to 512x512 like the
v2 pipeline's init images.
"""
import argparse
import ast
import io
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--limit", type=int, default=0, help="first N photos only (testing)")
a = ap.parse_args()

from datasets import load_dataset  # noqa: E402

ds = load_dataset("yerevann/coco-karpathy", split="test")
os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
rows = []
for gid, r in enumerate(ds):
    sents = r["sentences"]
    if isinstance(sents, str):  # some mirrors store the list as its repr
        sents = ast.literal_eval(sents)
    rows.append(dict(global_id=gid, caption=" ".join(str(sents[0]).split()), cocoid=int(r["cocoid"]),
                     url=r["url"], image=f"images/{int(r['cocoid'])}.jpg"))
assert len(rows) == 5000 and len({r["cocoid"] for r in rows}) == 5000
rows = rows[:a.limit] if a.limit else rows


def fetch(r):
    path = os.path.join(a.out, r["image"])
    if os.path.exists(path):
        return True
    for t in range(5):
        try:
            data = urllib.request.urlopen(r["url"], timeout=60).read()
            im = Image.open(io.BytesIO(data)).convert("RGB").resize((a.size, a.size), Image.BICUBIC)
            im.save(path + ".tmp.jpg", quality=95)
            os.replace(path + ".tmp.jpg", path)
            return True
        except Exception:
            time.sleep(2 * (t + 1))
    return False


with ThreadPoolExecutor(a.workers) as ex:
    ok = list(ex.map(fetch, rows))
print(f"{sum(ok)}/{len(rows)} photos")
assert all(ok), "some photos failed to download"
with open(os.path.join(a.out, "prompts.jsonl"), "w") as f:
    f.writelines(json.dumps({k: r[k] for k in ("global_id", "caption", "cocoid")}) + "\n" for r in rows)
with open(os.path.join(a.out, "init.jsonl"), "w") as f:
    f.writelines(json.dumps({"global_id": r["global_id"], "image": r["image"]}) + "\n" for r in rows)
with open(os.path.join(a.out, "real.jsonl"), "w") as f:
    f.writelines(json.dumps({"image": r["image"], "text": r["caption"], "group": r["global_id"], "model": "real",
                             "mode": "real", "label": "real", "cocoid": r["cocoid"]}) + "\n" for r in rows)
