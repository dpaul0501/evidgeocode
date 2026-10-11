#!/usr/bin/env python
"""Convert meme / face / generic image datasets into an external JSONL for run_eval.py.

Each output line: {"image": <path relative to the JSONL>, "text": ..., "label": ..., "group": int}

  # Hateful Memes (original layout: <dir>/{train,dev_seen,test_seen}.jsonl + img/)
  python scripts/prepare_external.py hateful_memes --src /data/hateful_memes --out ext/hateful_memes.jsonl

  # MemeCap (memes_*.json + images); choose which text the evidence is measured against
  python scripts/prepare_external.py memecap --src /data/memecap --text interpretation --out ext/memecap.jsonl

  # FaceCaption-15M attribute captions -> generation manifest (no real photos are downloaded)
  python scripts/prepare_external.py facecaption --n 300 --out ext/faces_manifest.jsonl

  # Anything else: CSV with columns image,text[,label,group]
  python scripts/prepare_external.py csv --src my.csv --out ext/my.jsonl

Licences: Hateful Memes, MemeCap, MAMI, CelebA-family and most face sets are
research-only and must not be redistributed. Keep images local (or in a
private Drive / Kaggle dataset) and never publish them with this repo.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from glob import glob


def write(rows, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    base = os.path.dirname(os.path.abspath(out))
    n = 0
    with open(out, "w") as f:
        for r in rows:
            if not r.get("text") or not os.path.exists(r["image"]):
                continue
            r["image"] = os.path.relpath(os.path.abspath(r["image"]), base)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows to {out}")


def first(d: dict, *keys, default=""):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            v = d[k]
            return v[0] if isinstance(v, list) else v
    return default


def hateful_memes(a):
    seen = set()
    for split in ("train", "dev_seen", "dev_unseen", "test_seen", "test_unseen"):
        p = os.path.join(a.src, f"{split}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if r["id"] in seen or "label" not in r:
                continue
            seen.add(r["id"])
            yield {"image": os.path.join(a.src, r["img"]), "text": r["text"],
                   "label": "hateful" if int(r["label"]) == 1 else "not_hateful",
                   "group": int(r["id"]), "split": split}


def memecap(a):
    field = {"ocr": ("title",), "caption": ("img_captions",),
             "interpretation": ("meme_captions",)}[a.text]
    files = sorted(glob(os.path.join(a.src, "*.json")))
    img_dirs = [a.src] + [d for d in glob(os.path.join(a.src, "*")) if os.path.isdir(d)]
    i = 0
    for p in files:
        data = json.load(open(p))
        for r in data if isinstance(data, list) else data.values():
            fname = first(r, "img_fname", "image", "img", "filename")
            path = next((os.path.join(d, fname) for d in img_dirs
                         if os.path.exists(os.path.join(d, fname))), os.path.join(a.src, fname))
            yield {"image": path, "text": first(r, *field), "label": a.text, "group": i,
                   "split": os.path.basename(p)}
            i += 1


def facecaption(a):
    """FaceCaption-15M attribute captions only (CC-BY-4.0 text).

    The dataset stores web URLs of scraped photos of identifiable people, not
    images, so we deliberately do not download real faces. The captions are
    used as prompts for face generation (see --manifest-out); no real pairs.
    """
    import ast
    import urllib.request

    rows, off = [], 0
    while len(rows) < a.n:
        url = ("https://datasets-server.huggingface.co/rows?dataset=OpenFace-CQUPT/FaceCaption-15M"
               f"&config=default&split=train&offset={off}&length=100")
        with urllib.request.urlopen(url, timeout=60) as f:
            batch = json.load(f)["rows"]
        if not batch:
            break
        for r in batch:
            cap = r["row"]["caption"]
            cap = ast.literal_eval(cap)[0] if isinstance(cap, str) and cap.startswith("[") else \
                (cap[0] if isinstance(cap, list) else cap)
            if cap:
                rows.append(cap.strip())
        off += 100
    rows = rows[:a.n]
    with open(a.manifest_out or a.out, "w") as f:
        for i, cap in enumerate(rows):
            f.write(json.dumps({"global_id": i, "caption": cap}) + "\n")
    print(f"wrote {len(rows)} face captions to {a.manifest_out or a.out}")
    return []


def generic_csv(a):
    base = os.path.dirname(os.path.abspath(a.src))
    for i, r in enumerate(csv.DictReader(open(a.src))):
        img = r["image"] if os.path.isabs(r["image"]) else os.path.join(base, r["image"])
        yield {"image": img, "text": r.get("text", ""), "label": r.get("label", ""),
               "group": int(r.get("group") or i)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["hateful_memes", "memecap", "facecaption", "csv"])
    ap.add_argument("--src", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="interpretation", choices=["ocr", "caption", "interpretation"],
                    help="MemeCap: which text the image evidence is measured against")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--manifest-out", default="", help="facecaption: caption manifest for generate_modern.py")
    a = ap.parse_args()
    gen = {"hateful_memes": hateful_memes, "memecap": memecap, "facecaption": facecaption,
           "csv": generic_csv}[a.kind]
    rows = gen(a)
    if a.kind != "facecaption":
        write(rows, a.out)


if __name__ == "__main__":
    main()
