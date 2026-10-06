#!/usr/bin/env python
"""Convert meme / face / generic image datasets into an external JSONL for run_eval.py.

Each output line: {"image": <path relative to the JSONL>, "text": ..., "label": ..., "group": int}

  # Hateful Memes (original layout: <dir>/{train,dev_seen,test_seen}.jsonl + img/)
  python scripts/prepare_external.py hateful_memes --src /data/hateful_memes --out ext/hateful_memes.jsonl

  # MemeCap (memes_*.json + images); choose which text the evidence is measured against
  python scripts/prepare_external.py memecap --src /data/memecap --text interpretation --out ext/memecap.jsonl

  # FaceCaption-15M from Hugging Face (streams N samples, saves images locally)
  python scripts/prepare_external.py facecaption --n 5000 --out ext/facecaption.jsonl

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
    from datasets import load_dataset

    ds = load_dataset("OpenFace-CQUPT/FaceCaption-15M", split="train", streaming=True)
    img_dir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "facecaption_img")
    os.makedirs(img_dir, exist_ok=True)
    for i, r in enumerate(ds):
        if i >= a.n:
            break
        path = os.path.join(img_dir, f"{i:06d}.jpg")
        if not os.path.exists(path):
            r["image"].convert("RGB").save(path, quality=95)
        yield {"image": path, "text": r.get("caption") or "", "label": "real_face", "group": i}


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
    a = ap.parse_args()
    gen = {"hateful_memes": hateful_memes, "memecap": memecap, "facecaption": facecaption,
           "csv": generic_csv}[a.kind]
    write(gen(a), a.out)


if __name__ == "__main__":
    main()
