#!/usr/bin/env python
"""Face and meme domains as EvidGeo corpora, on their standard releases.

    python scripts/prepare_domains.py memecap  --src data_ext/memecap
    python scripts/prepare_domains.py mmceleba --src data_ext/mmceleba

memecap:  every meme of MemeCap (Hwang & Shwartz, EMNLP 2023), memes-trainval.json then
          memes-test.json in file order; prompt = the first literal image caption
          (img_captions[0]); the real meme is the paired real image (images in <src>/img/memes).
mmceleba: the official MM-CelebA-HQ test split (test/filenames.pickle, 6,000 CelebA-HQ ids, in
          order); prompt = the first of the image's ten captions (text_all/<id>.txt); the real
          CelebA-HQ photo (<src>/img/images/<id>.jpg) is the paired real image.
Writes <name>_prompts.jsonl ({global_id, caption, ...}) for generate_modern --prompts-jsonl and
<name>_real.jsonl ({image, text, group, ...}, image paths relative to the images folder) for
run_eval --external. Neither dataset may be redistributed: release ids, not images.
"""
import argparse
import ast
import glob
import json
import os
import pickle


class _ListOnly(pickle.Unpickler):
    """filenames.pickle is a third-party file: allow plain containers and strings only."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def memecap(src):
    prompts, real, gid = [], [], 0
    for split, f in (("trainval", "memes-trainval.json"), ("test", "memes-test.json")):
        for r in json.load(open(os.path.join(src, f))):
            caps = ast.literal_eval(r["img_captions"]) if isinstance(r["img_captions"], str) else r["img_captions"]
            cap = caps[0].strip()
            prompts.append({"global_id": gid, "caption": cap, "post_id": r["post_id"], "split": split})
            real.append({"image": f"memes/{r['img_fname']}", "text": cap, "group": gid, "model": "real",
                         "mode": "real", "label": split, "post_id": r["post_id"]})
            gid += 1
    return prompts, real


def mmceleba(src):
    with open(os.path.join(src, "test", "filenames.pickle"), "rb") as fh:
        ids = [str(x) for x in _ListOnly(fh).load()]
    prompts, real = [], []
    for gid, i in enumerate(ids):
        p = os.path.join(src, "text_all", f"{i}.txt")
        if not os.path.exists(p):
            p = glob.glob(os.path.join(src, "text_all", "**", f"{i}.txt"), recursive=True)[0]
        cap = [line.strip() for line in open(p) if line.strip()][0]
        assert os.path.exists(os.path.join(src, "img", "images", f"{i}.jpg")), i
        prompts.append({"global_id": gid, "caption": cap, "celebahq_id": int(i)})
        real.append({"image": f"images/{i}.jpg", "text": cap, "group": gid, "model": "real", "mode": "real",
                     "label": "test", "celebahq_id": int(i)})
    return prompts, real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["memecap", "mmceleba"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="", help="output folder (default: --src)")
    a = ap.parse_args()
    prompts, real = {"memecap": memecap, "mmceleba": mmceleba}[a.kind](a.src)
    out = a.out or a.src
    for name, rows in ((f"{a.kind}_prompts.jsonl", prompts), (f"{a.kind}_real.jsonl", real)):
        with open(os.path.join(out, name), "w") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in rows)
    print(f"{a.kind}: {len(prompts)} captions -> {out}")


if __name__ == "__main__":
    main()
