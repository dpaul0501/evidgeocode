"""Discovery of every EvidGeo image corpus on disk.

Layouts (as produced by datagen_coco.py and the consistency notebook):

  guidance     <root>/images/[sd/]<model>/<mode>/<gid:05d>.png
               <root>/cache/manifest*.jsonl   {"global_id", "caption"}; gid may be
                                              local (v3), so each image also gets
                                              hf_row, its row in the master order
               <root>/meta/*.jsonl            per-image generation records
  consistency  <root>/images/<mode>/<prompt_id:04d>_seed<k>.png
               <root>/cache/selected_prompts.json  [{"global_id", "caption"}]
  real         rows of the HF dataset after .shuffle(42) (the "master" order,
               manifest_v3_N5000_seed42.jsonl), the same rows used as captions
               and i2i sources. Paired to generated images by hf_row.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from glob import glob

from PIL import Image

GID_RE = re.compile(r"^(\d+)\.png$")
CONS_RE = re.compile(r"^(\d+)_seed(\d+)\.png$")


@dataclass
class Record:
    corpus: str        # guidance | consistency | real
    dataset: str       # e.g. multiguide_coco_v2
    model: str         # sd14 | sd15 | biggan | coco
    mode: str          # text | i2i | weak | real
    gid: int           # prompt / caption index into this corpus's manifest
    seed: int
    hf_row: int        # row of the paired real image in the master HF order
    path: str          # image file, or "hf:<row>" for real images not yet cached
    caption: str

    def key(self) -> str:
        return f"{self.corpus}/{self.dataset}/{self.model}/{self.mode}/{self.gid}/{self.seed}"


MANIFEST_PREFERENCE = ("only_new", "Erland__coco_captions_small", "manifest")


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def find_manifest(root: str, explicit: str = "") -> dict[int, str]:
    """global_id -> caption, from the newest manifest under <root>/cache."""
    path = explicit
    if not path:
        cands = sorted(glob(os.path.join(root, "cache", "*manifest*.jsonl")) +
                       glob(os.path.join(root, "*manifest*.jsonl")))
        for pref in MANIFEST_PREFERENCE:  # the manifest the generator used, not the newest
            hit = [c for c in cands if pref in os.path.basename(c)]
            if hit:
                path = hit[0]
                break
    if not path:
        return {}
    out = {}
    for i, r in enumerate(read_jsonl(path)):
        gid = int(r.get("global_id", r.get("gid", i)))
        out[gid] = r.get("caption") or r.get("prompt") or ""
    return out


def meta_index(root: str) -> dict[tuple, dict]:
    """(model, mode, gid) -> generation record from <root>/meta/*.jsonl."""
    idx = {}
    for p in glob(os.path.join(root, "meta", "*.jsonl")):
        for r in read_jsonl(p):
            model = r.get("model") or r.get("model_name")
            mode = r.get("mode")
            gid = r.get("global_id", r.get("gid"))
            if model is None or mode is None or gid is None:
                # fall back to file-name convention <model>_<mode>_shXX.jsonl
                parts = os.path.basename(p).split("_")
                model = model or parts[0]
                mode = mode or (parts[1] if len(parts) > 1 else None)
            if gid is not None:
                idx[(model, mode, int(gid))] = r
    return idx


def hf_rows_for(manifest: dict[int, str], master: list[str]) -> dict[int, int]:
    """Map a corpus's local gid to its row in the master HF order, by caption.

    v2 gids are already master rows; v3 gids are local and offset. Ties
    (repeated captions) resolve to the row equal to the gid when possible.
    """
    by_cap: dict[str, list[int]] = {}
    for row, cap in enumerate(master):
        by_cap.setdefault(cap, []).append(row)
    out = {}
    for gid, cap in manifest.items():
        rows = by_cap.get(cap, [])
        out[gid] = gid if gid in rows else (rows[0] if rows else -1)
    return out


def image_dirs(img_root: str):
    """Yield (model, mode, dir) for every leaf folder holding <gid>.png files."""
    for dirpath, dirnames, filenames in os.walk(img_root):
        if any(GID_RE.match(f) for f in filenames):
            yield os.path.basename(os.path.dirname(dirpath)), os.path.basename(dirpath), dirpath


def guidance_records(root: str, master: list[str], manifest: dict[int, str] | None = None
                     ) -> list[Record]:
    dataset = os.path.basename(os.path.normpath(root))
    manifest = manifest if manifest is not None else find_manifest(root)
    rows = hf_rows_for(manifest, master)
    meta = meta_index(root)
    recs = []
    img_root = os.path.join(root, "images")
    if not os.path.isdir(img_root):
        return recs
    for model, mode, ddir in sorted(image_dirs(img_root)):
        for name in sorted(os.listdir(ddir)):
            m = GID_RE.match(name)
            if not m:
                continue
            gid = int(m.group(1))
            rec = meta.get((model, mode, gid), {})
            cap = manifest.get(gid) or rec.get("caption") or rec.get("caption_proxy") or ""
            recs.append(Record("guidance", dataset, model, mode, gid, int(rec.get("seed", -1)),
                               rows.get(gid, -1), os.path.join(ddir, name), cap))
    return recs


def consistency_manifest(root: str) -> dict[int, str]:
    p = os.path.join(root, "cache", "selected_prompts.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        items = json.load(f)
    return {int(r["global_id"]): r["caption"] for r in items}


def consistency_records(root: str, master: list[str], model: str = "sd") -> list[Record]:
    dataset = os.path.basename(os.path.normpath(root))
    manifest = consistency_manifest(root) or dict(enumerate(master))
    rows = hf_rows_for(manifest, master)
    recs = []
    for path in sorted(glob(os.path.join(root, "images", "**", "*.png"), recursive=True)):
        m = CONS_RE.match(os.path.basename(path))
        if not m:
            continue
        mode = os.path.basename(os.path.dirname(path))
        pid, seed = int(m.group(1)), int(m.group(2))
        recs.append(Record("consistency", dataset, model, mode, pid, seed, rows.get(pid, -1),
                           path, manifest.get(pid, "")))
    return recs


def real_records(hf_rows: set[int], master: list[str],
                 dataset: str = "Erland/coco_captions_small") -> list[Record]:
    name = dataset.replace("/", "__")
    return [Record("real", name, "coco", "real", r, -1, r, f"hf:{r}", master[r])
            for r in sorted(hf_rows) if 0 <= r < len(master)]


def cache_real_images(recs: list[Record], cache_dir: str, dataset: str, split: str,
                      shuffle_seed: int, size: int = 512) -> list[Record]:
    """Materialise HF rows to PNG once, square-resized as the v2 generator did."""
    os.makedirs(cache_dir, exist_ok=True)
    todo = [r for r in recs if not os.path.exists(os.path.join(cache_dir, f"{r.hf_row:05d}.png"))]
    if todo:
        from datasets import load_dataset

        n = max(r.hf_row for r in recs) + 1
        ds = load_dataset(dataset, split=split).shuffle(seed=shuffle_seed).select(range(n))
        for r in todo:
            ds[r.hf_row]["image"].convert("RGB").resize((size, size)).save(
                os.path.join(cache_dir, f"{r.hf_row:05d}.png"))
    for r in recs:
        r.path = os.path.join(cache_dir, f"{r.hf_row:05d}.png")
    return recs


def load_image(rec: Record) -> Image.Image:
    return Image.open(rec.path).convert("RGB")


def to_rows(recs: list[Record]) -> list[dict]:
    return [asdict(r) for r in recs]
