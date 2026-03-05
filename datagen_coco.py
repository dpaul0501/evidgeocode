"""
datagen_coco.py
================
Sharded, resumable dataset generator for the EvidGeo benchmark.

Generates 22,500 images across 3 guidance modes × 2 SD models from COCO captions:
  - weak   : minimal-guidance text-to-image  (CFG = 1.0, prompt = "a photo")
  - text   : full text-conditioned           (CFG = 7.5, COCO caption)
  - image  : image-to-image conditioned      (CFG = 7.5, strength = 0.5)

Models: SD 1.4 (CompVis/stable-diffusion-v1-4)
        SD 1.5 (runwayml/stable-diffusion-v1-5)

Usage (one Colab cell per shard):
    cfg.SHARD_ID = 0   # change 0..NUM_SHARDS-1 per run
    # then Execute All

Requirements:
    pip install diffusers transformers accelerate safetensors \
                ftfy datasets pillow tqdm
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import json
import math
import traceback
from dataclasses import dataclass, field
from typing import Dict

import torch
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Config  ← edit only this block
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Cfg:
    # Output root (Google Drive path for Colab, or any local path)
    BASE: str = "/content/drive/MyDrive/multiguide_coco_v2"

    # Image generation
    IMG_SIZE: int = 256
    STEPS: int = 30

    # Guidance modes
    PROMPT_WEAK: str = "a photo"
    CFG_WEAK: float = 1.0
    CFG_TEXT: float = 7.5
    CFG_I2I: float = 7.5
    I2I_STRENGTH: float = 0.5

    # Dataset scale
    N_PER_MODEL_PER_MODE: int = 3750   # → 7 500 per mode across 2 models
    SEED_BASE: int = 777
    COCO_DATASET: str = "Erland/coco_captions_small"
    COCO_SPLIT: str = "train"
    COCO_SHUFFLE_SEED: int = 42

    # Sharding (set SHARD_ID = 0..NUM_SHARDS-1 per Colab run)
    NUM_SHARDS: int = 10
    SHARD_ID: int = 0

    # Set True on the first run to create the frozen caption manifest only
    BUILD_MANIFEST_ONLY: bool = False

    # Models
    MODELS: Dict[str, str] = field(default_factory=lambda: {
        "sd14": "CompVis/stable-diffusion-v1-4",
        "sd15": "runwayml/stable-diffusion-v1-5",
    })


cfg = Cfg()
assert 0 <= cfg.SHARD_ID < cfg.NUM_SHARDS, "SHARD_ID out of range"

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Paths & device
# ──────────────────────────────────────────────────────────────────────────────
IMG_DIR   = os.path.join(cfg.BASE, "images")
META_DIR  = os.path.join(cfg.BASE, "meta")
CACHE_DIR = os.path.join(cfg.BASE, "cache")
for d in [IMG_DIR, META_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

MANIFEST_PATH = os.path.join(
    CACHE_DIR,
    (f"manifest_{cfg.COCO_DATASET.replace('/','__')}"
     f"_{cfg.COCO_SPLIT}_N{cfg.N_PER_MODEL_PER_MODE}"
     f"_seed{cfg.COCO_SHUFFLE_SEED}.jsonl"),
)

DEVICE = "cuda"
DTYPE  = torch.float16
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

print("BASE    :", cfg.BASE)
print("MANIFEST:", MANIFEST_PATH)
print("SHARD   :", cfg.SHARD_ID, "/", cfg.NUM_SHARDS)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Caption manifest  (frozen IDs + captions — written once)
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_caption(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list) and len(x) > 0:
        return x[0] if isinstance(x[0], str) else str(x[0])
    return str(x)


def build_manifest() -> None:
    """Write a deterministic caption manifest used by all shards."""
    coco = load_dataset(cfg.COCO_DATASET, split=cfg.COCO_SPLIT)
    coco = coco.shuffle(seed=cfg.COCO_SHUFFLE_SEED).select(range(cfg.N_PER_MODEL_PER_MODE))

    with open(MANIFEST_PATH, "w") as f:
        for gid in range(cfg.N_PER_MODEL_PER_MODE):
            row = coco[gid]
            cap = None
            for key in ("caption", "captions"):
                if key in row:
                    cap = _normalize_caption(row[key])
                    break
            if cap is None:
                for key in row.keys():
                    if "caption" in key:
                        cap = _normalize_caption(row[key])
                        break
            if cap is None:
                raise ValueError(f"No caption field in row keys: {list(row.keys())}")
            f.write(json.dumps({"global_id": gid, "caption": cap}) + "\n")

    print("✅ Manifest written:", MANIFEST_PATH)


if cfg.BUILD_MANIFEST_ONLY or not os.path.exists(MANIFEST_PATH):
    build_manifest()
    if cfg.BUILD_MANIFEST_ONLY:
        print("Manifest-only run complete. Set BUILD_MANIFEST_ONLY=False and re-run shards.")
        raise SystemExit

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Load manifest + COCO in the same deterministic order
# ──────────────────────────────────────────────────────────────────────────────
manifest = []
with open(MANIFEST_PATH) as f:
    for line in f:
        manifest.append(json.loads(line))
assert len(manifest) == cfg.N_PER_MODEL_PER_MODE

coco = (
    load_dataset(cfg.COCO_DATASET, split=cfg.COCO_SPLIT)
    .shuffle(seed=cfg.COCO_SHUFFLE_SEED)
    .select(range(cfg.N_PER_MODEL_PER_MODE))
)


def get_coco_item(global_id: int):
    row = coco[global_id]
    img = row["image"].convert("RGB").resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
    cap = manifest[global_id]["caption"]
    return img, cap

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Pipeline loader
# ──────────────────────────────────────────────────────────────────────────────
def load_pipes(model_id: str):
    """Load shared-weight T2I and I2I pipelines for a given SD checkpoint."""
    pipe_t2i = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=DTYPE, safety_checker=None
    ).to(DEVICE)
    # Keep default PNDM scheduler for reproducibility
    pipe_i2i = StableDiffusionImg2ImgPipeline(
        vae=pipe_t2i.vae,
        text_encoder=pipe_t2i.text_encoder,
        tokenizer=pipe_t2i.tokenizer,
        unet=pipe_t2i.unet,
        scheduler=pipe_t2i.scheduler,
        safety_checker=None,
        feature_extractor=None,
    ).to(DEVICE)
    return pipe_t2i, pipe_i2i

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Sharding + resumable helpers
# ──────────────────────────────────────────────────────────────────────────────
items_per_shard = math.ceil(cfg.N_PER_MODEL_PER_MODE / cfg.NUM_SHARDS)
shard_start     = cfg.SHARD_ID * items_per_shard
shard_end       = min(cfg.N_PER_MODEL_PER_MODE, shard_start + items_per_shard)
shard_ids       = list(range(shard_start, shard_end))
print(f"Shard covers global_id [{shard_start}, {shard_end}) => {len(shard_ids)} items")


def img_path(model_name: str, mode: str, global_id: int) -> str:
    d = os.path.join(IMG_DIR, model_name, mode)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{global_id:05d}.png")


def meta_path(model_name: str, mode: str) -> str:
    return os.path.join(META_DIR, f"{model_name}_{mode}_sh{cfg.SHARD_ID:02d}.jsonl")


def already_done(model_name: str, mode: str, global_id: int) -> bool:
    return os.path.exists(img_path(model_name, mode, global_id))


def write_meta(fp, rec: dict) -> None:
    fp.write(json.dumps(rec) + "\n")
    fp.flush()


def seed_for(global_id: int, model_name: str) -> int:
    offset = 0 if model_name == "sd14" else 10_000_000
    return cfg.SEED_BASE + offset + global_id

# ──────────────────────────────────────────────────────────────────────────────
# 7.  Generation loop
# ──────────────────────────────────────────────────────────────────────────────
for model_name, model_id in cfg.MODELS.items():
    print(f"\n=== Loading: {model_name}  ({model_id}) ===")
    pipe_t2i, pipe_i2i = load_pipes(model_id)

    f_weak = open(meta_path(model_name, "weak"), "a")
    f_text = open(meta_path(model_name, "text"), "a")
    f_i2i  = open(meta_path(model_name, "i2i"),  "a")

    try:
        for global_id in tqdm(shard_ids, desc=f"{model_name} shard {cfg.SHARD_ID}"):
            init_img, caption = get_coco_item(global_id)
            seed = seed_for(global_id, model_name)

            # ── Mode 1: weak-conditioned ──────────────────────────────────────
            if not already_done(model_name, "weak", global_id):
                try:
                    g   = torch.Generator(device=DEVICE).manual_seed(seed)
                    out = pipe_t2i(
                        prompt=cfg.PROMPT_WEAK,
                        guidance_scale=cfg.CFG_WEAK,
                        num_inference_steps=cfg.STEPS,
                        height=cfg.IMG_SIZE,
                        width=cfg.IMG_SIZE,
                        generator=g,
                    ).images[0]
                    p = img_path(model_name, "weak", global_id)
                    out.save(p)
                    write_meta(f_weak, {
                        "model": model_name, "mode": "weak",
                        "global_id": global_id, "seed": seed,
                        "prompt": cfg.PROMPT_WEAK, "cfg": cfg.CFG_WEAK,
                        "steps": cfg.STEPS, "path": p,
                    })
                except Exception as e:
                    write_meta(f_weak, {
                        "model": model_name, "mode": "weak",
                        "global_id": global_id,
                        "error": str(e), "trace": traceback.format_exc()[:2000],
                    })

            # ── Mode 2: text-conditioned ──────────────────────────────────────
            if not already_done(model_name, "text", global_id):
                try:
                    g   = torch.Generator(device=DEVICE).manual_seed(seed)
                    out = pipe_t2i(
                        prompt=caption,
                        guidance_scale=cfg.CFG_TEXT,
                        num_inference_steps=cfg.STEPS,
                        height=cfg.IMG_SIZE,
                        width=cfg.IMG_SIZE,
                        generator=g,
                    ).images[0]
                    p = img_path(model_name, "text", global_id)
                    out.save(p)
                    write_meta(f_text, {
                        "model": model_name, "mode": "text",
                        "global_id": global_id, "seed": seed,
                        "prompt": caption, "cfg": cfg.CFG_TEXT,
                        "steps": cfg.STEPS, "manifest": MANIFEST_PATH, "path": p,
                    })
                except Exception as e:
                    write_meta(f_text, {
                        "model": model_name, "mode": "text",
                        "global_id": global_id,
                        "error": str(e), "trace": traceback.format_exc()[:2000],
                    })

            # ── Mode 3: image-to-image ────────────────────────────────────────
            if not already_done(model_name, "i2i", global_id):
                try:
                    g   = torch.Generator(device=DEVICE).manual_seed(seed)
                    out = pipe_i2i(
                        prompt=caption,
                        image=init_img,
                        strength=cfg.I2I_STRENGTH,
                        guidance_scale=cfg.CFG_I2I,
                        num_inference_steps=cfg.STEPS,
                        generator=g,
                    ).images[0]
                    p = img_path(model_name, "i2i", global_id)
                    out.save(p)
                    write_meta(f_i2i, {
                        "model": model_name, "mode": "i2i",
                        "global_id": global_id, "seed": seed,
                        "prompt": caption, "cfg": cfg.CFG_I2I,
                        "strength": cfg.I2I_STRENGTH,
                        "steps": cfg.STEPS, "manifest": MANIFEST_PATH, "path": p,
                    })
                except Exception as e:
                    write_meta(f_i2i, {
                        "model": model_name, "mode": "i2i",
                        "global_id": global_id,
                        "error": str(e), "trace": traceback.format_exc()[:2000],
                    })

    finally:
        f_weak.close()
        f_text.close()
        f_i2i.close()

print(f"\n✅ Done shard {cfg.SHARD_ID}")
print("Images  :", IMG_DIR)
print("Meta    :", META_DIR)
print("Manifest:", MANIFEST_PATH)
