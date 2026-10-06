#!/usr/bin/env python
"""Generate the modern-model extension of EvidGeo (Colab or Kaggle GPU).

Design (fixes the v2 confound where "weak" changed prompt AND guidance):
  cfg<s>   real COCO caption, guidance scale s  (the guidance sweep)
  generic  prompt "a photo" at the model's default guidance (= v2 "weak")
  i2i      real COCO image + caption, strength 0.75, default guidance

Captions are rows of the master manifest, so every image pairs with its real
COCO image (hf_row = gid). Several seeds per prompt give seed uncertainty;
--traj-prompts saves decoded x0 previews along the denoising trajectory.

Layout (read by scripts/run_eval.py):
  <out>/images/<model>/<mode>/<gid:05d>_seed<k>.png
  <out>/traj/<model>/<mode>/<gid:05d>_seed<k>_step<t>.png
  <out>/meta/<model>_sh<i>.jsonl
  <out>/cache/manifest_master.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidgeo import data as D  # noqa: E402
from evidgeo.generators import MODELS, load_model  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--master-manifest", required=True)
    ap.add_argument("--prompts-jsonl", default="",
                    help="Benchmark prompts {global_id, caption} instead of COCO (no real pairs, no i2i)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-gid", type=int, default=0)
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--modes", nargs="+", default=["sweep", "generic", "i2i"])
    ap.add_argument("--sweep", type=float, nargs="*", default=None,
                    help="Guidance values; default is the model's own sweep")
    ap.add_argument("--strength", type=float, default=0.75)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--traj-prompts", type=int, default=0,
                    help="Save x0 previews for the first N prompts (seed 0 only)")
    ap.add_argument("--traj-every", type=int, default=4)
    ap.add_argument("--shard", default="0/1", help="i/n: this process handles prompts i::n")
    ap.add_argument("--hf-dataset", default="Erland/coco_captions_small")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--hf-shuffle-seed", type=int, default=42)
    ap.add_argument("--seed-base", type=int, default=20261005)
    ap.add_argument("--device-map", default=None,
                    help='"balanced" splits the pipeline over all GPUs (e.g. Kaggle 2xT4)')
    ap.add_argument("--offload", action="store_true",
                    help="CPU-offload idle submodules (needed for 16 GB GPUs with large text encoders)")
    return ap.parse_args()


def jobs_for(args, spec):
    sweep = args.sweep if args.sweep is not None else spec.sweep
    modes = []
    for m in args.modes:
        if m == "sweep":
            modes += [(f"cfg{g:g}", g, "caption") for g in sweep]
        elif m == "generic":
            modes.append(("generic", spec.default_guidance, "generic"))
        elif m == "i2i":
            if args.prompts_jsonl or not spec.has_i2i:
                print(f"skipping i2i for {spec.name}")
                continue
            modes.append(("i2i", spec.default_guidance, "caption"))
        else:
            raise ValueError(m)
    return modes


def main():
    args = parse_args()
    spec = MODELS[args.model]
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    os.makedirs(os.path.join(args.out, "cache"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "meta"), exist_ok=True)

    src = args.prompts_jsonl or args.master_manifest
    manifest = D.find_manifest("", src)
    dst = os.path.join(args.out, "cache",
                       "manifest_master.jsonl" if not args.prompts_jsonl else "manifest_prompts.jsonl")
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    gids = [g for g in sorted(manifest)[args.first_gid:args.first_gid + args.n_prompts]]
    gids = gids[shard_i::shard_n]
    modes = jobs_for(args, spec)
    print(f"{spec.name}: {len(gids)} prompts x {len(modes)} modes x {args.seeds} seeds "
          f"= {len(gids) * len(modes) * args.seeds} images")

    coco = None
    if any(m[0] == "i2i" for m in modes):
        from datasets import load_dataset

        coco = load_dataset(args.hf_dataset, split=args.hf_split).shuffle(
            seed=args.hf_shuffle_seed).select(range(max(gids) + 1))

    print(f"GPUs visible: {torch.cuda.device_count()} "
          f"{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}", flush=True)
    gen = load_model(spec, steps=args.steps, size=args.size, offload=args.offload,
                     device_map=args.device_map)
    meta_fp = open(os.path.join(args.out, "meta", f"{spec.name}_sh{shard_i:02d}.jsonl"), "a")
    t0, done = time.time(), 0
    for gid in gids:
        cap = manifest[gid]
        for mode, guidance, ptype in modes:
            prompt = cap if ptype == "caption" else "a photo"
            for k in range(args.seeds):
                path = os.path.join(args.out, "images", spec.name, mode, f"{gid:05d}_seed{k}.png")
                if os.path.exists(path):
                    continue
                os.makedirs(os.path.dirname(path), exist_ok=True)
                seed = args.seed_base + 1000 * gid + k  # same seed across models and modes
                traj_dir = None
                if k == 0 and gids.index(gid) < args.traj_prompts:
                    traj_dir = os.path.join(args.out, "traj", spec.name, mode)
                    os.makedirs(traj_dir, exist_ok=True)
                init = None
                if mode == "i2i":
                    init = coco[gid]["image"].convert("RGB").resize((args.size, args.size))
                rec = dict(model=spec.name, repo=spec.repo, mode=mode, global_id=gid, seed=seed,
                           seed_index=k, prompt=prompt, guidance=guidance,
                           guidance_param=spec.guidance_param, steps=gen.steps, size=args.size,
                           strength=args.strength if init is not None else None, path=path)
                try:
                    img, traj = gen(prompt, guidance, seed, init=init, strength=args.strength,
                                    traj_every=args.traj_every if traj_dir else 0)
                    tmp = path + ".tmp.png"
                    img.save(tmp)
                    os.replace(tmp, path)
                    for step, im in traj:
                        im.save(os.path.join(traj_dir, f"{gid:05d}_seed{k}_step{step:03d}.png"))
                    rec["traj_steps"] = [s for s, _ in traj]
                except torch.cuda.OutOfMemoryError as e:
                    rec["error"] = f"OOM: {e}"
                    torch.cuda.empty_cache()
                meta_fp.write(json.dumps(rec) + "\n")
                meta_fp.flush()
                done += 1
                print(f"  {mode} seed{k}: {time.time() - t0:.0f}s elapsed, {done} images", flush=True)
        rate = done / max(1e-9, time.time() - t0)
        print(f"gid {gid}: {done} images, {rate:.2f} img/s", flush=True)


if __name__ == "__main__":
    main()
