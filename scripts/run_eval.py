#!/usr/bin/env python
"""Extract evidence maps and per-image scalars for every EvidGeo image.

Resumable: work is split into chunks keyed by gid range, and a chunk whose
output exists is skipped. Re-running after a Colab disconnect continues.

Outputs under --out:
  features/<corpus>/<dataset>/<model>/<mode>/c<NNNNN>.parquet   scalars
  maps/<corpus>/<dataset>/<model>/<mode>/c<NNNNN>.npz           raw maps + CLIP image embeddings
  run_config.json                                               exact arguments + versions
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidgeo import data as D  # noqa: E402
from evidgeo import metrics as M  # noqa: E402
from evidgeo import smoothing as S  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guidance-roots", nargs="*", default=[],
                    help="Folders with images/<model>/<mode>/<gid>.png")
    ap.add_argument("--consistency-root", default="")
    ap.add_argument("--consistency-model", default="sd",
                    help="Generator of the consistency corpus (not recorded on Drive)")
    ap.add_argument("--master-manifest", default="",
                    help="Captions in HF shuffle order covering every corpus "
                         "(multiguide_coco_v2/cache/manifest_v3_N5000_seed42.jsonl)")
    ap.add_argument("--trajectory-roots", nargs="*", default=[],
                    help="Roots with traj/<model>/<mode>/<gid>_seed<k>_step<t>.png")
    ap.add_argument("--external", nargs="*", default=[],
                    help="JSONL manifests of other image sets (faces, memes, ...)")
    ap.add_argument("--real", action="store_true", help="Also evaluate the paired real COCO images")
    ap.add_argument("--hf-dataset", default="Erland/coco_captions_small")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--hf-shuffle-seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grids", type=int, nargs="+", default=[7])
    ap.add_argument("--mask-mode", default="mean", choices=["mean", "zero", "blur"])
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--alpha-sweep", type=float, nargs="*", default=[0.5, 0.7, 0.95])
    ap.add_argument("--sigmas", type=float, nargs="*", default=[1.0, 2.0])
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--faith-per-bucket", type=int, default=0,
                    help="Deletion/insertion curves for the first N gids of each bucket")
    ap.add_argument("--limit-per-bucket", type=int, default=0, help="Smoke-test cap")
    ap.add_argument("--only", nargs="*", default=[], help="Restrict to corpus/model/mode substrings")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--io-threads", type=int, default=16)
    ap.add_argument("--clip", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def collect(args) -> list[D.Record]:
    master = ([c for _, c in sorted(D.find_manifest("", args.master_manifest).items())]
              if args.master_manifest else [])
    print(f"master caption order: {len(master)} rows")
    recs: list[D.Record] = []
    for root in args.guidance_roots:
        recs += D.guidance_records(root, master)
    if args.consistency_root:
        recs += D.consistency_records(args.consistency_root, master, args.consistency_model)
    for root in args.trajectory_roots:
        recs += D.trajectory_records(root, master)
    ext = [r for j in args.external for r in D.external_records(j)]
    unpaired = sum(1 for r in recs if r.hf_row < 0)
    recs += ext
    if unpaired:
        print(f"WARNING: {unpaired} generated images have no master row (no real pair)")
    if args.real:
        real = D.real_records({r.hf_row for r in recs if r.hf_row >= 0}, master, args.hf_dataset)
        recs += D.cache_real_images(real, os.path.join(args.out, "real_cache"), args.hf_dataset,
                                    args.hf_split, args.hf_shuffle_seed)
    if args.only:
        recs = [r for r in recs if any(s in r.key() for s in args.only)]
    missing = sum(1 for r in recs if not r.caption)
    if missing:
        print(f"WARNING: {missing} records have no caption and will be skipped")
        recs = [r for r in recs if r.caption]
    return recs


def bucket_of(r: D.Record):
    return (r.corpus, r.dataset, r.model, r.mode)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a.ravel(), b.ravel()).statistic)


def per_map_scalars(prefix: str, maps: dict[str, np.ndarray], i: int) -> dict:
    row = {}
    for name, arr in maps.items():
        row.update(M.concentration(arr[i], f"{prefix}{name}"))
    return row


def process_chunk(attr, recs, imgs, args) -> tuple[pd.DataFrame, dict]:
    rows, store = [], {}
    caps = [r.caption for r in recs]
    for s in range(0, len(recs), args.batch):
        x = attr.preprocess(imgs[s:s + args.batch])
        txt = attr.text_emb(caps[s:s + args.batch])
        batch_rows = [dict(D.asdict(r)) for r in recs[s:s + args.batch]]
        for g in args.grids:
            pre = f"g{g}_"
            maps = attr.grad_maps(x, txt, g)
            if len(caps[s:s + args.batch]) > 1:
                # prompt-swap control: same image, another image's caption
                swap = attr.grad_maps(x, txt.roll(1, dims=0), g)["pec"]
                bc = caps[s:s + args.batch]
                for i, row in enumerate(batch_rows):
                    same = bc[i] == bc[i - 1]  # consistency seeds share a caption
                    row[f"{pre}pec_swap_rho"] = np.nan if same else spearman(maps["pec"][i], swap[i])
                    row[f"{pre}pec_sal_rho"] = spearman(maps["pec"][i], maps["sal"][i])
            maps["rollout"], emb = attr.rollout(x, g)
            cos = (emb * txt).sum(-1).double().cpu().numpy()  # full-image CLIP similarity
            for i, row in enumerate(batch_rows):
                row["clip_cos"] = float(cos[i])
            if not args.no_loo:
                drops, base = attr.loo(x, txt, g)
                pos = np.clip(drops, 0, None)
                maps["loo"] = pos
                maps["ss"] = np.stack([S.ppr_smooth(m, args.alpha) for m in pos])
                for a in args.alpha_sweep:
                    maps[f"ss{int(a * 100)}"] = np.stack([S.ppr_smooth(m, a) for m in pos])
                for sg in args.sigmas:
                    maps[f"gauss{sg:g}"] = np.stack([S.gaussian_smooth(m, sg) for m in pos])
            for i, row in enumerate(batch_rows):
                row.update(per_map_scalars(pre, maps, i))
                if not args.no_loo:
                    neg = np.clip(-drops[i], 0, None).sum()
                    row[f"{pre}loo_pos_mass"] = float(pos[i].sum())
                    row[f"{pre}loo_neg_frac"] = float(neg / (neg + pos[i].sum() + 1e-12))
                    row[f"{pre}ss_peak_shift"] = int(np.argmax(maps["ss"][i]) != np.argmax(pos[i]))
                    for sg in args.sigmas:
                        row[f"{pre}gauss{sg:g}_leak"] = S.mass_leak(pos[i], sg)
                        row[f"{pre}gauss{sg:g}_peak_shift"] = int(
                            np.argmax(maps[f"gauss{sg:g}"][i]) != np.argmax(pos[i]))

            for name in ("pec", "sal", "rollout", "loo", "ss"):
                if name in maps:
                    store.setdefault(f"g{g}_{name}", []).append(maps[name].astype(np.float32))
            if g == args.grids[0]:
                store.setdefault("clip_emb", []).append(emb.float().cpu().numpy())
                if args.faith_per_bucket and not args.no_loo:
                    sel = [i for i, r in enumerate(recs[s:s + args.batch])
                           if r.gid < args.faith_per_bucket_cutoff.get(bucket_of(r), -1)]
                    if sel:
                        fm = {k: maps[k][sel] for k in ("pec", "rollout", "loo", "ss", "gauss1", "gauss2")
                              if k in maps}
                        f = attr.faithfulness(x[sel], txt[sel], fm, g)
                        for j, i in enumerate(sel):
                            for k, v in f.items():
                                batch_rows[i][f"{pre}{k}"] = float(v[j])
        rows += batch_rows
    df = pd.DataFrame(rows)
    df["clipscore"] = 2.5 * df["clip_cos"].clip(lower=0) if "clip_cos" in df else np.nan
    return df, {k: np.concatenate(v) for k, v in store.items()}


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = collect(args)
    print(f"{len(recs)} images found")
    recs.sort(key=lambda r: (bucket_of(r), r.gid, r.seed, r.step))

    # deterministic faithfulness subset: first N gids of each bucket
    args.faith_per_bucket_cutoff = {}
    for b, grp in groupby(recs, key=bucket_of):
        gids = sorted({r.gid for r in grp})
        if args.faith_per_bucket and gids:
            args.faith_per_bucket_cutoff[b] = gids[min(args.faith_per_bucket, len(gids)) - 1] + 1

    import torch
    import transformers

    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        cfg = {k: v for k, v in vars(args).items() if k != "faith_per_bucket_cutoff"}
        cfg.update(torch=torch.__version__, transformers=transformers.__version__,
                   python=platform.python_version(), n_images=len(recs),
                   started=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(cfg, f, indent=2)

    from evidgeo.attribution import ClipAttributor

    attr = ClipAttributor(args.clip, args.device, mask_mode=args.mask_mode)

    jobs = []
    for b, grp in groupby(recs, key=bucket_of):
        grp = list(grp)
        if args.limit_per_bucket:
            grp = grp[:args.limit_per_bucket]
        sub = os.path.join(*b)
        for cid, chunk in groupby(grp, key=lambda r: r.gid // args.chunk):
            fp = os.path.join(args.out, "features", sub, f"c{cid:05d}.parquet")
            if not os.path.exists(fp):
                jobs.append((fp, os.path.join(args.out, "maps", sub, f"c{cid:05d}.npz"), list(chunk)))
    print(f"{len(jobs)} chunks to run")

    pool = ThreadPoolExecutor(args.io_threads)
    load = lambda chunk: list(pool.map(D.load_image, chunk))  # noqa: E731
    nxt = pool.submit(load, jobs[0][2]) if jobs else None
    t0 = time.time()
    done = 0
    for j, (fp, mp, chunk) in enumerate(jobs):
        imgs = nxt.result()
        nxt = pool.submit(load, jobs[j + 1][2]) if j + 1 < len(jobs) else None
        df, store = process_chunk(attr, chunk, imgs, args)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        np.savez_compressed(mp, keys=np.array([r.key() for r in chunk]), **store)
        df.to_parquet(fp + ".tmp", index=False)
        os.replace(fp + ".tmp", fp)  # atomic: a chunk is either complete or absent
        done += len(chunk)
        rate = done / (time.time() - t0)
        print(f"[{j + 1}/{len(jobs)}] {fp}  {rate:.1f} img/s", flush=True)


if __name__ == "__main__":
    main()
