#!/usr/bin/env python
"""Paper figures, drawn only from analysis outputs.

    python scripts/make_figures.py --out ieee_tai_paper/v2/figs \
        --run sd35=kaggle_out/sd35m_merged/results --run klein=results_klein/results \
        --run faces=results_tasks/faces/results --run edit=results_tasks/edit/results \
        --edit-root results_tasks/edit \
        [--run trajs=<dense SD3.5 trajectories> --run trajk=<dense FLUX.2 trajectories>]

Palette: first four categorical slots of the reference palette, validated on
white (CVD worst adjacent dE 9.1). Two slots are below 3:1 on white, so every
series also has its own marker shape and a direct label.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = [  # fixed order: identity follows the entity, never its rank
    ("sd35", "SD 3.5 · COCO", "#2a78d6", "o"),
    ("klein", "FLUX.2 · COCO", "#eb6834", "s"),
    ("faces", "FLUX.2 · faces", "#1baf7a", "^"),
    ("edit", "FLUX.2 · editing", "#eda100", "D"),
]

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlabel("guidance scale $s$ (log)")


def sweep_rows(path, measure):
    if measure.endswith("traj_commit_frac"):  # per-prompt trajectory file
        tp = os.path.join(os.path.dirname(path), "uncertainty_trajectory.csv")
        if not os.path.exists(tp):
            return None
        t = pd.read_csv(tp)
        t = t[t["mode"].str.match(r"^(edit-)?cfg[0-9.]+$")]
        if t.empty:
            return None
        t = t.assign(s=t["mode"].str.extract(r"cfg([0-9.]+)$")[0].astype(float))
        g = t.groupby("s")[measure].mean()
        return np.array(list(g.items()))
    d = pd.read_csv(path)
    d = d[d.measure == measure]
    if d.empty:
        return None
    r = d.iloc[0]
    pts = sorted((float(c[8:]), float(r[c])) for c in d.columns if c.startswith("mean_cfg") and not pd.isna(r[c]))
    return np.array(pts)


def guidance_figure(runs, out):
    panels = [("pec_gini", "(a) Grad-PEC concentration (Gini)"),
              ("pec_seed_sem_var", "(b) Semantic variance across seeds"),
              ("pec_traj_commit_frac", "(c) Trajectory share before commit")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), constrained_layout=True)
    for ax, (measure, title) in zip(axes, panels):
        style(ax, title)
        for key, label, color, marker in SERIES:
            if key not in runs:
                continue
            run = runs[key]
            if measure.endswith("traj_commit_frac"):  # dense trajectory runs replace the small sets
                run = runs.get({"sd35": "trajs", "klein": "trajk"}.get(key, ""), run)
            pts = sweep_rows(os.path.join(run, "guidance_sweep.csv"), measure)
            if pts is None or len(pts) < 2:
                continue
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, marker=marker, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.8, label=label, zorder=3)
            # selective direct labels in panel (a) only, nudged apart; the legend carries identity
            if measure == "pec_gini":
                dy = {"edit": 5, "klein": -5}.get(key, 0)
                ax.annotate({"sd35": "SD 3.5", "klein": "COCO", "faces": "faces", "edit": "editing"}[key],
                            pts[-1], xytext=(5, dy), textcoords="offset points", va="center",
                            fontsize=6.5, color=INK2)
        ax.set_xlim(0.9, 13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_guidance.{ext}"), dpi=300)
    plt.close(fig)


def overlay(ax, img, grid, mask=None, title=None):
    from PIL import Image

    ax.imshow(img)
    h, w = img.shape[:2]
    m = np.asarray(Image.fromarray((grid / grid.max() * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)) / 255
    ax.imshow(m, cmap="Blues", alpha=0.55, vmin=0, vmax=1)  # sequential: one hue, light -> dark
    if mask is not None:
        ax.contour(mask, levels=[0.5], colors=[INK], linewidths=0.8)
    ax.set_xticks([]), ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7, color=INK, pad=2)


def edit_figure(edit_root, run_dir, out, gid):
    """One MagicBrush edit: source, human target, FLUX.2 at s=1 and s=8, with maps."""
    from glob import glob

    from PIL import Image

    def img(sub):
        p = os.path.join(edit_root, "edit", "images", sub, f"{gid:05d}_seed0.png")
        return np.asarray(Image.open(p).convert("RGB").resize((224, 224)))

    store = {}
    for p in glob(os.path.join(run_dir, "maps", "**", "*.npz"), recursive=True):
        with np.load(p) as z:
            arrs = {f: z[f] for f in z.files}
        for i, k in enumerate(arrs.pop("keys")):
            if f"/{gid}/" in str(k):
                store[str(k)] = {f: a[i] for f, a in arrs.items()}
    mask = np.asarray(Image.open(os.path.join(edit_root, "edit", "masks", f"{gid:05d}_edit.png"))
                      .convert("L").resize((224, 224))) / 255.0
    cols = [("magicbrush/source", "guidance/edit/magicbrush/source", "Source"),
            ("magicbrush/human-target", "guidance/edit/magicbrush/human-target", "Human edit"),
            ("flux2-klein-base/edit-cfg1", "guidance/edit/flux2-klein-base/edit-cfg1", "FLUX.2, $s{=}1$"),
            ("flux2-klein-base/edit-cfg8", "guidance/edit/flux2-klein-base/edit-cfg8", "FLUX.2, $s{=}8$")]
    fig, axes = plt.subplots(3, 4, figsize=(3.5, 2.75), constrained_layout=True)
    for j, (sub, kp, title) in enumerate(cols):
        im = img(sub)
        key = next((k for k in store if k.startswith(kp + f"/{gid}/")), None)
        axes[0, j].imshow(im)
        axes[0, j].contour(mask, levels=[0.5], colors=["white"], linewidths=0.8)
        axes[0, j].set_xticks([]), axes[0, j].set_yticks([])
        axes[0, j].set_title(title, fontsize=6.5, color=INK, pad=2)
        if key:
            overlay(axes[1, j], im, store[key]["g7_pec"], mask)
            overlay(axes[2, j], im, store[key]["g7_ss"], mask)
    axes[1, 0].set_ylabel("Grad-PEC", fontsize=6.5, color=INK)
    axes[2, 0].set_ylabel("SpatialShap", fontsize=6.5, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_edit_example.{ext}"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--edit-root", default="")
    ap.add_argument("--edit-gid", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = dict(r.split("=", 1) for r in a.run)
    guidance_figure(runs, a.out)
    if a.edit_root and "edit" in runs:
        edit_figure(a.edit_root, os.path.dirname(runs["edit"]), a.out, a.edit_gid)
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
