# EvidGeo — Evidence Geometry Benchmark for Generative Models

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**EvidGeo** measures how *evidence geometry* — the spatial distribution of semantically relevant patches — varies across guidance modes, model families, and random seeds in latent diffusion models.

The benchmark addresses a core question in generative model interpretability:

> *Do different guidance regimes (text CFG, image-to-image, weak conditioning) produce systematically different patterns of evidence concentration, and are those patterns stable?*

---

## Status

Results are being regenerated with the corrected pipeline in `evidgeo/` and `scripts/` (prompt-aware Grad-PEC, exact mass-preserving SpatialShap, full corpora, paired statistics). Numbers from earlier notebook runs are not reported here until they are reproduced by `scripts/analyze.py`.

---

## Repository Structure

```
evidgeo/                  # library: attribution maps, smoothing, metrics, data discovery
scripts/run_eval.py       # resumable per-image extraction (all corpora)
scripts/analyze.py        # tables + numbers.tex for the paper
notebooks/EvidGeo_full_eval.ipynb   # Colab driver for the full run
tests/                    # unit tests (CPU, no downloads except CLIP for smoke runs)
datagen_coco.py           # generator for multiguide_coco_v2
eval_evidgeo.py           # original analysis script (superseded; kept for reference)
```

---

## Dataset

Images are generated from the **[Erland/coco_captions_small](https://huggingface.co/datasets/Erland/coco_captions_small)** HuggingFace dataset using two Stable Diffusion checkpoints and three guidance modes.

| Corpus | Contents |
|--------|----------|
| v2 guidance | SD 1.4, SD 1.5 × {weak, text, i2i}; BigGAN-deep-256 × {weak, text}; 3,750 captions (some shards missing, ~27.4k images) |
| v3 guidance | same models/modes on 1,249 further captions (~10k images) |
| consistency | 150 prompts × 3 seeds × 3 modes (1,350 images) |
| real | the paired COCO image for every caption (HF rows, shuffle seed 42) |

SD settings (v2): 512×512, 50 PNDM steps, CFG 7.5 for text and i2i, i2i strength 0.75. BigGAN "text" = truncation 0.4, "weak" = truncation 1.0 (BigGAN is class-conditional, so this is a proxy).

### Guidance Mode Definitions

- **weak** — Generic prompt `"a photo"` at CFG = 4.0 (caption-free conditioning baseline).
- **text** — Full text-guided synthesis from the COCO caption at CFG = 7.5.
- **i2i** — Image-to-image conditioning: COCO source image denoised at strength = 0.75 with the caption as text prompt.

---

## Evaluation Metrics

All metrics are computed over a **7×7 patch grid** extracted from CLIP ViT-B/32 features.

### Concentration Metrics (per image)

| Metric | Description |
|--------|-------------|
| `PEC_gini` | Gini coefficient of Grad-PEC patch attributions |
| `PEC_top10` | Fraction of total Grad-PEC mass in top 10% of patches |
| `PEC_entropy` | Shannon entropy (nats) of the Grad-PEC map |
| `ATTN_gini` | Gini coefficient of ViT attention-rollout map |
| `ATTN_entropy` | Shannon entropy of attention-rollout map |
| `CLIPScore` | CLIP image–text alignment score (2.5 × cosine logit / 100) |

### Attribution Methods

| Method | Description |
|--------|-------------|
| **Grad-PEC** | Gradient of CLIP visual-projection norm w.r.t. input pixel patches. 1 forward + 1 backward pass. |
| **Attn Rollout** | Propagated ViT self-attention weights through all 12 transformer layers (Abnar & Zuidema, 2020). |
| **SpatialShap** | Leave-one-out Shapley attribution: CLIP-score drop per masked patch (50 passes). |
| **ShapleyFlow** | Personalized PageRank (α = 0.85) over a 4-connected patch graph seeded by SpatialShap. |

### Stability Metrics (per prompt, cross-seed)

| Metric | Description |
|--------|-------------|
| `sem_var` | Mean squared deviation of CLIP image embeddings across seeds |
| `PEC_gini_var` | Variance of per-image Gini coefficient across seeds |

---

## Evaluation Modules

| Module | Output CSVs | Description |
|--------|-------------|-------------|
| **A** | `A_empirical_v2_perimage.csv`, `A_empirical_v2_table.csv` | Per-image and per-bucket aggregation of CLIPScore, Grad-PEC, and attention metrics |
| **B** | `B_consistency_perimage.csv`, `B_consistency_perprompt.csv`, `B_consistency_table.csv`, `B_consistency_embs.npy` | Cross-seed semantic-embedding variance and PEC stability |
| **C** | `C_mechanistic_shapley.csv` | Shapley + ShapleyFlow attribution for SD images with captions |
| **D** | `D_detection_results.csv` | Logistic-regression probe: can features distinguish guidance modes? |
| **F** | `F_correlation_gini_vs_semvar.csv` | Pearson/Spearman correlation between Gini and semantic variance |

---

## Installation

```bash
pip install diffusers transformers accelerate safetensors \
            ftfy datasets pillow tqdm \
            lpips scikit-image scikit-learn \
            networkx matplotlib pandas scipy
```

**CUDA is strongly recommended** for both generation and evaluation.

---

## Usage

### Step 1 — Generate the Dataset

Edit `datagen_coco.py` at the `Cfg` block:

```python
cfg.BASE     = "/your/output/root"   # e.g. Google Drive path for Colab
cfg.SHARD_ID = 0                     # 0 .. NUM_SHARDS-1
```

**First run** (build caption manifest only):
```bash
cfg.BUILD_MANIFEST_ONLY = True
python datagen_coco.py
```

**Generation shards** (run once per `SHARD_ID`):
```bash
cfg.BUILD_MANIFEST_ONLY = False
for SHARD in 0 1 2 3 4 5 6 7 8 9; do
    python datagen_coco.py  # set cfg.SHARD_ID = $SHARD first
done
```

Output layout:
```
<BASE>/
├── images/
│   ├── sd14/  {weak,text,i2i}/  00000.png … 03749.png
│   └── sd15/  {weak,text,i2i}/  …
├── meta/
│   └── {model}_{mode}_sh{id}.jsonl
└── cache/
    └── manifest_*.jsonl
```

### Step 2 — Run Evaluation

```python
cfg.V2_ROOT = "/your/output/root"
cfg.OUT_DIR = "/your/output/root/paper_eval"
python eval_evidgeo.py
```

All CSVs are written to `cfg.OUT_DIR`.

---

## Colab Quick-Start

Both scripts are designed to run in Google Colab with Drive mounted:

```python
from google.colab import drive
drive.mount("/content/drive")
# then run datagen_coco.py or eval_evidgeo.py cell-by-cell
```

Set `cfg.BASE` to a Google Drive path for persistent storage across sessions.

---

## Reproducibility

| Aspect | Value |
|--------|-------|
| COCO shuffle seed | 42 |
| Generation seed base | 777 (SD 1.4) / 10 000 777 (SD 1.5) |
| Eval random seed | 1337 |
| Scheduler | PNDM (default, not overridden) |
| CLIP model | `openai/clip-vit-base-patch32` |

All seeds are fixed deterministically per `(model, global_id)` so any shard can be re-run without changing other shards' outputs.

---

## Citation

If you use EvidGeo in your research, please cite:

```bibtex
@misc{evidgeo2025,
  title   = {Evidence Geometry in Generative Models},
  author  = {<authors>},
  year    = {2025},
  note    = {ECCV 2026 submission}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
