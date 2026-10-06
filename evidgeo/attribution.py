"""Patch-level evidence maps from a CLIP backbone.

All maps are returned as float64 arrays of shape (B, G, G) and are NOT yet
normalised, so callers can inspect raw magnitudes; use metrics.normalize.

Maps
----
pec      Grad-PEC: |d cos(f_I(x), f_T(t)) / dx| pooled per patch. Prompt-aware.
sal      Prompt-free saliency |d ||f_I(x)|| / dx|, i.e. the quantity the
         original eval_evidgeo.py computed. Kept only as an ablation.
rollout  Attention rollout (Abnar & Zuidema 2020) from the CLS token.
loo      Leave-one-out utility drop u(V) - u(V \\ {i}), u = cos(f_I, f_T).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def patch_bounds(size: int, g: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, size, g + 1).round().astype(int)
    return list(zip(edges[:-1], edges[1:]))


def to_grid(maps: torch.Tensor, g: int) -> torch.Tensor:
    """Resample (B, H, W) maps to (B, g, g) by area averaging."""
    if maps.shape[-1] != maps.shape[-2] or maps.shape[-1] % g != 0:
        maps = F.interpolate(maps[:, None], size=(g * 32, g * 32), mode="bilinear",
                             align_corners=False)[:, 0]
    return F.adaptive_avg_pool2d(maps[:, None], g)[:, 0]


@dataclass
class ClipAttributor:
    model_id: str = "openai/clip-vit-base-patch32"
    device: str = "cuda"
    size: int = 224
    mask_mode: str = "mean"  # mean | zero | blur

    def __post_init__(self):
        from transformers import CLIPModel, CLIPTokenizer

        self.model = CLIPModel.from_pretrained(self.model_id, attn_implementation="eager")
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = CLIPTokenizer.from_pretrained(self.model_id)
        self.mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self.vit_grid = self.model.config.vision_config.image_size // \
            self.model.config.vision_config.patch_size

    # ---- encoders -------------------------------------------------------
    def preprocess(self, imgs: list[Image.Image]) -> torch.Tensor:
        """Square resize (no crop, so every pixel is attributed) and normalise."""
        arr = np.stack([np.asarray(im.convert("RGB").resize((self.size, self.size),
                                                            Image.BICUBIC)) for im in imgs])
        x = torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std

    def image_emb(self, x: torch.Tensor, output_attentions: bool = False):
        out = self.model.vision_model(pixel_values=x, output_attentions=output_attentions,
                                      return_dict=True)
        emb = self.model.visual_projection(out.pooler_output)
        return (emb, out.attentions) if output_attentions else emb

    @torch.no_grad()
    def text_emb(self, texts: list[str]) -> torch.Tensor:
        t = self.tok(texts, padding=True, truncation=True, max_length=77,
                     return_tensors="pt").to(self.device)
        out = self.model.text_model(input_ids=t["input_ids"],
                                    attention_mask=t["attention_mask"], return_dict=True)
        return F.normalize(self.model.text_projection(out.pooler_output), dim=-1)

    def similarity(self, x: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        return (F.normalize(self.image_emb(x), dim=-1) * txt).sum(-1)

    # ---- gradient maps --------------------------------------------------
    def grad_maps(self, x: torch.Tensor, txt: torch.Tensor, g: int) -> dict:
        """Grad-PEC (prompt-aware) and the prompt-free saliency ablation."""
        out = {}
        for name in ("pec", "sal"):
            xi = x.detach().clone().requires_grad_(True)
            emb = self.image_emb(xi)
            if name == "pec":
                score = (F.normalize(emb, dim=-1) * txt).sum()
            else:
                score = emb.norm(dim=-1).sum()
            score.backward()
            out[name] = to_grid(xi.grad.abs().mean(1), g).double().cpu().numpy()
        return out

    # ---- attention rollout ---------------------------------------------
    @torch.no_grad()
    def rollout(self, x: torch.Tensor, g: int) -> tuple[np.ndarray, torch.Tensor]:
        emb, attns = self.image_emb(x, output_attentions=True)
        b, _, t, _ = attns[0].shape
        eye = torch.eye(t, device=x.device).expand(b, t, t)
        r = eye.clone()
        for a in attns:
            a = a.float().mean(1)
            a = 0.5 * (a + eye)
            a = a / a.sum(-1, keepdim=True)
            r = a @ r
        cls = r[:, 0, 1:].reshape(b, self.vit_grid, self.vit_grid)
        return to_grid(cls, g).double().cpu().numpy(), F.normalize(emb, dim=-1)

    # ---- occlusion ------------------------------------------------------
    def _masked(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """keep: (N, H, W) bool, True = visible. x: (N, 3, H, W)."""
        if self.mask_mode == "mean":
            fill = torch.zeros_like(x)            # CLIP mean colour after normalisation
        elif self.mask_mode == "zero":
            fill = ((0 - self.mean) / self.std).expand_as(x)   # black pixels
        elif self.mask_mode == "blur":
            k = self.size // 7 | 1
            fill = F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)
        else:
            raise ValueError(self.mask_mode)
        keep = keep[:, None].to(x.dtype)
        return x * keep + fill * (1 - keep)

    def patch_masks(self, g: int) -> torch.Tensor:
        """(g*g, H, W) bool, mask k hides patch k only."""
        b = patch_bounds(self.size, g)
        m = torch.ones(g * g, self.size, self.size, dtype=torch.bool, device=self.device)
        for r, (r0, r1) in enumerate(b):
            for c, (c0, c1) in enumerate(b):
                m[r * g + c, r0:r1, c0:c1] = False
        return m

    @torch.no_grad()
    def loo(self, x: torch.Tensor, txt: torch.Tensor, g: int, chunk: int = 512):
        """Returns (B, g, g) LOO drops and (B,) full-image utility."""
        masks = self.patch_masks(g)
        n = g * g
        base = self.similarity(x, txt)
        drops = torch.empty(x.shape[0], n, device=x.device)
        for i in range(x.shape[0]):
            xi = x[i:i + 1].expand(n, -1, -1, -1)
            sims = torch.cat([self.similarity(self._masked(xi[s:s + chunk], masks[s:s + chunk]),
                                              txt[i:i + 1])
                              for s in range(0, n, chunk)])
            drops[i] = base[i] - sims
        return drops.view(-1, g, g).double().cpu().numpy(), base.double().cpu().numpy()

    @torch.no_grad()
    def faithfulness(self, x: torch.Tensor, txt: torch.Tensor, maps: dict, g: int,
                     steps: int = 16, seed: int = 0) -> dict:
        """Deletion / insertion AUC for each map (mean utility along the curve).

        Deletion hides the highest-evidence patches first (lower AUC = more
        faithful); insertion reveals them first onto a fully masked image
        (higher = more faithful). A random patch order gives the reference.
        """
        n = g * g
        ks = np.unique(np.linspace(0, n, steps + 1).round().astype(int))
        b = patch_bounds(self.size, g)
        rng = np.random.default_rng(seed)
        out = {}
        for i in range(x.shape[0]):
            orders = {k: np.argsort(-v[i].ravel(), kind="stable") for k, v in maps.items()}
            orders["random"] = rng.permutation(n)
            for name, order in orders.items():
                keep_del = torch.ones(len(ks), self.size, self.size, dtype=torch.bool,
                                      device=self.device)
                keep_ins = torch.zeros_like(keep_del)
                for j, k in enumerate(ks):
                    for p in order[:k]:
                        (r0, r1), (c0, c1) = b[p // g], b[p % g]
                        keep_del[j, r0:r1, c0:c1] = False
                        keep_ins[j, r0:r1, c0:c1] = True
                xi = x[i:i + 1].expand(len(ks), -1, -1, -1)
                t = txt[i:i + 1]
                d = self.similarity(self._masked(xi, keep_del), t).double().cpu().numpy()
                s = self.similarity(self._masked(xi, keep_ins), t).double().cpu().numpy()
                out.setdefault(f"{name}_del_auc", []).append(float(np.trapezoid(d, ks / n)))
                out.setdefault(f"{name}_ins_auc", []).append(float(np.trapezoid(s, ks / n)))
        return {k: np.asarray(v) for k, v in out.items()}
