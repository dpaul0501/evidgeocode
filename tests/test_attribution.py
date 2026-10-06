"""Behavioural checks on a tiny random CLIP (CPU, no downloads)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
from transformers import CLIPConfig, CLIPModel  # noqa: E402

from evidgeo.attribution import CLIP_MEAN, CLIP_STD, ClipAttributor  # noqa: E402


@pytest.fixture(scope="module")
def attr():
    torch.manual_seed(0)
    cfg = CLIPConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_attention_heads=2,
                         num_hidden_layers=2, vocab_size=100, max_position_embeddings=77),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_attention_heads=2,
                           num_hidden_layers=2, image_size=224, patch_size=32),
        projection_dim=16,
    )
    cfg._attn_implementation = "eager"
    a = object.__new__(ClipAttributor)
    a.device, a.size, a.mask_mode = "cpu", 224, "mean"
    a.model = CLIPModel(cfg).eval()
    for p in a.model.parameters():
        p.requires_grad_(False)
    a.mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    a.std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    a.vit_grid = 7
    return a


@pytest.fixture
def batch():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 224, 224)
    t1 = F.normalize(torch.randn(2, 16), dim=-1)
    t2 = F.normalize(torch.randn(2, 16), dim=-1)
    return x, t1, t2


def test_grad_pec_depends_on_prompt_but_saliency_does_not(attr, batch):
    x, t1, t2 = batch
    a, b = attr.grad_maps(x, t1, 7), attr.grad_maps(x, t2, 7)
    assert a["pec"].shape == (2, 7, 7)
    assert not np.allclose(a["pec"], b["pec"])
    assert np.allclose(a["sal"], b["sal"])


@pytest.mark.parametrize("g", [5, 7, 9])
def test_rollout_and_loo_shapes(attr, batch, g):
    x, t1, _ = batch
    r, emb = attr.rollout(x, g)
    assert r.shape == (2, g, g) and np.all(r >= 0)
    assert emb.shape == (2, 16)
    drops, base = attr.loo(x, t1, g)
    assert drops.shape == (2, g, g) and base.shape == (2,)


@pytest.mark.parametrize("mode", ["mean", "zero", "blur"])
def test_mask_modes_hide_one_patch(attr, batch, mode):
    attr.mask_mode = mode
    x = batch[0][:1]
    masks = attr.patch_masks(7)
    y = attr._masked(x.expand(49, -1, -1, -1), masks)
    changed = (y != x).any(1)  # (49, H, W)
    assert torch.equal(changed[0, 32:, :], torch.zeros_like(changed[0, 32:, :]))
    assert changed[0, :32, :32].any()
    attr.mask_mode = "mean"


def test_faithfulness_keys(attr, batch):
    x, t1, _ = batch
    maps = {"pec": attr.grad_maps(x, t1, 7)["pec"]}
    out = attr.faithfulness(x, t1, maps, 7, steps=4)
    assert set(out) == {"pec_del_auc", "pec_ins_auc", "random_del_auc", "random_ins_auc"}
    assert out["pec_del_auc"].shape == (2,)
