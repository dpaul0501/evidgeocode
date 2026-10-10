"""Modern text-to-image generators with a common call signature.

    gen = load_model(MODELS["sd35-medium"])
    image, traj = gen(prompt, guidance, seed, init=None, strength=0.75, traj_every=4)

`traj` is a list of (step, PIL image) previews of the predicted clean image
x0 at intermediate steps, plus the final image as the last entry.

All four models use FlowMatchEulerDiscreteScheduler, where
    x_t = sigma_t * noise + (1 - sigma_t) * x0,   x_{t'} = x_t + (sigma_t' - sigma_t) * v
so from two consecutive latents  v = (x_{t'} - x_t) / (sigma_t' - sigma_t)  and
x0 = x_t - sigma_t * v.  The step callback only sees latents (never the model
output), so x0 is recovered this way.

Guidance semantics differ per model (checked against diffusers source, 2026-10):
  sd35-*            true CFG, `guidance_scale`; 1.0 = unguided
  flux2-klein-base  true CFG, `guidance_scale`; 1.0 = unguided. (The distilled
                    FLUX.2-klein-4B ignores guidance, so it cannot be swept.)
  qwen-image        true CFG, `true_cfg_scale`, active only with a negative prompt
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class Spec:
    name: str
    repo: str
    t2i: str
    i2i: str | None
    guidance_param: str
    default_guidance: float
    sweep: tuple
    steps: int
    min_diffusers: str
    license: str
    vram_gb: int
    extra: dict = field(default_factory=dict)
    inpaint: str | None = None      # diffusers inpainting pipeline (shares weights)
    edit_via_image: bool = False    # t2i pipeline accepts image= as edit conditioning

    @property
    def has_i2i(self) -> bool:
        return self.i2i is not None


MODELS = {s.name: s for s in [
    Spec("sd35-medium", "stabilityai/stable-diffusion-3.5-medium", "StableDiffusion3Pipeline",
         "StableDiffusion3Img2ImgPipeline", "guidance_scale", 4.5, (1, 2, 3.5, 5, 7, 9), 28,
         "0.31.0", "Stability Community (gated: accept terms, set HF_TOKEN)", 16,
         inpaint="StableDiffusion3InpaintPipeline"),
    Spec("sd35-large", "stabilityai/stable-diffusion-3.5-large", "StableDiffusion3Pipeline",
         "StableDiffusion3Img2ImgPipeline", "guidance_scale", 4.0, (1, 2, 3.5, 5, 7, 9), 28,
         "0.31.0", "Stability Community (gated)", 32, inpaint="StableDiffusion3InpaintPipeline"),
    Spec("flux2-klein-base", "black-forest-labs/FLUX.2-klein-base-4B", "Flux2KleinPipeline",
         None, "guidance_scale", 4.0, (1, 2, 4, 6, 8), 28, "0.37.1", "Apache-2.0", 16,
         inpaint="Flux2KleinInpaintPipeline", edit_via_image=True),
    Spec("qwen-image", "Qwen/Qwen-Image", "QwenImagePipeline", "QwenImageImg2ImgPipeline",
         "true_cfg_scale", 4.0, (1, 2, 4, 6), 30, "0.35.0", "Apache-2.0", 80,
         extra={"negative_prompt": " "}, inpaint="QwenImageInpaintPipeline"),
]}


class Generator:
    def __init__(self, spec: Spec, steps: int | None = None, size: int = 512,
                 device: str = "cuda", dtype=torch.bfloat16, offload: bool = False,
                 device_map: str | None = None):
        import diffusers

        self.spec, self.size, self.device = spec, size, device
        self.steps = steps or spec.steps
        if not torch.cuda.is_bf16_supported():  # T4 / P100
            dtype = torch.float16
        self.dtype = dtype
        cls = getattr(diffusers, spec.t2i)
        if device_map:  # e.g. "balanced": spread components over all visible GPUs
            self.pipe = cls.from_pretrained(spec.repo, torch_dtype=dtype, device_map=device_map)
        else:
            self.pipe = cls.from_pretrained(spec.repo, torch_dtype=dtype)
            if offload:
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        if hasattr(self.pipe.vae, "enable_tiling"):
            # i2i encodes the source photo; without tiling this needs a ~4.5 GB block (OOM on T4)
            self.pipe.vae.enable_tiling()
        self.i2i = self._derive(spec.i2i, offload) if spec.i2i else None
        self.inpaint = self._derive(spec.inpaint, offload) if spec.inpaint else None

    def _derive(self, cls_name: str, offload: bool):
        """Another pipeline on the same modules. Not from_pipe(): that calls .to() and
        would pull a multi-GPU (device_map) pipeline back onto one device."""
        import inspect

        import diffusers

        cls = getattr(diffusers, cls_name)
        params = inspect.signature(cls.__init__).parameters
        p = cls(**{k: v for k, v in self.pipe.components.items() if k in params})
        if getattr(self.pipe, "hf_device_map", None):
            p.hf_device_map = self.pipe.hf_device_map
        elif offload:
            p.enable_model_cpu_offload()
        p.set_progress_bar_config(disable=True)
        return p

    # ------------------------------------------------------------ decoding
    @torch.no_grad()
    def decode(self, pipe, lat: torch.Tensor) -> Image.Image:
        vae = pipe.vae
        lat = lat.to(vae.device)  # with device_map the VAE may sit on another GPU
        name = self.spec.name
        if name.startswith("sd35"):
            x = lat / vae.config.scaling_factor + vae.config.shift_factor
            img = vae.decode(x.to(vae.dtype)).sample
        elif name.startswith("flux2"):
            b, n, c = lat.shape
            h = w = int(round(n ** 0.5))
            x = lat.transpose(1, 2).reshape(b, c, h, w)
            std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
            x = x * std.to(x) + vae.bn.running_mean.view(1, -1, 1, 1).to(x)
            x = pipe._unpatchify_latents(x)
            img = vae.decode(x.to(vae.dtype)).sample
        elif name.startswith("qwen"):
            x = pipe._unpack_latents(lat, self.size, self.size, pipe.vae_scale_factor)
            mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(x)
            std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(x)
            img = vae.decode((x * std + mean).to(vae.dtype)).sample[:, :, 0]
        else:
            raise ValueError(name)
        img = (img.float().clamp(-1, 1) + 1) / 2
        return Image.fromarray((img[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))

    # ------------------------------------------------------------- calling
    def __call__(self, prompt: str, guidance: float, seed: int, init: Image.Image | None = None,
                 strength: float = 0.75, traj_every: int = 0, mask: Image.Image | None = None,
                 edit: Image.Image | None = None):
        """init -> img2img (SDEdit); init + mask -> inpainting (white = regenerate);
        edit -> instruction editing with the image as conditioning (FLUX.2)."""
        if mask is not None:
            pipe = self.inpaint
        elif init is not None:
            pipe = self.i2i
        else:
            pipe = self.pipe
        if pipe is None:
            raise ValueError(f"{self.spec.name} lacks the pipeline for this task")
        if edit is not None and not self.spec.edit_via_image:
            raise ValueError(f"{self.spec.name} does not support image editing")
        traj, state = [], {"prev": None, "last": 0}

        def cb(p, i, t, kw):
            lat = kw["latents"]
            sched = p.scheduler
            k = sched.step_index  # already advanced past step i
            if traj_every and state["prev"] is not None and (i % traj_every == 0):
                s0, s1 = sched.sigmas[k - 1].item(), sched.sigmas[k].item()
                v = (lat.float() - state["prev"].float()) / (s1 - s0)
                x0 = state["prev"].float() - s0 * v
                traj.append((i, self.decode(p, x0.to(lat.dtype))))
            state["prev"] = lat.detach().clone() if traj_every else None
            state["last"] = i + 1
            return kw

        kwargs = dict(prompt=prompt, num_inference_steps=self.steps,
                      generator=torch.Generator(self.device).manual_seed(seed),
                      callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
        kwargs[self.spec.guidance_param] = guidance
        kwargs.update(self.spec.extra)
        if mask is not None:
            kwargs.update(image=init, mask_image=mask, strength=strength,
                          height=self.size, width=self.size)
        elif init is not None:
            kwargs.update(image=init, strength=strength)
        else:
            kwargs.update(height=self.size, width=self.size)
            if edit is not None:
                kwargs.update(image=edit)
        image = pipe(**kwargs).images[0]
        if traj_every:
            traj.append((state["last"], image))  # i2i runs fewer steps than self.steps
        return image, traj


def load_model(spec: Spec, **kw) -> Generator:
    return Generator(spec, **kw)
