"""Local Hugging Face ``diffusers`` adapter: the first real image engine.

Safety defaults:
  * ``allow_download`` is False, so ``from_pretrained`` runs with ``local_files_only=True``
    and never downloads weights unless the config explicitly enables it.
  * torch / diffusers are imported lazily; without them the engine reports itself
    unavailable instead of crashing.

The profile numbers below (download sizes, step ranges) are approximate and were NOT
measured in the environment this adapter was written in. Check them against the model
card before running.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from atelier.budget import Cancelled, CancelToken
from atelier.engines.base import (
    Availability,
    EngineCapabilities,
    EngineError,
    EngineOutput,
    ImageEngine,
    ParamSpec,
)


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    native_side: int
    max_side: int
    steps_default: int
    steps_range: tuple[int, int]
    guidance_default: float
    guidance_range: tuple[float, float]
    negative_prompt_min_guidance: float | None
    approx_download_gb: str
    license_note: str
    preferred_pixels: int = 0  # pixel count the model was trained around; 0 = native_side ** 2
    scheduler: str | None = None  # "dpmpp_2m_karras": good quality at ~20 steps


PROFILES: dict[str, ModelProfile] = {
    "stabilityai/sd-turbo": ModelProfile(
        model_id="stabilityai/sd-turbo",
        native_side=512, max_side=768,
        steps_default=2, steps_range=(1, 4),
        guidance_default=0.0, guidance_range=(0.0, 0.0),
        negative_prompt_min_guidance=1.0,
        approx_download_gb="~2.5 GB (fp16 variant) to ~5 GB (fp32); approximate, unverified here",
        preferred_pixels=768 * 432,
        license_note="Check the current license on the model card (Stability AI licenses for turbo models "
                     "have restricted commercial use). Unresolved for this project.",
    ),
    "stabilityai/sdxl-turbo": ModelProfile(
        model_id="stabilityai/sdxl-turbo",
        native_side=512, max_side=1024,
        steps_default=2, steps_range=(1, 4),
        guidance_default=0.0, guidance_range=(0.0, 0.0),
        negative_prompt_min_guidance=1.0,
        approx_download_gb="~7 GB (fp16 variant); approximate, unverified here",
        license_note="Check the current license on the model card. Unresolved for this project.",
    ),
    "stabilityai/stable-diffusion-xl-base-1.0": ModelProfile(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        native_side=1024, max_side=1536,
        steps_default=20, steps_range=(8, 50),
        guidance_default=6.0, guidance_range=(1.0, 12.0),
        negative_prompt_min_guidance=1.0,
        approx_download_gb="~7 GB (fp16 diffusers files only); approximate, unverified here",
        scheduler="dpmpp_2m_karras",
        license_note="CreativeML Open RAIL++-M (use-based restrictions; commercial use allowed within them). "
                     "Verify on the model card.",
        preferred_pixels=1024 * 1024,
    ),
    "stable-diffusion-v1-5/stable-diffusion-v1-5": ModelProfile(
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        native_side=512, max_side=768,
        steps_default=25, steps_range=(5, 50),
        guidance_default=7.0, guidance_range=(1.0, 15.0),
        negative_prompt_min_guidance=1.0,
        approx_download_gb="~2-4 GB depending on variant; approximate, unverified here",
        license_note="CreativeML OpenRAIL-M (use-based restrictions). Verify on the model card.",
    ),
}


def profile_capabilities(profile: ModelProfile, engine_id: str, model_id: str | None = None,
                         cost_usd_per_image: float = 0.0, cost_basis: str | None = None) -> EngineCapabilities:
    lo, hi = profile.guidance_range
    return EngineCapabilities(
        engine_id=engine_id,
        model_id=model_id or profile.model_id,
        supports_negative_prompt=True,
        negative_prompt_min_guidance=profile.negative_prompt_min_guidance,
        max_prompt_tokens=75,  # CLIP context of 77 tokens minus BOS/EOS
        supports_seed=True,
        max_reference_images=0,  # text-to-image only in this adapter version
        size_multiple_of=8,
        min_side=256,
        max_side=profile.max_side,
        max_pixels=max(profile.max_side * profile.native_side, profile.preferred_pixels),
        native_side=profile.native_side,
        preferred_pixels=profile.preferred_pixels or profile.native_side ** 2,
        params={
            "num_inference_steps": ParamSpec("int", profile.steps_default, *profile.steps_range,
                                             description="denoising steps"),
            "guidance_scale": ParamSpec("float", profile.guidance_default, lo, hi,
                                        description="classifier-free guidance; negative prompts need > 1"),
        },
        cost_usd_per_image=cost_usd_per_image,
        cost_basis=cost_basis or ("local inference: no API charge; electricity, hardware and download "
                                  "bandwidth are not metered"),
        notes=(f"approx download: {profile.approx_download_gb}", profile.license_note),
    )


class DiffusersLocalEngine(ImageEngine):
    engine_id = "diffusers-local"

    def __init__(
        self,
        model_id: str = "stabilityai/sd-turbo",
        *,
        device: str = "auto",
        dtype: str = "auto",
        variant: str | None = None,
        allow_download: bool = False,
        cache_dir: str | None = None,
        cost_usd_per_image: float = 0.0,
        preferred_pixels: int | None = None,
        _pipeline_factory: Callable[..., Any] | None = None,
        _torch: Any = None,
    ):
        if model_id not in PROFILES:
            raise ValueError(f"no capability profile for {model_id!r}; known: {sorted(PROFILES)}")
        self.model_id = model_id
        self.profile = PROFILES[model_id]
        self.device = device
        self.dtype = dtype
        self.variant = variant
        self.allow_download = bool(allow_download)
        self.cache_dir = cache_dir
        self.cost_usd_per_image = cost_usd_per_image
        self.preferred_pixels = preferred_pixels  # e.g. lower for SDXL on a 16 GB Mac
        self._pipeline_factory = _pipeline_factory
        self._torch = _torch
        self._pipe = None
        self._resolved: dict[str, str] = {}

    # ------------------------------------------------------------ capability
    def capabilities(self) -> EngineCapabilities:
        caps = profile_capabilities(self.profile, self.engine_id, self.model_id, self.cost_usd_per_image)
        if self.preferred_pixels:
            import dataclasses

            caps = dataclasses.replace(caps, preferred_pixels=int(self.preferred_pixels))
        return caps

    def availability(self) -> Availability:
        if self._pipeline_factory is not None:
            return Availability(True)
        missing = [m for m in ("torch", "diffusers", "transformers") if importlib.util.find_spec(m) is None]
        if missing:
            return Availability(False, f"python packages not installed: {', '.join(missing)} "
                                       "(see docs/LIVE_GENERATION.md; nothing is installed automatically)")
        if not self.allow_download:
            try:
                from huggingface_hub import try_to_load_from_cache  # type: ignore
            except ImportError:
                return Availability(False, "huggingface_hub not installed; cannot check local model cache")
            hit = try_to_load_from_cache(self.model_id, "model_index.json", cache_dir=self.cache_dir)
            if not isinstance(hit, str):
                return Availability(False, f"{self.model_id} is not in the local Hugging Face cache and "
                                           "allow_download is false (expected download: "
                                           f"{self.profile.approx_download_gb})")
        return Availability(True)

    # ------------------------------------------------------------ execution
    def _torch_mod(self):
        if self._torch is None:
            self._torch = importlib.import_module("torch")
        return self._torch

    def _resolve_device_dtype(self):
        torch = self._torch_mod()
        device = self.device
        if device == "auto":
            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                device = "cuda"
            elif getattr(getattr(torch, "backends", None), "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        dtype_name = self.dtype if self.dtype != "auto" else ("float16" if device in ("cuda", "mps") else "float32")
        self._resolved = {"device": device, "dtype": dtype_name}
        return device, getattr(torch, dtype_name)

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        device, dtype = self._resolve_device_dtype()
        kwargs: dict[str, Any] = {"torch_dtype": dtype, "local_files_only": not self.allow_download}
        if self.variant:
            kwargs["variant"] = self.variant
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        try:
            if self._pipeline_factory is not None:
                pipe = self._pipeline_factory(self.model_id, **kwargs)
            else:
                from diffusers import AutoPipelineForText2Image  # type: ignore

                pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, **kwargs)
        except OSError as exc:
            msg = str(exc)
            if "incomplete" in msg or "missing" in msg or "local_files_only" in msg:
                key = {"stabilityai/stable-diffusion-xl-base-1.0": "sdxl", "stabilityai/sd-turbo": "sd-turbo"}.get(self.model_id)
                fix = f"python scripts/download_model.py {key}" if key else "re-download the model"
                raise EngineError("model_files_incomplete",
                                  f"{self.model_id} is not fully downloaded ({msg}). Fix: {fix}",
                                  retryable=False, details={"fix": fix}) from exc
            raise EngineError("model_load_failed", f"could not load {self.model_id}: {msg}", retryable=False) from exc
        pipe = pipe.to(device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        if device == "mps" and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()  # lowers peak memory on Apple GPUs (diffusers' advice below 64 GB)
            self._resolved["attention_slicing"] = "on"
        if self.profile.scheduler == "dpmpp_2m_karras" and self._pipeline_factory is None:
            try:
                from diffusers import DPMSolverMultistepScheduler  # type: ignore

                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
                self._resolved["scheduler"] = "DPMSolverMultistep (Karras)"
            except Exception:  # noqa: BLE001 - keep the model's default scheduler
                self._resolved["scheduler"] = "default (DPM++ unavailable)"
        if device == "mps":
            # The SDXL VAE upcasts to float32 to decode; tiling bounds that memory spike on 16 GB machines.
            vae = getattr(pipe, "vae", None)
            if vae is not None and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
                self._resolved["vae_tiling"] = "on"
            elif hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
                self._resolved["vae_tiling"] = "on"
        self._pipe = pipe
        return pipe

    def warmup(self, cancel: CancelToken | None = None) -> float:
        import time

        if self._pipe is not None:
            return 0.0
        t0 = time.monotonic()
        self._progress("loading")
        self._load()
        return round(time.monotonic() - t0, 2)

    def close(self) -> None:
        """Release the loaded pipeline so another model can use the memory."""
        self._pipe = None
        import gc

        gc.collect()
        torch = self._torch
        try:
            if torch is not None and getattr(getattr(torch, "backends", None), "mps", None) is not None \
                    and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best effort
            pass

    def generate(self, request: dict[str, Any], out_dir: Path, cancel: CancelToken) -> EngineOutput:
        cancel.raise_if_cancelled()
        if self._pipe is None:
            self._progress("loading")
        pipe = self._load()
        torch = self._torch_mod()
        params = dict(request["engine_params"])
        seed = request["seed"]
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

        total_steps = int(params.get("num_inference_steps", self.profile.steps_default))
        self._progress("step", 0, total_steps)

        def on_step_end(pipeline, step, timestep, callback_kwargs):
            self._progress("step", step + 1, total_steps)
            if cancel.is_cancelled():
                pipeline._interrupt = True  # supported by recent diffusers pipelines
            return callback_kwargs

        call: dict[str, Any] = {
            "prompt": request["prompt"]["positive"],
            "width": request["width"],
            "height": request["height"],
            "num_inference_steps": params.get("num_inference_steps", self.profile.steps_default),
            "guidance_scale": params.get("guidance_scale", self.profile.guidance_default),
            "generator": generator,
            "callback_on_step_end": on_step_end,
        }
        if request["prompt"]["negative"] is not None:
            call["negative_prompt"] = request["prompt"]["negative"]
        try:
            out = pipe(**call)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise EngineError("out_of_memory", str(exc), retryable=False) from exc
            raise EngineError("inference_failed", str(exc), retryable=False) from exc
        if cancel.is_cancelled():
            raise Cancelled("generation interrupted by cancellation/timeout")
        images = getattr(out, "images", None)
        if not images:
            raise EngineError("empty_output", "pipeline returned no images", retryable=False)
        path = Path(out_dir) / "image_0.png"
        images[0].save(path, format="PNG")
        meta = {
            "device": self._resolved.get("device"),
            "dtype": self._resolved.get("dtype"),
            "num_inference_steps": call["num_inference_steps"],
            "guidance_scale": call["guidance_scale"],
            "negative_prompt_sent": "negative_prompt" in call,
            "local_files_only": not self.allow_download,
            "attention_slicing": self._resolved.get("attention_slicing", "off"),
            "vae_tiling": self._resolved.get("vae_tiling", "off"),
            "scheduler": self._resolved.get("scheduler", "default"),
        }
        for mod in ("torch", "diffusers"):
            try:
                meta[f"{mod}_version"] = importlib.import_module(mod).__version__
            except Exception:  # noqa: BLE001 - metadata only
                pass
        return EngineOutput(images=[path], model_id=self.model_id, seed_used=seed, metadata=meta)
