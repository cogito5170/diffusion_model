"""Deterministic procedural mock engine: a TEST DOUBLE, not an image model.

It renders a shaded "metal blob" whose appearance responds to keywords in the compiled
prompt (materials, lighting direction, negative space, palette words). That lets the
pipeline's feedback loop be exercised offline. Its images say nothing about real
model quality.

It mirrors the sd-turbo capability profile so compilation behaves as it would for the
real adapter. Failure injection (per slot, per attempt) supports the tests.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from atelier.budget import CancelToken
from atelier.engines.base import EngineCapabilities, EngineError, EngineOutput, ImageEngine
from atelier.engines.diffusers_local import PROFILES, profile_capabilities
from atelier.util import stable_int


class SimulatedCrash(BaseException):
    """Simulates the process dying mid-batch (like KeyboardInterrupt); used by resume tests."""


COLOR_WORDS: dict[str, tuple[int, int, int]] = {
    "black": (12, 12, 14), "graphite": (52, 54, 58), "gunmetal": (62, 66, 72), "slate": (78, 88, 98),
    "titanium": (135, 134, 128), "steel": (150, 156, 162), "silver": (196, 200, 204), "chrome": (214, 218, 224),
    "bone": (227, 218, 201), "ivory": (238, 232, 216), "white": (246, 246, 246),
    "cobalt": (40, 72, 168), "amber": (214, 146, 58), "oxblood": (98, 22, 28), "bronze": (150, 110, 62),
    "teal": (30, 110, 112), "rust": (150, 70, 40),
}


def _palette(prompt: str) -> list[tuple[int, int, int]]:
    found = []
    for m in re.finditer(r"[a-z]+", prompt):
        if m.group(0) in COLOR_WORDS and COLOR_WORDS[m.group(0)] not in found:
            found.append(COLOR_WORDS[m.group(0)])
    return found or [COLOR_WORDS["graphite"], COLOR_WORDS["chrome"]]


def render(prompt: str, width: int, height: int, seed: int) -> Image.Image:
    p = prompt.lower()
    rng = np.random.default_rng(seed)  # geometry depends on the seed only (fixed across refinements)
    detail_rng = np.random.default_rng(stable_int(seed, p, bits=63))  # prompt-dependent detail
    side = max(width, height)
    v, u = np.mgrid[0:height, 0:width].astype(np.float32) / side
    cx0, cy0 = 0.5 * width / side, 0.5 * height / side
    if "off-center" in p or "thirds" in p:
        cx0 = 0.36 * width / side
    if "low angle" in p:
        cy0 += 0.08
    scale = {"generous negative space": 0.6, "minimal negative space": 1.25}.get(
        next((k for k in ("generous negative space", "minimal negative space") if k in p), ""), 0.9)

    field = np.zeros_like(u)
    for _ in range(int(rng.integers(2, 5))):
        cx = cx0 + rng.normal(0, 0.08) * scale
        cy = cy0 + rng.normal(0, 0.08) * scale
        s = rng.uniform(0.08, 0.17) * scale
        field += rng.uniform(0.6, 1.0) * np.exp(-((u - cx) ** 2 + (v - cy) ** 2) / (2 * s * s))
    if "liquid" in p or "fluid" in p:
        field += 0.04 * np.sin(u * 40 + detail_rng.uniform(0, 6)) * np.sin(v * 33) * (field > 0.2)
    mask = field > 0.32 * field.max()

    gy, gx = np.gradient(field)
    k = 35.0
    n = np.stack([-gx * k * side / 100, -gy * k * side / 100, np.ones_like(field)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    light = np.array([-0.45, -0.65, 0.6])
    if "from the right" in p or "from right" in p:
        light = np.array([0.6, -0.35, 0.7])
    elif "from above" in p or "overhead" in p:
        light = np.array([0.0, -0.8, 0.6])
    light /= np.linalg.norm(light)
    shininess = 120.0 if any(w in p for w in ("mirror", "liquid chrome", "polished")) else 60.0
    if "brushed" in p:
        shininess = 28.0
    if "oxidized" in p or "matte" in p:
        shininess = 8.0
    spec_gain = 1.0 if "hard" in p else 0.55

    ndl = np.clip((n * light).sum(-1), 0, 1)
    r = 2 * ndl[..., None] * n - light
    spec = np.clip(r[..., 2], 0, 1) ** shininess

    colors = _palette(p)
    lum = [sum(c) for c in colors]
    dark = np.array(colors[int(np.argmin(lum))], np.float32) / 255
    light_c = np.array(colors[int(np.argmax(lum))], np.float32) / 255
    horizon = 1 / (1 + np.exp(-n[..., 1] * 14))  # chrome-like sky/ground split in the reflection
    env = dark[None, None] * horizon[..., None] + light_c[None, None] * (1 - horizon[..., None])
    metal = env * (0.35 + 0.65 * ndl[..., None]) + spec_gain * spec[..., None]

    if "brushed" in p:
        streak = detail_rng.normal(0, 1, size=(height, 1)).astype(np.float32)
        streak = np.repeat(streak, width, axis=1) + 0.3 * detail_rng.normal(0, 1, size=(height, width))
        metal *= (1 + 0.07 * streak)[..., None]

    bg_top = dark * 0.5
    bg_bottom = dark * 0.9 + 0.03
    t = (np.arange(height, dtype=np.float32) / max(height - 1, 1))[:, None, None]
    bg = bg_top[None, None] * (1 - t) + bg_bottom[None, None] * t
    bg = np.broadcast_to(bg, (height, width, 3)).copy()
    if "industrial" in p:
        for x in range(int(width * 0.1), width, max(8, width // 7)):
            bg[:, x:x + 2] *= 1.35
    img = np.where(mask[..., None], metal, bg)

    m = re.search(r"(\d+(?:\.\d)?):1 contrast", p)
    if m:
        squash = min(1.0, float(m.group(1)) / 8.0)
        img = 0.5 + (img - 0.5) * (0.55 + 0.45 * squash)
    if "monochrome" in p:
        g = img @ np.array([0.299, 0.587, 0.114], np.float32)
        img = np.repeat(g[..., None], 3, axis=-1)
    elif "muted" in p:
        g = (img @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
        img = 0.5 * img + 0.5 * g
    out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB")

    if "scratch" in p:
        draw = ImageDraw.Draw(out)
        for _ in range(40):
            x0, y0 = int(detail_rng.integers(0, width)), int(detail_rng.integers(0, height))
            if mask[min(y0, height - 1), min(x0, width - 1)]:
                L = int(detail_rng.integers(6, 30))
                draw.line([(x0, y0), (x0 + L, y0 + int(detail_rng.integers(-2, 3)))], fill=(235, 235, 235), width=1)
    if "soft focus" in p:
        out = out.filter(ImageFilter.GaussianBlur(2))
    return out


class MockEngine(ImageEngine):
    engine_id = "mock"

    def __init__(self, model_id: str = "mock-procedural-v1", *, latency_s: float = 0.0,
                 failures: dict[str, list[str]] | None = None, profile: str = "stabilityai/sd-turbo",
                 cost_usd_per_image: float = 0.0):
        """``failures`` maps a slot ("0", "1", ... or "*") to per-attempt modes:
        ok | transient | error | timeout | malformed | not_an_object | blank | crash."""
        self.model_id = model_id
        self.latency_s = latency_s
        self.failures = failures or {}
        self.profile = PROFILES[profile]
        self.cost_usd_per_image = cost_usd_per_image
        self.calls: list[dict[str, Any]] = []
        self._attempts: dict[str, int] = {}
        self._lock = threading.Lock()

    def capabilities(self) -> EngineCapabilities:
        return profile_capabilities(self.profile, self.engine_id, self.model_id, self.cost_usd_per_image,
                                    cost_basis="mock engine: no compute or API cost")

    def _mode(self, request: dict[str, Any]) -> str:
        with self._lock:
            cid = request["candidate_id"]
            self._attempts[cid] = self._attempts.get(cid, 0) + 1
            n = self._attempts[cid]
        modes = self.failures.get(str(request["slot"])) or self.failures.get("*") or []
        return modes[n - 1] if n - 1 < len(modes) else "ok"

    def generate(self, request: dict[str, Any], out_dir: Path, cancel: CancelToken) -> EngineOutput:
        mode = self._mode(request)
        self.calls.append({"candidate_id": request["candidate_id"], "mode": mode})
        cancel.raise_if_cancelled()
        if self.latency_s:
            time.sleep(self.latency_s)
        if mode == "crash":
            raise SimulatedCrash("simulated process interruption")
        if mode == "sigkill":  # the OS killing the process (e.g. out of memory); only use inside an isolated worker
            import os
            import signal

            os.kill(os.getpid(), signal.SIGKILL)
        if mode == "transient":
            raise EngineError("transient_backend_error", "simulated transient failure", retryable=True)
        if mode == "error":
            raise EngineError("backend_error", "simulated permanent failure", retryable=False)
        if mode == "timeout":
            while not cancel.is_cancelled():
                time.sleep(0.01)
            cancel.raise_if_cancelled()
        if mode == "not_an_object":
            return {"image": "not-a-real-response"}  # type: ignore[return-value]
        path = Path(out_dir) / "image_0.png"
        if mode == "malformed":
            path.write_bytes(b"this is not a png")
            return EngineOutput(images=[path], model_id=self.model_id, seed_used=request["seed"])
        if mode == "blank":
            Image.new("RGB", (request["width"], request["height"]), (128, 128, 128)).save(path)
        else:
            render(request["prompt"]["positive"], request["width"], request["height"], request["seed"] or 0).save(path)
        return EngineOutput(images=[path], model_id=self.model_id, seed_used=request["seed"],
                            metadata={"mode": mode, "renderer": "procedural-numpy", "not_a_generative_model": True})
