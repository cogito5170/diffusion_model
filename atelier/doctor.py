"""Environment inspection before any live generation. Makes no network calls and installs nothing."""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


def _version(mod: str) -> str | None:
    if importlib.util.find_spec(mod) is None:
        return None
    try:
        return getattr(importlib.import_module(mod), "__version__", "installed")
    except Exception as exc:  # noqa: BLE001
        return f"import failed: {exc}"


def _mem_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        return None
    return None


def inspect(runs_dir: str = "runs") -> dict[str, Any]:
    pkgs = {m: _version(m) for m in ("jsonschema", "referencing", "PIL", "numpy", "torch", "diffusers",
                                      "transformers", "accelerate", "huggingface_hub", "pytest")}
    accel: dict[str, Any] = {"cuda": None, "mps": None}
    if pkgs["torch"] and not str(pkgs["torch"]).startswith("import failed"):
        import torch  # type: ignore

        accel["cuda"] = bool(torch.cuda.is_available())
        accel["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    cached = sorted(p.name for p in (Path(hf_home) / "hub").glob("models--*")) if (Path(hf_home) / "hub").exists() else []
    disk_root = Path(runs_dir) if Path(runs_dir).exists() else Path(".")
    free_gb = round(shutil.disk_usage(disk_root).free / 1e9, 1)
    offline_ready = all(pkgs[m] for m in ("jsonschema", "referencing", "PIL", "numpy"))
    live_ready = offline_ready and all(pkgs[m] for m in ("torch", "diffusers", "transformers"))
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_gb": _mem_gb(),
        "disk_free_gb": free_gb,
        "packages": pkgs,
        "accelerators": accel,
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
        "hf_cache": hf_home,
        "hf_cached_models": cached,
        "api_keys_present": {k: bool(os.environ.get(k)) for k in ("HF_TOKEN", "REPLICATE_API_TOKEN", "OPENAI_API_KEY", "GEMINI_API_KEY")},
        "offline_pipeline_ready": offline_ready,
        "live_engine_ready": live_ready,
        "notes": [
            "No network access was attempted.",
            "Live generation (diffusers-local) needs torch + diffusers + transformers and cached model weights; "
            "see docs/LIVE_GENERATION.md for sizes and the explicit opt-in steps.",
        ],
    }


def format_report(info: dict[str, Any]) -> str:
    lines = [f"python {info['python']} on {info['platform']}",
             f"cpus {info['cpu_count']}, memory {info['memory_gb']} GB, free disk {info['disk_free_gb']} GB",
             f"GPU: cuda={info['accelerators']['cuda']} mps={info['accelerators']['mps']} nvidia-smi={info['nvidia_smi']}",
             "packages:"]
    lines += [f"  {k:16s} {v or '-- not installed'}" for k, v in info["packages"].items()]
    lines += [f"HF cache {info['hf_cache']}: {', '.join(info['hf_cached_models']) or 'no models cached'}",
              f"API keys present: {', '.join(k for k, v in info['api_keys_present'].items() if v) or 'none'}",
              f"offline pipeline ready: {info['offline_pipeline_ready']}",
              f"live engine ready:      {info['live_engine_ready']}"]
    lines += [f"note: {n}" for n in info["notes"]]
    return "\n".join(lines)
