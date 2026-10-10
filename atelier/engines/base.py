"""Image-engine adapter interface and capability description.

The compiler negotiates against ``EngineCapabilities``. It never assumes a feature the
engine does not declare. Adapters must report what actually happened: real paths, the
real model id, and the seed that was actually used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from atelier.budget import CancelToken


class CapabilityError(ValueError):
    """The request needs something the engine cannot do (raised at compile time)."""


class EngineError(RuntimeError):
    """Structured engine failure. ``retryable`` marks transient errors."""

    def __init__(self, code: str, message: str, retryable: bool = False, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}


@dataclass(frozen=True)
class ParamSpec:
    type: str                      # "int" | "float" | "bool" | "str"
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] | None = None
    description: str = ""

    def check(self, name: str, value: Any) -> None:
        py = {"int": int, "float": (int, float), "bool": bool, "str": str}[self.type]
        if isinstance(value, bool) and self.type != "bool":
            raise CapabilityError(f"engine param {name!r} must be {self.type}, got bool")
        if not isinstance(value, py):
            raise CapabilityError(f"engine param {name!r} must be {self.type}, got {type(value).__name__}")
        if self.minimum is not None and value < self.minimum:
            raise CapabilityError(f"engine param {name!r}={value} is below minimum {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise CapabilityError(f"engine param {name!r}={value} is above maximum {self.maximum}")
        if self.choices is not None and value not in self.choices:
            raise CapabilityError(f"engine param {name!r}={value!r} not in {self.choices}")


@dataclass(frozen=True)
class EngineCapabilities:
    engine_id: str
    model_id: str
    supports_negative_prompt: bool
    # If set, negative prompts only have an effect when guidance_scale exceeds this value
    # (classifier-free guidance). Distilled "turbo" models run at guidance 0.
    negative_prompt_min_guidance: float | None
    max_prompt_tokens: int | None
    supports_seed: bool
    max_reference_images: int
    size_multiple_of: int
    min_side: int
    max_side: int
    max_pixels: int
    native_side: int
    preferred_pixels: int = 0  # pixel count the model was trained around (0 = native_side ** 2)
    params: Mapping[str, ParamSpec] = field(default_factory=dict)
    cost_usd_per_image: float = 0.0
    cost_basis: str = "unspecified"
    notes: tuple[str, ...] = ()

    def negative_prompt_effective(self, params: Mapping[str, Any]) -> bool:
        if not self.supports_negative_prompt:
            return False
        if self.negative_prompt_min_guidance is None:
            return True
        return float(params.get("guidance_scale", 0.0)) > self.negative_prompt_min_guidance

    def describe(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "model_id": self.model_id,
            "supports_negative_prompt": self.supports_negative_prompt,
            "negative_prompt_min_guidance": self.negative_prompt_min_guidance,
            "max_prompt_tokens": self.max_prompt_tokens,
            "supports_seed": self.supports_seed,
            "max_reference_images": self.max_reference_images,
            "size": {"multiple_of": self.size_multiple_of, "min_side": self.min_side, "max_side": self.max_side,
                     "max_pixels": self.max_pixels, "native_side": self.native_side,
                     "preferred_pixels": self.preferred_pixels or self.native_side ** 2},
            "params": {k: vars(v) for k, v in self.params.items()},
            "cost_usd_per_image": self.cost_usd_per_image,
            "cost_basis": self.cost_basis,
            "notes": list(self.notes),
        }


@dataclass
class EngineOutput:
    images: list[Path]
    model_id: str
    seed_used: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Availability:
    ok: bool
    reason: str = ""


class ImageEngine(ABC):
    engine_id: str = "abstract"
    # Optional hook set by the caller: progress_callback(stage, step, total) with stage "loading" or "step".
    progress_callback = None

    def _progress(self, stage: str, step: int = 0, total: int = 0) -> None:
        cb = self.progress_callback
        if cb is not None:
            try:
                cb(stage, step, total)
            except Exception:  # noqa: BLE001 - progress reporting must never break generation
                pass

    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        ...

    def availability(self) -> Availability:
        return Availability(True)

    @abstractmethod
    def generate(self, request: dict[str, Any], out_dir: Path, cancel: CancelToken) -> EngineOutput:
        """Produce image file(s) in ``out_dir`` for one generation request.

        Must raise EngineError for failures, honour ``cancel`` where possible, and never
        return paths outside ``out_dir``.
        """

    def close(self) -> None:  # pragma: no cover - optional hook
        pass
