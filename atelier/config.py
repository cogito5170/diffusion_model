"""Run configuration: schema-validated plus cross-field checks (no silent coercion)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from atelier import schemas
from atelier.util import read_json

DIMENSIONS = (
    "intent_alignment",
    "material_plausibility",
    "composition",
    "visual_coherence",
    "originality",
    "technical_quality",
)

# Initial experimental weights. They are hypotheses to be tested, not validated constants.
DEFAULT_WEIGHTS = {
    "intent_alignment": 0.25,
    "material_plausibility": 0.20,
    "composition": 0.15,
    "visual_coherence": 0.15,
    "originality": 0.15,
    "technical_quality": 0.10,
}


class ConfigError(ValueError):
    pass


def check_weights(weights: dict[str, float]) -> None:
    if set(weights) != set(DIMENSIONS):
        raise ConfigError(f"weights must have exactly the keys {DIMENSIONS}, got {sorted(weights)}")
    if any(w < 0 for w in weights.values()):
        raise ConfigError("weights must be non-negative")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"weights must sum to 1.0 (got {total:.6f}); they are not renormalized automatically")


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        schemas.validate(cfg, "atelier.run_config.v1")
    except schemas.SchemaValidationError as exc:
        raise ConfigError(str(exc)) from exc
    check_weights(cfg["weights"])
    if cfg["selection"]["shortlist_size"] > cfg["batch_size"] * (1 + cfg["max_iterations"]):
        raise ConfigError("shortlist_size exceeds the maximum number of candidates the run can produce")
    return cfg


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = read_json(Path(path))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return validate_config(cfg)


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
