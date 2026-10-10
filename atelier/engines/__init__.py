"""Engine registry. Exactly one real adapter (diffusers-local) plus the mock test double."""

from __future__ import annotations

from typing import Any

from atelier.engines.base import ImageEngine


def create_engine(engine_cfg: dict[str, Any]) -> ImageEngine:
    engine_id = engine_cfg["engine_id"]
    options = dict(engine_cfg.get("options", {}))
    if engine_id == "mock":
        from atelier.engines.mock import MockEngine

        return MockEngine(model_id=engine_cfg["model_id"], **options)
    if engine_id == "diffusers-local":
        from atelier.engines.diffusers_local import DiffusersLocalEngine

        return DiffusersLocalEngine(model_id=engine_cfg["model_id"], **options)
    raise ValueError(f"unknown engine {engine_id!r}")
