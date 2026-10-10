"""Shared fixtures for the offline test suite (no GPU, no network, no API keys)."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from atelier.config import deep_merge, validate_config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def brief() -> dict:
    return load(EXAMPLES / "briefs" / "metallic_identity.json")


def art_direction() -> dict:
    return load(EXAMPLES / "artifacts" / "art_direction.example.json")


def example(name: str) -> dict:
    return load(EXAMPLES / "artifacts" / f"{name}.example.json")


def config(**overrides) -> dict:
    cfg = load(ROOT / "configs" / "mock_offline.json")
    return validate_config(deep_merge(cfg, overrides)) if overrides else validate_config(cfg)


class TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="atelier-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


def clone(obj):
    return copy.deepcopy(obj)


def new_run(tmp: Path, cfg: dict | None = None, engine=None, agents=None, run_id: str = "run_t", b: dict | None = None):
    from atelier.orchestrator import Orchestrator

    return Orchestrator.create(tmp, run_id, b or brief(), cfg or config(), engine=engine, agents=agents)
