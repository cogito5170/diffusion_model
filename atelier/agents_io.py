"""How judgement-requiring steps obtain their artifacts.

``claude_code`` backend (FileDropBackend). The orchestrator writes a task file
``tasks/<step>__<key>.task.json`` and pauses. The main Claude Code session invokes the
named subagent (.claude/agents/<agent>.md), which writes its JSON artifact to
``inbox/<step>__<key>.json``. ``atelier resume`` then validates it strictly. An invalid
output is moved to ``rejected/`` and the task is re-issued with the errors attached,
up to ``agents.max_rejections_per_step`` times.

``offline`` backend (OfflineBackend). Deterministic stubs and heuristic proxies from
offline_agents.py, used for mock runs and tests. Their output is labelled
(producer.kind = offline_stub / heuristic) and is never presented as agent judgement.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atelier import schemas
from atelier.util import atomic_write_json, utc_now

STEPS: dict[str, tuple[str, str]] = {
    "research": ("concept-researcher", "atelier.research_notes.v1"),
    "concepts": ("concept-generator", "atelier.concept_set.v1"),
    "art_direction_draft": ("art-director", "atelier.art_direction_draft.v1"),
    "material_spec": ("material-specialist", "atelier.material_spec.v1"),
    "critique": ("art-critic", "atelier.critique.v1"),
    "curation": ("curator", "atelier.curation.v1"),
}


class AgentOutputError(ValueError):
    """The agent produced something that is not even parseable JSON."""


@dataclass
class AgentRequest:
    step: str
    key: str
    inputs: dict[str, str]                   # name -> path (relative to run dir)
    required_values: dict[str, Any]
    instructions: str
    context: dict[str, Any] = field(default_factory=dict)  # in-memory inputs for offline stubs

    @property
    def agent(self) -> str:
        return STEPS[self.step][0]

    @property
    def schema_version(self) -> str:
        return STEPS[self.step][1]

    @property
    def slug(self) -> str:
        return f"{self.step}__{self.key}"


class AgentBackend(ABC):
    name = "abstract"
    synchronous = False  # True: obtain() always answers immediately, so rejections are retried in-step

    @abstractmethod
    def obtain(self, req: AgentRequest, store, previous_errors: list[str], attempt: int) -> dict[str, Any] | None:
        """Return the (unvalidated) artifact, or None while it is still pending."""

    def task_path(self, req: AgentRequest, store) -> Path | None:
        return None

    def on_rejected(self, req: AgentRequest, store, errors: list[str]) -> None:
        pass


class FileDropBackend(AgentBackend):
    name = "claude_code"

    def task_path(self, req: AgentRequest, store) -> Path:
        return store.root / "tasks" / f"{req.slug}.task.json"

    def inbox_path(self, req: AgentRequest, store) -> Path:
        return store.root / "inbox" / f"{req.slug}.json"

    def obtain(self, req, store, previous_errors, attempt):
        inbox = self.inbox_path(req, store)
        if inbox.exists():
            try:
                return json.loads(inbox.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise AgentOutputError(f"{inbox.name} is not valid JSON: {exc}") from None
        task = {
            "schema_version": "atelier.agent_task.v1",
            "task_id": f"task_{store.root.name.split('_', 1)[-1]}.{req.slug}.a{attempt}",
            "run_id": store.root.name,
            "step": req.step,
            "key": req.key,
            "agent": req.agent,
            "inputs": req.inputs,
            "output_path": store.rel(inbox),
            "output_schema": str(schemas.schema_path(req.schema_version)),
            "expected_schema_version": req.schema_version,
            "required_values": req.required_values,
            "instructions": req.instructions,
            "previous_errors": previous_errors,
            "attempt": attempt,
            "created_at": utc_now(),
        }
        path = self.task_path(req, store)
        existing = json.loads(path.read_text()) if path.exists() else None
        if not existing or existing.get("attempt") != attempt:
            schemas.validate(task, "atelier.agent_task.v1")
            atomic_write_json(path, task)
        return None

    def on_rejected(self, req, store, errors):
        inbox = self.inbox_path(req, store)
        if inbox.exists():
            n = len(list((store.root / "rejected").glob(f"{req.slug}.*.json"))) + 1
            inbox.replace(store.root / "rejected" / f"{req.slug}.{n}.json")
            atomic_write_json(store.root / "rejected" / f"{req.slug}.{n}.errors.json", {"errors": errors})


class OfflineBackend(AgentBackend):
    name = "offline"
    synchronous = True

    def obtain(self, req, store, previous_errors, attempt):
        from atelier import offline_agents

        fn = getattr(offline_agents, req.step)
        return fn(req, store)


def make_backend(name: str) -> AgentBackend:
    if name == "claude_code":
        return FileDropBackend()
    if name == "offline":
        return OfflineBackend()
    raise ValueError(f"unknown agent backend {name!r}")
