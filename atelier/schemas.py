"""Versioned JSON Schema registry and strict validation.

Artifacts are self-describing: ``schema_version`` (e.g. ``atelier.critique.v1``) selects
the schema. Unknown versions, unknown fields and missing fields are rejected; nothing
is coerced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
BASE_URI = "https://atelier.invalid/schemas/"

# schema_version -> (schema file, identifier field)
ARTIFACT_TYPES: dict[str, tuple[str, str]] = {
    "atelier.creative_brief.v1": ("creative_brief.v1.json", "brief_id"),
    "atelier.research_notes.v1": ("research_notes.v1.json", "research_id"),
    "atelier.concept_set.v1": ("concept_set.v1.json", "concept_set_id"),
    "atelier.art_direction_draft.v1": ("art_direction_draft.v1.json", "draft_id"),
    "atelier.material_spec.v1": ("material_spec.v1.json", "material_spec_id"),
    "atelier.art_direction.v1": ("art_direction.v1.json", "art_direction_id"),
    "atelier.generation_request.v1": ("generation_request.v1.json", "request_id"),
    "atelier.generation_result.v1": ("generation_result.v1.json", "result_id"),
    "atelier.qa_report.v1": ("qa_report.v1.json", "qa_id"),
    "atelier.critique.v1": ("critique.v1.json", "critique_id"),
    "atelier.selection.v1": ("selection.v1.json", "selection_id"),
    "atelier.iteration_plan.v1": ("iteration_plan.v1.json", "plan_id"),
    "atelier.curation.v1": ("curation.v1.json", "curation_id"),
    "atelier.approval.v1": ("approval.v1.json", "approval_id"),
    "atelier.run_config.v1": ("run_config.v1.json", "config_id"),
    "atelier.run_state.v1": ("run_state.v1.json", "run_id"),
    "atelier.run_report.v1": ("run_report.v1.json", "report_id"),
    "atelier.agent_task.v1": ("agent_task.v1.json", "task_id"),
}


class SchemaValidationError(ValueError):
    """Raised when an artifact does not conform to its declared schema."""

    def __init__(self, schema_version: str | None, errors: list[str]):
        self.schema_version = schema_version
        self.errors = errors
        head = "; ".join(errors[:5]) + (f" (+{len(errors) - 5} more)" if len(errors) > 5 else "")
        super().__init__(f"{schema_version or '<unknown schema>'}: {head}")


@dataclass(frozen=True)
class ArtifactType:
    schema_version: str
    schema_file: str
    id_field: str

    @property
    def kind(self) -> str:
        return self.schema_version.split(".")[1]


@lru_cache(maxsize=1)
def _registry() -> Registry:
    resources = []
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        contents = json.loads(path.read_text(encoding="utf-8"))
        if contents.get("$id") != BASE_URI + path.name:
            raise RuntimeError(f"schema {path.name} has unexpected $id {contents.get('$id')!r}")
        resources.append((contents["$id"], Resource.from_contents(contents)))
    return Registry().with_resources(resources)


@lru_cache(maxsize=None)
def _validator(schema_file: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / schema_file).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=_registry())


def schema_path(schema_version: str) -> Path:
    return SCHEMA_DIR / artifact_type(schema_version).schema_file


def artifact_type(schema_version: str) -> ArtifactType:
    try:
        schema_file, id_field = ARTIFACT_TYPES[schema_version]
    except KeyError:
        raise SchemaValidationError(schema_version, [f"unknown schema_version {schema_version!r}"]) from None
    return ArtifactType(schema_version, schema_file, id_field)


def _format_error(err) -> str:
    loc = "/" + "/".join(str(p) for p in err.absolute_path)
    return f"{loc}: {err.message}"


def validation_errors(artifact: Any, expected_schema_version: str | None = None) -> list[str]:
    if not isinstance(artifact, dict):
        return [f"artifact must be a JSON object, got {type(artifact).__name__}"]
    version = artifact.get("schema_version")
    if not isinstance(version, str):
        return ["missing or non-string schema_version"]
    if expected_schema_version is not None and version != expected_schema_version:
        return [f"expected schema_version {expected_schema_version!r}, got {version!r}"]
    if version not in ARTIFACT_TYPES:
        return [f"unknown schema_version {version!r}"]
    validator = _validator(ARTIFACT_TYPES[version][0])
    errors = sorted(validator.iter_errors(artifact), key=lambda e: (list(e.absolute_path), e.message))
    return [_format_error(e) for e in errors]


def validate(artifact: Any, expected_schema_version: str | None = None) -> ArtifactType:
    """Validate strictly; return the artifact type or raise SchemaValidationError."""
    errors = validation_errors(artifact, expected_schema_version)
    if errors:
        version = artifact.get("schema_version") if isinstance(artifact, dict) else None
        raise SchemaValidationError(version, errors)
    return artifact_type(artifact["schema_version"])


def artifact_id(artifact: dict) -> str:
    return artifact[artifact_type(artifact["schema_version"]).id_field]


def check_all_schemas() -> list[str]:
    """Meta-validate every schema file; returns the list of checked files."""
    checked = []
    for schema_file in sorted({f for f, _ in ARTIFACT_TYPES.values()}):
        _validator(schema_file)
        checked.append(schema_file)
    return checked
