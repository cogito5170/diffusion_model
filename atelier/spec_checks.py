"""Semantic checks for art direction that JSON Schema cannot express.

These checks never fix anything themselves. Errors cause the artifact to be rejected and
sent back to the producing agent. Warnings are recorded and the run continues.
"""

from __future__ import annotations

import re
from typing import Any

# Engine-specific prompt syntax must not leak into the engine-independent spec.
_ENGINE_SYNTAX = [
    (re.compile(r"\([^()]+:\s*\d+(\.\d+)?\)"), "attention-weight syntax like (word:1.2)"),
    (re.compile(r"--(ar|v|stylize|chaos|seed)\b"), "Midjourney-style flags"),
    (re.compile(r"<lora:[^>]+>"), "LoRA tags"),
    (re.compile(r"\bBREAK\b"), "prompt BREAK keyword"),
]

# The negation may govern a list: "no text, logos, signage or watermark".
_NEGATION = re.compile(r"\b(no|not|without|never|avoid|free of|absence of)\b[\w\s,/&-]*$")

_SPECULAR_TYPES = {"liquid_chrome", "polished_chrome", "mirror_steel", "polished_titanium"}
_BRUSHED_TYPES = {"brushed_steel", "brushed_aluminum"}


def _strings(obj: Any, path: str = ""):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}/{i}")


def _ratio(s: str) -> float:
    a, b = s.split(":")
    return float(a) / float(b)


def check_art_direction(ad: dict[str, Any], brief: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    out = ad["output"]
    declared = _ratio(out["aspect_ratio"])
    actual = out["width"] / out["height"]
    if abs(actual - declared) / declared > 0.03:
        errors.append(f"/output: {out['width']}x{out['height']} does not match aspect_ratio {out['aspect_ratio']}")

    ids = [m["material_id"] for m in ad.get("materials", [])]
    if len(ids) != len(set(ids)):
        errors.append("/materials: material_id values must be unique")
    for i, m in enumerate(ad.get("materials", [])):
        if m["type"] in _SPECULAR_TYPES and (m["reflectivity"] < 0.6 or m["roughness"] > 0.35):
            errors.append(f"/materials/{i}: {m['type']} with reflectivity {m['reflectivity']} and roughness "
                          f"{m['roughness']} is physically implausible (expect reflectivity>=0.6, roughness<=0.35)")
        if m["type"] in _BRUSHED_TYPES and m["anisotropy"] < 0.3:
            warnings.append(f"/materials/{i}: brushed metal with anisotropy {m['anisotropy']} will read as polished")

    if not any(c["role"] == "dominant" for c in ad["palette"]["colors"]):
        errors.append("/palette/colors: at least one colour must have role 'dominant'")

    avoid = [a["term"].lower() for a in ad["avoidance_constraints"]]
    must_texts = [c["constraint"].lower() for c in ad["semantic_constraints"] if c["priority"] == "must"]
    for term in avoid:
        for t in must_texts:
            m = re.search(rf"\b{re.escape(term)}\b", t)
            # "no visible face" / "without robots" restate the avoidance; only an affirmative mention conflicts.
            if m and not _NEGATION.search(t[: m.start()]):
                errors.append(f"avoidance term {term!r} conflicts with the 'must' constraint {t!r}")

    for path, text in _strings({k: v for k, v in ad.items() if k not in ("schema_version", "provenance")}):
        for pattern, label in _ENGINE_SYNTAX:
            if pattern.search(text):
                errors.append(f"{path}: contains {label}; art direction must stay engine-independent")

    if brief is not None:
        if out["aspect_ratio"] != brief["constraints"]["aspect_ratio"]:
            errors.append(f"/output/aspect_ratio {out['aspect_ratio']} differs from brief {brief['constraints']['aspect_ratio']}")
        for term in brief["constraints"]["must_avoid"]:
            if term.lower() not in avoid:
                errors.append(f"brief must_avoid {term!r} is missing from avoidance_constraints")
        haystack = " ".join(t.lower() for _, t in _strings(ad))
        for item in brief["constraints"]["must_include"]:
            if item.lower() not in haystack:
                warnings.append(f"brief must_include {item!r} is not mentioned anywhere in the art direction")
    return errors, warnings
