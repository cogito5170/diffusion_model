"""Deterministic translation of the engine-independent art direction into engine-specific
generation requests, negotiated against ``EngineCapabilities``.

Rules:
  * Unsupported *required* features (e.g. a required reference image) are compile errors.
  * Optional features the engine cannot honour are dropped, and every drop is recorded in
    ``compilation.dropped`` / ``warnings``.
  * Engine parameters are validated against the engine's declared ranges. Out-of-range
    values are rejected, never clamped.
  * Avoidance constraints never enter the positive prompt (naming a thing tends to summon it).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atelier.engines.base import CapabilityError, EngineCapabilities
from atelier.util import sha256_file

COMPILER_VERSION = "1.2.0"

DEFAULT_NEGATIVE_TERMS = ("text", "watermark", "logo", "signature", "lowres", "jpeg artifacts")

MATERIAL_WORDS = {
    "liquid_chrome": "liquid chrome",
    "polished_chrome": "polished chrome",
    "brushed_steel": "brushed steel",
    "mirror_steel": "mirror-polished steel",
    "polished_titanium": "polished titanium",
    "anodized_titanium": "anodized titanium",
    "oxidized_steel": "oxidized steel",
    "brushed_aluminum": "brushed aluminum",
    "gunmetal": "gunmetal",
    "other_metal": "metal",
}
_GENERIC_MATERIAL_IDS = {"primary", "secondary", "main", "base", "material", "metal"}
CURVATURE_WORDS = {
    "flat": "flat planes",
    "gentle": "gently curved",
    "sculptural": "sculptural curvature",
    "fluid": "fluid molten curvature",
    "faceted": "faceted planes",
}
FRAMING_WORDS = {
    "extreme_close_up": "extreme close-up",
    "close_up": "close-up",
    "medium": "medium shot",
    "wide": "wide shot",
    "environmental": "environmental wide view",
}


class CompilationError(CapabilityError):
    pass


def estimate_tokens(text: str) -> int:
    """Approximate CLIP-BPE token count without downloading a tokenizer: common words are
    one token, words over 9 letters count as two, digits as one per two, punctuation as one.
    This is an estimate; the diffusers pipeline truncates beyond 77 real tokens anyway,
    and the budget only decides *which* segments get cut."""
    total = 0
    for w in re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", text):
        if w.isalpha():
            total += 1 if len(w) <= 9 else 2
        elif w.isdigit():
            total += math.ceil(len(w) / 2)
        else:
            total += 1
    return total


@dataclass
class _Segment:
    name: str
    text: str
    priority: int
    required: bool = False


def _short(text: str, max_words: int) -> tuple[str, bool]:
    """First sentence of ``text``; if longer than ``max_words``, cut at the last comma inside the budget
    (or at the budget if there is none) so the clause never ends mid-phrase."""
    first = re.split(r"(?<=[.;])\s", text.strip(), maxsplit=1)[0].rstrip(".;")
    words = first.split()
    if len(words) > max_words:
        head = " ".join(words[:max_words])
        cut = head.rfind(",")
        clause = head[:cut] if cut >= len(head) // 2 else head
        return clause.rstrip(" ,"), True
    return first, first != text.strip().rstrip(".;")


def describe_material(m: dict[str, Any], with_imperfections: bool = True) -> str:
    refl = ("mirror-like reflections" if m["reflectivity"] >= 0.85 else "strong reflections" if m["reflectivity"] >= 0.6
            else "soft sheen" if m["reflectivity"] >= 0.3 else "dull surface")
    rough = ("razor-sharp highlights" if m["roughness"] <= 0.1 else "crisp highlights" if m["roughness"] <= 0.3
             else "diffused highlights" if m["roughness"] <= 0.6 else "matte finish")
    # Name the object the material belongs to ("brushed steel tool head"), so the engine does not apply the
    # material to whatever subject happens to precede it in the prompt.
    obj = m["material_id"].replace("_", " ").replace("-", " ").strip()
    noun = "" if obj in _GENERIC_MATERIAL_IDS or re.fullmatch(r"m?\d+", obj) else f" {obj}"
    parts = [f"{MATERIAL_WORDS[m['type']]}{noun} ({m['finish']})", CURVATURE_WORDS[m["curvature"]], refl, rough]
    if m["anisotropy"] >= 0.5:
        parts.append("directional brushed grain")
    if with_imperfections:
        parts.extend(m["imperfections"][:2])
    return ", ".join(parts)


def _temperature_words(kelvin: int) -> str:
    if kelvin < 3500:
        return "warm tungsten light"
    if kelvin < 5000:
        return "warm-neutral light"
    if kelvin < 6500:
        return "neutral daylight"
    return "cool light"


def build_segments(ad: dict[str, Any]) -> tuple[list[_Segment], list[dict[str, str]]]:
    warnings: list[dict[str, str]] = []
    subject, shortened = _short(ad["subject"]["description"], 16)
    if shortened:
        warnings.append({"code": "subject_shortened", "message": "subject description reduced to its first clause for the prompt"})
    segs = [_Segment("subject", f"fine-art editorial photograph, {subject}", 0, required=True)]
    for i, m in enumerate(ad["materials"]):
        segs.append(_Segment(f"materials/{i}", describe_material(m, with_imperfections=(i != 0)), 1 if i == 0 else 7,
                             required=(i == 0)))
        if i == 0 and m["imperfections"]:
            # Surface detail ranks below lighting and palette: it refines the material, the light defines the image.
            segs.append(_Segment("materials/0/imperfections", ", ".join(m["imperfections"][:2]), 4))
    lt = ad["lighting"]
    light = [f"{lt['key']['hardness']} key light {lt['key']['direction']}", _temperature_words(lt["key"]["color_temperature_k"])]
    light.append("no fill, deep shadows" if lt["fill"] == "none" else f"{lt['fill']} fill")
    if lt["rim"]:
        light.append("rim light")
    light.append(f"{lt['contrast_ratio']} contrast")
    segs.append(_Segment("lighting", ", ".join(light), 2))
    musts = [c["constraint"] for c in ad["semantic_constraints"] if c["priority"] == "must"]
    if musts:
        segs.append(_Segment("semantic_constraints/must", ", ".join(musts), 5))
    comp = ad["composition"]
    comp_parts = [FRAMING_WORDS[comp["framing"]], comp["camera_angle"], comp["layout"], f"{comp['negative_space']} negative space"]
    if comp["lens_mm"]:
        comp_parts.insert(2, f"{comp['lens_mm']:g}mm lens")
    segs.append(_Segment("composition", ", ".join(comp_parts), 4))
    pal = ad["palette"]
    sat = {"monochrome": "monochrome", "muted": "muted tones", "moderate": "restrained colour", "vivid": "vivid colour"}[pal["saturation"]]
    segs.append(_Segment("palette", f"{' and '.join(c['name'].lower() for c in pal['colors'])} palette, {sat}", 3))
    segs.append(_Segment("mood", ", ".join(ad["mood"]), 6))
    shoulds = [c["constraint"] for c in ad["semantic_constraints"] if c["priority"] == "should"]
    if shoulds:
        segs.append(_Segment("semantic_constraints/should", ", ".join(shoulds), 8))
    segs.append(_Segment("lighting/environment", lt["environment"], 9))
    return segs, warnings


def _fit(segments: list[_Segment], budget: int | None) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    warnings: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    ordered = sorted(segments, key=lambda s: s.priority)
    chosen: list[_Segment] = []
    used = 0
    for seg in ordered:
        cost = estimate_tokens(seg.text) + (1 if chosen else 0)
        if budget is None or seg.required or used + cost <= budget:
            chosen.append(seg)
            used += cost
            continue
        # Keep the leading comma-separated parts that fit (the most important words come first), else drop.
        parts = [p.strip() for p in seg.text.split(",")]
        while len(parts) > 1 and used + estimate_tokens(", ".join(parts)) + 1 > budget:
            parts.pop()
        kept = ", ".join(parts)
        if len(parts) < len(seg.text.split(",")) and used + estimate_tokens(kept) + 1 <= budget and estimate_tokens(kept) >= 3:
            chosen.append(_Segment(seg.name, kept, seg.priority))
            used += estimate_tokens(kept) + 1
            dropped.append({"field": f"{seg.name} (tail)", "reason": f"trimmed to fit token budget {budget}"})
        else:
            dropped.append({"field": seg.name, "reason": f"token budget {budget} exceeded (estimate {used}+{cost})"})
    text = ", ".join(s.text for s in chosen)
    if budget is not None and estimate_tokens(text) > budget:
        words = text.split()
        while words and estimate_tokens(" ".join(words)) > budget:
            words.pop()
        text = " ".join(words).rstrip(",")
        warnings.append({"code": "prompt_truncated", "message": f"required segments exceeded {budget} tokens; truncated"})
    if dropped:
        warnings.append({"code": "prompt_segments_dropped",
                         "message": "lower-priority segments dropped: " + ", ".join(d["field"] for d in dropped)})
    return text, warnings, dropped


def negotiate_size(width: int, height: int, caps: EngineCapabilities) -> tuple[int, int, bool]:
    w, h = float(width), float(height)
    scale = min(1.0, caps.max_side / max(w, h), math.sqrt(caps.max_pixels / (w * h)))
    w, h = w * scale, h * scale
    if min(w, h) < caps.min_side:
        up = caps.min_side / min(w, h)
        w, h = w * up, h * up
    m = caps.size_multiple_of
    nw, nh = max(m, int(round(w / m)) * m), max(m, int(round(h / m)) * m)
    if max(nw, nh) > caps.max_side or nw * nh > caps.max_pixels:
        nw, nh = max(m, int(w // m) * m), max(m, int(h // m) * m)  # rounding up overshot a limit: round down
    if max(nw, nh) > caps.max_side or nw * nh > caps.max_pixels or min(nw, nh) < caps.min_side:
        raise CompilationError(f"cannot fit {width}x{height} within engine size limits")
    return nw, nh, (nw, nh) != (width, height)


def resolve_params(caps: EngineCapabilities, overrides: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(overrides) - set(caps.params))
    if unknown:
        raise CompilationError(f"engine {caps.engine_id}/{caps.model_id} does not accept params {unknown}")
    params = {k: spec.default for k, spec in caps.params.items()}
    params.update(overrides)
    for k, v in params.items():
        caps.params[k].check(k, v)
    return params


def seed_for(base_seed: int, slot: int, iteration: int, keep_seeds: bool) -> int:
    seed = base_seed + slot * 7919 + (0 if keep_seeds else iteration * 104729)
    return seed % (2**32)


def negotiate_references(refs: list[dict[str, Any]], caps: EngineCapabilities, base_dir: Path | None
                         ) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    warnings, dropped, kept = [], [], []
    required = [r for r in refs if r["required"]]
    if len(required) > caps.max_reference_images:
        raise CompilationError(f"{len(required)} required reference image(s) but engine "
                               f"{caps.engine_id}/{caps.model_id} accepts {caps.max_reference_images}")
    budget = caps.max_reference_images - len(required)
    for r in refs:
        if r["required"] or budget > 0:
            if not r["required"]:
                budget -= 1
            p = Path(r["path"]) if base_dir is None else (base_dir / r["path"])
            kept.append({"path": r["path"], "role": r["role"], "sha256": sha256_file(p) if p.is_file() else None})
        else:
            dropped.append({"field": f"reference_images[{r['path']}]", "reason": "engine does not accept reference images"})
    if dropped:
        warnings.append({"code": "reference_images_dropped",
                         "message": f"{len(dropped)} optional reference image(s) dropped; they will only inform critique"})
    return kept, warnings, dropped


def _ids(run_id: str, iteration: int, slot: int) -> dict[str, str]:
    stem = f"{run_id.split('_', 1)[1]}-i{iteration}-s{slot}"
    return {"candidate_id": f"cand_{stem}", "request_id": f"req_{stem}"}


def compile_request(
    ad: dict[str, Any],
    brief: dict[str, Any],
    caps: EngineCapabilities,
    *,
    run_id: str,
    iteration: int,
    slot: int,
    base_seed: int,
    keep_seeds: bool,
    engine_params: dict[str, Any],
    timeout_s: float,
    max_retries: int,
    reference_base_dir: Path | None = None,
) -> dict[str, Any]:
    params = resolve_params(caps, engine_params)
    segments, warnings = build_segments(ad)
    positive, w2, dropped = _fit(segments, caps.max_prompt_tokens)
    warnings += w2

    neg_terms: list[str] = []
    for t in [a["term"] for a in ad["avoidance_constraints"]] + list(brief["constraints"]["must_avoid"]) + list(DEFAULT_NEGATIVE_TERMS):
        if t.lower() not in (x.lower() for x in neg_terms):
            neg_terms.append(t)
    negative: str | None = ", ".join(neg_terms)
    if not caps.supports_negative_prompt:
        negative = None
        dropped.append({"field": "prompt.negative", "reason": "engine has no negative prompt"})
    elif not caps.negative_prompt_effective(params):
        negative = None
        dropped.append({"field": "prompt.negative",
                        "reason": f"negative prompts have no effect at guidance_scale={params.get('guidance_scale')}"})
    if negative is None:
        warnings.append({"code": "negative_prompt_unavailable",
                         "message": "avoidance constraints are enforced only by critique, not by the engine"})
    elif caps.max_prompt_tokens and estimate_tokens(negative) > caps.max_prompt_tokens:
        while neg_terms and estimate_tokens(", ".join(neg_terms)) > caps.max_prompt_tokens:
            dropped.append({"field": f"prompt.negative[{neg_terms[-1]}]", "reason": "negative prompt token budget"})
            neg_terms.pop()
        negative = ", ".join(neg_terms)

    width, height, adjusted = negotiate_size(ad["output"]["width"], ad["output"]["height"], caps)
    if adjusted:
        warnings.append({"code": "size_adjusted",
                         "message": f"{ad['output']['width']}x{ad['output']['height']} -> {width}x{height} for engine limits"})

    refs, w3, d3 = negotiate_references(brief["constraints"]["reference_images"], caps, reference_base_dir)
    warnings += w3
    dropped += d3

    seed = seed_for(base_seed, slot, iteration, keep_seeds) if caps.supports_seed else None
    if seed is None:
        warnings.append({"code": "seed_unsupported", "message": "engine does not accept seeds; results are not reproducible"})

    ids = _ids(run_id, iteration, slot)
    return {
        "schema_version": "atelier.generation_request.v1",
        "request_id": ids["request_id"],
        "run_id": run_id,
        "candidate_id": ids["candidate_id"],
        "iteration": iteration,
        "slot": slot,
        "art_direction_id": ad["art_direction_id"],
        "engine": {"engine_id": caps.engine_id, "model_id": caps.model_id},
        "prompt_source": "compiled",
        "prompt": {"positive": positive, "negative": negative},
        "reference_images": refs,
        "width": width,
        "height": height,
        "seed": seed,
        "engine_params": params,
        "limits": {"timeout_s": timeout_s, "max_retries": max_retries},
        "compilation": {
            "compiler_version": COMPILER_VERSION,
            "token_estimate": estimate_tokens(positive),
            "token_budget": caps.max_prompt_tokens,
            "warnings": warnings,
            "dropped": dropped,
            "size_adjusted": adjusted,
        },
    }


def compile_baseline_request(
    text: str,
    caps: EngineCapabilities,
    *,
    run_id: str,
    slot: int,
    width: int,
    height: int,
    base_seed: int,
    engine_params: dict[str, Any],
    timeout_s: float,
    max_retries: int,
    negative: str | None = None,
) -> dict[str, Any]:
    """The text as the only prompt (experiment condition A, and direct-prompt tools such as the studio).
    ``negative`` is used only where the engine actually honours negative prompts."""
    params = resolve_params(caps, engine_params)
    positive, warnings, dropped = _fit([_Segment("original_intent", " ".join(text.split()), 0, required=True)],
                                       caps.max_prompt_tokens)
    if negative and not caps.negative_prompt_effective(params):
        dropped.append({"field": "prompt.negative",
                        "reason": f"negative prompts have no effect on {caps.model_id} at "
                                  f"guidance_scale={params.get('guidance_scale')}"})
        warnings.append({"code": "negative_prompt_unavailable",
                         "message": "this model ignores negative prompts (--no); use the high-quality model"})
        negative = None
    w, h, adjusted = negotiate_size(width, height, caps)
    ids = _ids(run_id, 0, slot)
    return {
        "schema_version": "atelier.generation_request.v1",
        "request_id": ids["request_id"],
        "run_id": run_id,
        "candidate_id": ids["candidate_id"],
        "iteration": 0,
        "slot": slot,
        "art_direction_id": None,
        "engine": {"engine_id": caps.engine_id, "model_id": caps.model_id},
        "prompt_source": "baseline_raw",
        "prompt": {"positive": positive, "negative": " ".join(negative.split()) if negative else None},
        "reference_images": [],
        "width": w,
        "height": h,
        "seed": seed_for(base_seed, slot, 0, True) if caps.supports_seed else None,
        "engine_params": params,
        "limits": {"timeout_s": timeout_s, "max_retries": max_retries},
        "compilation": {
            "compiler_version": COMPILER_VERSION,
            "token_estimate": estimate_tokens(positive),
            "token_budget": caps.max_prompt_tokens,
            "warnings": warnings,
            "dropped": dropped,
            "size_adjusted": adjusted,
        },
    }


def engine_input(request: dict[str, Any]) -> dict[str, Any]:
    """The part of a request that actually reaches the engine (used to detect no-op changes)."""
    return {k: request[k] for k in ("prompt", "width", "height", "engine_params", "reference_images")}
