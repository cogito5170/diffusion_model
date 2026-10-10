"""Deterministic offline stand-ins for the judgement steps.

These are NOT creative or critical judgement. They exist so the whole pipeline can be
tested end to end without network, API keys or GPU. Every artifact they emit is
labelled: producer.kind = "offline_stub", or "heuristic" for the critic. Heuristic
critiques use evidence_type "heuristic_proxy" and low confidence.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from atelier.config import DIMENSIONS, DEFAULT_WEIGHTS
from atelier.evaluator import image_stats
from atelier.iteration import MUTABLE_SECTIONS, pointer_get, pointer_set
from atelier.selector import distance, image_features

STUB = {"name": "offline-stub", "kind": "offline_stub"}

_THEME_LEXICON = {
    "identity": "identity as surface", "mirror": "reflection and self-recognition", "reflect": "reflection and self-recognition",
    "chrome": "industrial perfection", "steel": "industrial endurance", "titanium": "engineered permanence",
    "body": "the body as material", "face": "facelessness", "skin": "skin versus shell", "memory": "material memory",
    "industrial": "industrial space", "machine": "the machinic", "liquid": "instability and flux",
}


def _concept_templates(brief: dict[str, Any]) -> list[dict[str, Any]]:
    stem = brief["brief_id"].split("_", 1)[1]
    return [
        {
            "concept_id": f"concept_{stem}-mirror",
            "title": "The Mirror That Refuses",
            "thesis": "A perfectly reflective form returns the whole industrial room to the viewer but withholds the "
                      "face: identity appears only as the distortion it causes in a surface.",
            "tension": {"pole_a": "self-recognition", "pole_b": "impersonal reflection",
                        "description": "The viewer looks for a self in the metal and finds only the environment."},
            "visual_hypothesis": "A single liquid-chrome sculptural mass, implied human contour, hard key light; "
                                 "the reflection carries more information than the object.",
            "key_motifs": ["liquid chrome mass", "implied human contour", "reflected girders"],
            "risks": ["can read as a generic 3D render", "reflection detail may turn to noise"],
        },
        {
            "concept_id": f"concept_{stem}-brushed",
            "title": "Brushed Memory",
            "thesis": "Brushed steel keeps the direction of every abrasion; identity is the accumulated record of "
                      "contact rather than a likeness.",
            "tension": {"pole_a": "personal history", "pole_b": "industrial process",
                        "description": "Marks of touch are indistinguishable from marks of manufacture."},
            "visual_hypothesis": "A torso-scaled brushed-steel plane with directional grain and one polished seam.",
            "key_motifs": ["directional grain", "polished seam", "raking light"],
            "risks": ["low drama", "grain may alias at small sizes"],
        },
        {
            "concept_id": f"concept_{stem}-molten",
            "title": "Molten Self",
            "thesis": "Identity as surface tension: a body-shaped chrome volume caught between solid and liquid.",
            "tension": {"pole_a": "form", "pole_b": "dissolution",
                        "description": "The figure is legible only while the metal holds; the image is the moment before collapse."},
            "visual_hypothesis": "Fluid chrome with drips and a stable shoulder line, cool light, black ground.",
            "key_motifs": ["drips", "shoulder line", "black ground"],
            "risks": ["melting clichés", "surreal kitsch"],
        },
    ]


def research(req, store) -> dict[str, Any]:
    brief = req.context["brief"]
    text = brief["original_intent"]["text"].lower()
    themes = sorted({v for k, v in _THEME_LEXICON.items() if k in text}) or ["metal as a surface for identity"]
    return {
        "schema_version": "atelier.research_notes.v1",
        **req.required_values,
        "produced_by": STUB,
        "research_mode": "offline_stub",
        "sources": [],
        "context_notes": [
            {"topic": "offline mode", "note": "No research was performed. These notes are fixed placeholders "
                                              "derived from keyword matching on the brief."},
        ],
        "interpretation": {
            "summary": "Offline keyword interpretation of the brief: " + "; ".join(themes) + ".",
            "themes": themes,
            "tensions": [{"pole_a": "human identity", "pole_b": "material surface",
                          "description": "Default tension of the metallic editorial domain."}],
            "open_questions": ["Should the human figure be visible, partial, or only implied?"],
        },
        "visual_vocabulary": ["liquid chrome", "brushed steel", "hard directional light", "industrial space"],
        "cautions": ["Offline stub: no sources consulted; nothing here may be cited as research."],
    }


def concepts(req, store) -> dict[str, Any]:
    brief = req.context["brief"]
    items = _concept_templates(brief)
    return {
        "schema_version": "atelier.concept_set.v1",
        **req.required_values,
        "produced_by": STUB,
        "concepts": items,
        "recommended_concept_id": items[0]["concept_id"],
        "distinctness_rationale": "Concepts differ in material state (mirror, brushed, molten) and in where "
                                  "identity is located (reflection, abrasion, surface tension).",
    }


def _size(aspect: str, long_side: int = 640) -> tuple[int, int]:
    a, b = (int(x) for x in aspect.split(":"))
    if a >= b:
        w, h = long_side, long_side * b / a
    else:
        w, h = long_side * a / b, long_side
    return int(round(w / 8) * 8), int(round(h / 8) * 8)


def art_direction_draft(req, store) -> dict[str, Any]:
    brief, cset = req.context["brief"], req.context["concepts"]
    concept = next(c for c in cset["concepts"] if c["concept_id"] == cset.get("recommended_concept_id", cset["concepts"][0]["concept_id"]))
    w, h = _size(brief["constraints"]["aspect_ratio"])
    avoid = [{"term": t, "reason": "brief must_avoid"} for t in brief["constraints"]["must_avoid"]]
    for term, why in (("cartoon", "domain is fine-art editorial"), ("sci-fi robot", "cliche of metal + human")):
        if term not in [a["term"] for a in avoid]:
            avoid.append({"term": term, "reason": why})
    return {
        "schema_version": "atelier.art_direction_draft.v1",
        **req.required_values,
        "concept_id": concept["concept_id"],
        "produced_by": STUB,
        "concept_thesis": concept["thesis"],
        "visual_hypothesis": concept["visual_hypothesis"],
        "subject": {"description": "a sculptural liquid chrome form with an implied human contour in an empty industrial hall",
                    "human_presence": "implied", "human_treatment": "contour only, no face"},
        "composition": {"framing": "medium", "camera_angle": "eye level", "lens_mm": 85,
                        "layout": "single subject, slightly off-center", "focal_hierarchy": ["reflective form", "reflected room"],
                        "negative_space": "balanced"},
        "palette": {"colors": [{"hex": "#2E3034", "name": "Graphite", "role": "dominant"},
                               {"hex": "#D6DAE0", "name": "Chrome", "role": "highlight"},
                               {"hex": "#0C0C0E", "name": "Black", "role": "shadow"}],
                    "saturation": "muted"},
        "mood": ["austere", "contemplative"],
        "semantic_constraints": [{"constraint": "no visible face", "priority": "must"},
                                 {"constraint": "the reflection shows the room", "priority": "should"}]
                                + [{"constraint": t, "priority": "must"} for t in brief["constraints"]["must_include"]],
        "avoidance_constraints": avoid,
        "output": {"aspect_ratio": brief["constraints"]["aspect_ratio"], "width": w, "height": h},
        "notes_for_material_specialist": "Primary surface should read as liquid chrome under one hard key light.",
    }


def material_spec(req, store) -> dict[str, Any]:
    return {
        "schema_version": "atelier.material_spec.v1",
        **req.required_values,
        "produced_by": STUB,
        "materials": [{"material_id": "primary", "type": "liquid_chrome", "finish": "mirror", "reflectivity": 0.95,
                       "roughness": 0.05, "anisotropy": 0.0, "curvature": "fluid", "imperfections": [],
                       "role": "the figure"}],
        "lighting": {"setup": "single hard key, black flags", "key": {"direction": "from upper left", "hardness": "hard",
                     "color_temperature_k": 5600}, "fill": "none", "rim": False,
                     "environment": "empty industrial hall with steel girders", "contrast_ratio": "8:1"},
        "rationale": "Fixed offline default for the liquid-chrome domain.",
        "plausibility_notes": ["Chrome needs a structured environment to reflect, otherwise it reads as grey plastic."],
    }


# ---------------------------------------------------------------- heuristic critic
def _rating(x: float) -> int:
    return int(min(5, max(1, round(x))))


def _hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255.0


def heuristic_ratings(image_path: Path, ad: dict[str, Any], qa: dict[str, Any],
                      peer_paths: list[Path]) -> tuple[dict[str, int], list[dict[str, str]]]:
    s = image_stats(image_path)
    obs: list[dict[str, str]] = []

    clipped = max(s["clipped_dark_fraction"], s["clipped_bright_fraction"])
    tech = 5 - (1 if clipped > 0.05 else 0) - (1 if clipped > 0.2 else 0) - (1 if s["sharpness"] < 5e-4 else 0)
    obs.append({"dimension": "technical_quality", "evidence_type": "technical_measurement",
                "observation": f"clipping {clipped:.3f}, Laplacian variance {s['sharpness']:.5f}, QA passed={qa['passed']}."})

    sf = s["specular_fraction"]
    mat = 1 + (2 if 0.003 <= sf <= 0.1 else 0.5 if sf > 0 else 0) + (1.5 if s["saturation_mean"] < 0.15 else 0.5 if s["saturation_mean"] < 0.3 else 0) \
        + (0.5 if s["std_luma"] > 0.15 else 0)
    obs.append({"dimension": "material_plausibility", "evidence_type": "heuristic_proxy",
                "observation": f"specular-highlight fraction {sf:.4f}, mean saturation {s['saturation_mean']:.3f}; "
                               "proxy for 'reads as metal', not a judgement of realism."})

    with Image.open(image_path) as im:
        luma = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    obj = luma > (luma.mean() + 0.05)
    occ = float(obj.mean())
    if obj.any():
        ys, xs = np.nonzero(obj)
        cx, cy = xs.mean() / luma.shape[1], ys.mean() / luma.shape[0]
    else:
        cx = cy = 0.5
    target = {"generous": 0.18, "balanced": 0.32, "minimal": 0.5}[ad["composition"]["negative_space"]]
    thirds = min(abs(cx - t) for t in (1 / 3, 0.5, 2 / 3)) + min(abs(cy - t) for t in (1 / 3, 0.5, 2 / 3))
    comp = 1 + 2 * max(0.0, 1 - abs(occ - target) / target) + 2 * max(0.0, 1 - thirds / 0.3)
    obs.append({"dimension": "composition", "evidence_type": "heuristic_proxy",
                "observation": f"subject occupancy {occ:.2f} vs target {target:.2f} for '{ad['composition']['negative_space']}' "
                               f"negative space; centroid ({cx:.2f}, {cy:.2f})."})

    with Image.open(image_path) as im:
        hsv = np.asarray(im.convert("RGB").resize((96, 96)).convert("HSV"), dtype=np.float32) / 255.0
    sel = hsv[..., 1] > 0.15
    if sel.sum() < 20:
        coh, spread = 5.0, 0.0
    else:
        ang = hsv[..., 0][sel] * 2 * math.pi
        R = math.hypot(np.cos(ang).mean(), np.sin(ang).mean())
        spread = math.sqrt(max(0.0, -2 * math.log(max(R, 1e-6))))
        coh = 5 - 4 * min(1.0, spread / 1.5)
    obs.append({"dimension": "visual_coherence", "evidence_type": "heuristic_proxy",
                "observation": f"circular hue spread {spread:.2f} rad over saturated pixels."})

    if peer_paths:
        f = image_features(image_path)
        d = sum(distance(f, image_features(p)) for p in peer_paths) / len(peer_paths)
        orig = 1 + 4 * min(1.0, d / 0.5)
        obs.append({"dimension": "originality", "evidence_type": "batch_comparison",
                    "observation": f"mean feature distance to {len(peer_paths)} batch peers {d:.3f}; "
                                   "measures within-batch novelty only, not originality in the world."})
    else:
        orig = 3.0
        obs.append({"dimension": "originality", "evidence_type": "heuristic_proxy",
                    "observation": "no peers to compare; neutral rating."})

    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    cols = sorted(ad["palette"]["colors"], key=lambda c: _hex_rgb(c["hex"]).sum())
    dark_ref, light_ref = _hex_rgb(cols[0]["hex"]), _hex_rgb(cols[-1]["hex"])
    dark_px, light_px = rgb[luma < 0.3], rgb[luma > 0.7]
    dists = []
    if len(dark_px):
        dists.append(float(np.linalg.norm(dark_px.mean(0) - dark_ref)))
    if len(light_px):
        dists.append(float(np.linalg.norm(light_px.mean(0) - light_ref)))
    pal_dist = sum(dists) / len(dists) if dists else 1.0
    intent = 5 - 6 * pal_dist
    if ad["palette"]["saturation"] == "monochrome" and s["saturation_mean"] > 0.08:
        intent -= 1
    obs.append({"dimension": "intent_alignment", "evidence_type": "spec_comparison",
                "observation": f"palette distance {pal_dist:.3f} between image shadows/highlights and the specified "
                               "darkest/lightest colours. Says nothing about whether the concept reads."})

    ratings = {"intent_alignment": _rating(intent), "material_plausibility": _rating(mat), "composition": _rating(comp),
               "visual_coherence": _rating(coh), "originality": _rating(orig), "technical_quality": _rating(tech)}
    return ratings, obs


def critique(req, store) -> dict[str, Any]:
    ctx = req.context
    ratings, obs = heuristic_ratings(ctx["image_path"], ctx["art_direction"], ctx["qa"], ctx["peer_paths"])
    weakest = min(DIMENSIONS, key=lambda d: (ratings[d], d))
    return {
        "schema_version": "atelier.critique.v1",
        **req.required_values,
        "critic": {"name": "heuristic-critic-v1", "kind": "heuristic"},
        "ratings": ratings,
        "observations": obs,
        "evidence_types": sorted({o["evidence_type"] for o in obs}),
        "confidence": 0.15,
        "next_experiment": {
            "hypothesis": f"{weakest} is the weakest proxy rating ({ratings[weakest]}).",
            "variable": weakest,
            "proposed_change": f"Change the single art-direction section most tied to {weakest}.",
            "expected_effect": f"Higher {weakest} proxy rating with other ratings stable.",
        },
    }


# ---------------------------------------------------------------- rule-based curator
_CYCLE = {"balanced": "generous", "generous": "minimal", "minimal": "balanced"}


def _change_for(dimension: str, ad: dict[str, Any]) -> tuple[str, Any, str]:
    if dimension == "material_plausibility":
        cur = ad["materials"][0]["imperfections"]
        if "fine micro-scratches" not in cur:
            return "/materials/0/imperfections", cur + ["fine micro-scratches"], \
                "Perfect surfaces read as CG; sparse micro-scratches are a physical cue of real metal."
        return "/materials/0/roughness", round(min(0.3, ad["materials"][0]["roughness"] + 0.05), 2), \
            "Slightly rougher highlights reduce the plastic look."
    if dimension == "technical_quality":
        cur = ad["lighting"]["contrast_ratio"]
        return "/lighting/contrast_ratio", "4:1" if cur != "4:1" else "3:1", \
            "Lower contrast reduces crushed shadows and blown highlights."
    if dimension == "composition":
        cur = ad["composition"]["negative_space"]
        return "/composition/negative_space", _CYCLE[cur], "Change subject occupancy to test the framing."
    if dimension == "visual_coherence":
        cur = ad["palette"]["saturation"]
        return "/palette/saturation", "monochrome" if cur != "monochrome" else "muted", \
            "A narrower palette should unify the frame."
    if dimension == "originality":
        cur = ad["composition"]["camera_angle"]
        new = "low angle, looking up at the form" if "low angle" not in cur else "overhead, looking straight down"
        return "/composition/camera_angle", new, "An unusual viewpoint is the cheapest single change toward novelty."
    cur = ad["mood"]
    new = cur[:3] + ["quiet unease"] if "quiet unease" not in cur else ["austere", "uncanny"]
    return "/mood", new, "Make the conceptual tension explicit in the mood descriptors."


def curation(req, store) -> dict[str, Any]:
    ctx = req.context
    sel, ad, it = ctx["selection"], ctx["art_direction"], ctx["iteration"]
    batch = ctx["batch_critiques"]
    notes = [{"candidate_id": r["candidate_id"],
              "note": f"score rank {r['score_rank']}, weighted {r['weighted_score']}" if r["eligible"]
              else f"ineligible: {r['ineligible_reason']}"} for r in sel["ranking"]]
    base = {"schema_version": "atelier.curation.v1", **req.required_values,
            "produced_by": {"name": "rule-based-curator-v1", "kind": "offline_stub"},
            "rationale": f"Shortlist {sel['shortlist']} chosen by weighted score with MMR diversity; "
                         "ratings are heuristic proxies (offline mode).",
            "candidate_notes": notes}
    if it >= ctx["max_iterations"] or not batch:
        return {**base, "recommendation": "stop", "iteration_plan": None,
                "stop_reason": "iteration budget exhausted" if batch else "no critiqued candidates in this batch"}
    means = {d: sum(c["ratings"][d] for c in batch) / len(batch) for d in DIMENSIONS}
    effective = ctx.get("changes_engine_input")
    weakest = path = None
    # Largest weighted deficit first; skip changes that would not reach the engine.
    for dim in sorted(DIMENSIONS, key=lambda d: (-DEFAULT_WEIGHTS[d] * (5 - means[d]), d)):
        p_, v_, w_ = _change_for(dim, ad)
        trial = copy.deepcopy(ad)
        pointer_set(trial, p_, v_)
        if effective is None or effective(trial):
            weakest, path, new_value, why = dim, p_, v_, w_
            break
    if path is None:
        return {**base, "recommendation": "stop", "iteration_plan": None,
                "stop_reason": "no single-section change in the rule table reaches the engine input"}
    if means[weakest] >= 5.0:
        return {**base, "recommendation": "stop", "iteration_plan": None,
                "stop_reason": f"no remaining deficit in dimensions the engine input can change ({weakest} mean is 5.0)"}
    section = path.split("/")[1]
    plan = {
        "schema_version": "atelier.iteration_plan.v1",
        "plan_id": f"plan_{ctx['run_stem']}-i{it + 1}",
        "iteration": it + 1,
        "based_on_selection_id": sel["selection_id"],
        "base_art_direction_id": ad["art_direction_id"],
        "what_failed": [{"dimension": weakest,
                         "evidence": f"batch mean {means[weakest]:.2f}/5 (largest weighted deficit)"}],
        "what_will_change": [{"path": path, "from": pointer_get(ad, path), "to": new_value, "rationale": why}],
        "what_remains_fixed": sorted((MUTABLE_SECTIONS - {section}) | {"concept_thesis", "output size", "seeds per slot",
                                                                        "engine and model", "engine params"}),
        "why_it_should_help": why,
        "improvement_criterion": {"metric": weakest, "aggregate": "mean", "comparator": ">",
                                  "threshold": round(means[weakest], 4)},
        "keep_seeds": True,
    }
    return {**base, "recommendation": "iterate", "stop_reason": None, "iteration_plan": plan}
