"""Bounded refinement: plan validation, plan application, and stopping conditions."""

from __future__ import annotations

import copy
from typing import Any, Callable

from atelier import schemas, spec_checks

# Sections an iteration may change ("major variables"). Changing several paths inside one
# section counts as ONE major variable.
MUTABLE_SECTIONS = frozenset({"subject", "composition", "materials", "lighting", "palette", "mood",
                              "semantic_constraints", "avoidance_constraints", "visual_hypothesis"})
# The concept itself and the output contract stay fixed within a run.
FROZEN_SECTIONS = frozenset({"schema_version", "art_direction_id", "version", "parent_id", "brief_id", "concept_id",
                             "provenance", "concept_thesis", "output"})

_MISSING = object()


def _unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def pointer_get(doc: Any, pointer: str) -> Any:
    cur = doc
    for raw in pointer.lstrip("/").split("/"):
        tok = _unescape(raw)
        if isinstance(cur, list):
            if not tok.isdigit() or int(tok) >= len(cur):
                return _MISSING
            cur = cur[int(tok)]
        elif isinstance(cur, dict):
            if tok not in cur:
                return _MISSING
            cur = cur[tok]
        else:
            return _MISSING
    return cur


def pointer_set(doc: Any, pointer: str, value: Any) -> None:
    parts = [_unescape(p) for p in pointer.lstrip("/").split("/")]
    cur = doc
    for tok in parts[:-1]:
        cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def major_sections(plan: dict[str, Any]) -> list[str]:
    return sorted({c["path"].lstrip("/").split("/")[0] for c in plan["what_will_change"]})


def validate_plan(plan: dict[str, Any], current_ad: dict[str, Any], *, current_iteration: int,
                  selection_id: str, max_major_variables: int, max_iterations: int,
                  brief: dict[str, Any] | None = None,
                  engine_input_fn: Callable[[dict[str, Any]], Any] | None = None) -> list[str]:
    """``engine_input_fn(art_direction)`` compiles a representative request and returns what
    the engine would receive. A plan whose change does not alter that input is rejected:
    it would spend a generation batch on a variable the compiler drops for this engine."""
    errors = schemas.validation_errors(plan, "atelier.iteration_plan.v1")
    if errors:
        return errors
    if plan["iteration"] != current_iteration + 1:
        errors.append(f"plan targets iteration {plan['iteration']}, expected {current_iteration + 1}")
    if plan["iteration"] > max_iterations:
        errors.append(f"plan targets iteration {plan['iteration']} beyond max_iterations={max_iterations}")
    if plan["base_art_direction_id"] != current_ad["art_direction_id"]:
        errors.append(f"plan is based on {plan['base_art_direction_id']}, current is {current_ad['art_direction_id']}")
    if plan["based_on_selection_id"] != selection_id:
        errors.append(f"plan cites selection {plan['based_on_selection_id']}, current is {selection_id}")
    crit = plan["improvement_criterion"]
    if crit["comparator"] == ">" and crit["threshold"] >= 5:
        errors.append("improvement criterion '> 5' can never be met on a 1-5 scale")
    sections = major_sections(plan)
    for s in sections:
        if s in FROZEN_SECTIONS or s not in MUTABLE_SECTIONS:
            errors.append(f"section {s!r} may not be changed by an iteration")
    if len(sections) > max_major_variables:
        errors.append(f"plan changes {len(sections)} major variables {sections}; limit is {max_major_variables}")
    paths = [c["path"] for c in plan["what_will_change"]]
    if len(paths) != len(set(paths)):
        errors.append("duplicate paths in what_will_change")
    for c in plan["what_will_change"]:
        current = pointer_get(current_ad, c["path"])
        if current is _MISSING:
            errors.append(f"{c['path']} does not exist in the current art direction")
        elif current != c["from"]:
            errors.append(f"{c['path']}: 'from' does not match the current value (stale plan)")
        elif c["from"] == c["to"]:
            errors.append(f"{c['path']}: 'to' equals 'from' (no change)")
    if not errors:
        try:
            candidate = apply_plan(plan, current_ad)
        except schemas.SchemaValidationError as exc:
            errors.extend(f"resulting art direction invalid: {e}" for e in exc.errors)
        else:
            sem_errors, _ = spec_checks.check_art_direction(candidate, brief)
            errors.extend(f"resulting art direction: {e}" for e in sem_errors)
            if not sem_errors and engine_input_fn is not None and engine_input_fn(candidate) == engine_input_fn(current_ad):
                errors.append("plan does not change the compiled engine input (prompt, size, params): the changed "
                              f"section(s) {major_sections(plan)} are dropped or unused by the prompt compiler for "
                              "this engine, so the experiment could not affect the images")
    return errors


def apply_plan(plan: dict[str, Any], current_ad: dict[str, Any]) -> dict[str, Any]:
    new = copy.deepcopy(current_ad)
    for c in plan["what_will_change"]:
        pointer_set(new, c["path"], copy.deepcopy(c["to"]))
    stem = current_ad["art_direction_id"].split("_", 1)[1].rsplit("-v", 1)[0]
    new["version"] = current_ad["version"] + 1
    new["art_direction_id"] = f"ad_{stem}-v{new['version']}"
    new["parent_id"] = current_ad["art_direction_id"]
    new["provenance"] = dict(current_ad["provenance"], derived_from_plan_id=plan["plan_id"])
    schemas.validate(new, "atelier.art_direction.v1")
    return new


def stopping_decision(
    *,
    iteration: int,
    max_iterations: int,
    best_overall: float | None,
    best_overall_min_rating: int | None,
    best_before_this_iteration: float | None,
    best_this_iteration: float | None,
    stopping_cfg: dict[str, Any],
    curator_recommendation: str,
    curator_stop_reason: str | None,
    can_afford_next_batch: tuple[bool, str | None],
) -> str | None:
    """Return a stop reason, or None to run another refinement iteration."""
    if best_overall is None:
        return "no_eligible_candidates: no candidate passed QA and critique"
    if (best_overall >= stopping_cfg["target_weighted_score"]
            and (best_overall_min_rating or 0) >= stopping_cfg["min_dimension_for_target"]):
        return f"target_reached: best weighted score {best_overall:.3f} >= {stopping_cfg['target_weighted_score']}"
    if iteration >= max_iterations:
        return f"max_iterations_reached: {iteration} refinement iteration(s) completed (limit {max_iterations})"
    if iteration >= 1 and best_before_this_iteration is not None:
        gain = (best_this_iteration or 0.0) - best_before_this_iteration
        if gain < stopping_cfg["min_improvement"]:
            return (f"plateau: best score changed by {gain:+.3f} (< {stopping_cfg['min_improvement']}) "
                    "after the last refinement")
    ok, why = can_afford_next_batch
    if not ok:
        return f"budget_insufficient_for_next_batch: {why}"
    if curator_recommendation == "stop":
        return f"curator_recommended_stop: {curator_stop_reason}"
    return None
