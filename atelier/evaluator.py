"""Deterministic technical QA, weighted scoring, and improvement-criterion evaluation.

Aesthetic judgement does not happen here. Critiques come from the art-critic agent (or,
offline, from clearly labelled heuristic proxies). This module only measures and aggregates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from atelier.config import DIMENSIONS, check_weights
from atelier.util import sha256_file


def image_stats(path: Path) -> dict[str, float]:
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    lap = 4 * luma[1:-1, 1:-1] - luma[:-2, 1:-1] - luma[2:, 1:-1] - luma[1:-1, :-2] - luma[1:-1, 2:]
    return {
        "mean_luma": float(luma.mean()),
        "std_luma": float(luma.std()),
        "clipped_dark_fraction": float((luma <= 0.02).mean()),
        "clipped_bright_fraction": float((luma >= 0.98).mean()),
        "sharpness": float(lap.var()) if lap.size else 0.0,
        "saturation_mean": float(sat.mean()),
        "specular_fraction": float((luma >= 0.9).mean()),
    }


def run_qa(result: dict[str, Any], request: dict[str, Any], store, qa_cfg: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, value: Any, threshold: Any, message: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "value": value, "threshold": threshold, "message": message})

    metrics: dict[str, float] = {}
    img = result["images"][0] if result["images"] else None
    check("image_present", img is not None, len(result["images"]), 1, "result lists at least one image")
    if img is not None:
        path = store.abs(img["path"])
        exists = path.is_file()
        check("file_exists", exists, exists, True, img["path"])
        if exists:
            check("sha256_matches", sha256_file(path) == img["sha256"], None, None, "file hash equals recorded hash")
            try:
                metrics = image_stats(path)
                decoded = True
            except Exception as exc:  # noqa: BLE001
                decoded = False
                check("decodes", False, str(exc)[:200], None, "image decodes")
            if decoded:
                check("decodes", True, True, True, "image decodes")
                dims_ok = (img["width"], img["height"]) == (request["width"], request["height"])
                check("dimensions_match", dims_ok, f"{img['width']}x{img['height']}",
                      f"{request['width']}x{request['height']}", "output size equals requested size")
                check("not_blank", metrics["std_luma"] >= qa_cfg["min_luma_std"], round(metrics["std_luma"], 5),
                      qa_cfg["min_luma_std"], "luma standard deviation (detects flat/blank frames)")
                clipped = max(metrics["clipped_dark_fraction"], metrics["clipped_bright_fraction"])
                check("exposure_clipping", clipped <= qa_cfg["max_clipped_fraction"], round(clipped, 5),
                      qa_cfg["max_clipped_fraction"], "fraction of pixels crushed to black or blown to white")
                check("sharpness", metrics["sharpness"] >= qa_cfg["min_sharpness"], round(metrics["sharpness"], 7),
                      qa_cfg["min_sharpness"], "Laplacian variance (detects fully blurred frames)")
    return {
        "schema_version": "atelier.qa_report.v1",
        "qa_id": "qa_" + result["candidate_id"].split("_", 1)[1],
        "candidate_id": result["candidate_id"],
        "result_id": result["result_id"],
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
    }


def weighted_score(ratings: dict[str, int], weights: dict[str, float]) -> float:
    check_weights(weights)
    return round(sum(weights[d] * ratings[d] for d in DIMENSIONS), 4)


def aggregate(values: Iterable[float], how: str) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return round(max(vals) if how == "best" else sum(vals) / len(vals), 4)


def metric_values(critiques: list[dict[str, Any]], metric: str, weights: dict[str, float]) -> list[float]:
    if metric == "weighted_score":
        return [weighted_score(c["ratings"], weights) for c in critiques]
    return [float(c["ratings"][metric]) for c in critiques]


def evaluate_criterion(plan: dict[str, Any], baseline_critiques: list[dict[str, Any]],
                       new_critiques: list[dict[str, Any]], weights: dict[str, float]) -> dict[str, Any]:
    """Did the refinement batch meet the plan's own pre-registered improvement criterion?"""
    crit = plan["improvement_criterion"]
    baseline = aggregate(metric_values(baseline_critiques, crit["metric"], weights), crit["aggregate"])
    measured = aggregate(metric_values(new_critiques, crit["metric"], weights), crit["aggregate"])
    if measured is None:
        met = False
    elif crit["comparator"] == ">=":
        met = measured >= crit["threshold"]
    else:
        met = measured > crit["threshold"]
    return {
        "plan_id": plan["plan_id"],
        "metric": crit["metric"],
        "aggregate": crit["aggregate"],
        "comparator": crit["comparator"],
        "threshold": crit["threshold"],
        "baseline": baseline,
        "measured": measured,
        "met": bool(met),
    }
