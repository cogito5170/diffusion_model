"""Executes one generation request against an engine with budget checks, a timeout,
bounded retries, and strict output validation. It always returns a schema-valid
generation_result describing what actually happened."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from atelier import schemas
from atelier.artifact_store import ArtifactStore
from atelier.budget import Budget, BudgetExceeded, Cancelled, CancelToken
from atelier.engines.base import EngineError, EngineOutput, ImageEngine
from atelier.util import sha256_file, utc_now


def _call_with_timeout(fn, timeout_s: float, *args) -> tuple[str, Any]:
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001 - transported to caller
            box["error"] = exc

    th = threading.Thread(target=target, name="engine-call", daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        return "timeout", None
    if "error" in box:
        exc = box["error"]
        if not isinstance(exc, Exception):  # KeyboardInterrupt / SystemExit / simulated crash
            raise exc
        return "error", exc
    return "ok", box.get("value")


def _check_output(output: Any, out_dir: Path, store: ArtifactStore) -> list[dict[str, Any]]:
    """Return image records or raise ValueError describing why the response is malformed."""
    if not isinstance(output, EngineOutput):
        raise ValueError(f"engine returned {type(output).__name__}, expected EngineOutput")
    if not isinstance(output.model_id, str) or not output.model_id:
        raise ValueError("engine output has no model_id")
    if output.seed_used is not None and (not isinstance(output.seed_used, int) or output.seed_used < 0):
        raise ValueError(f"invalid seed_used {output.seed_used!r}")
    if not isinstance(output.metadata, dict):
        raise ValueError("engine metadata must be a dict")
    try:
        json.dumps(output.metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"engine metadata is not JSON-serializable: {exc}") from None
    if not isinstance(output.images, list) or not output.images:
        raise ValueError("engine returned no images")
    records = []
    root = out_dir.resolve()
    for raw in output.images:
        p = Path(raw).resolve()
        if root not in p.parents:
            raise ValueError(f"image path {raw} is outside the attempt directory")
        if not p.is_file():
            raise ValueError(f"image path {raw} does not exist")
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                fmt, (w, h) = im.format, im.size
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"image {p.name} cannot be decoded: {exc}") from None
        if fmt not in ("PNG", "JPEG", "WEBP"):
            raise ValueError(f"unsupported image format {fmt}")
        records.append({"path": store.rel(p), "sha256": sha256_file(p), "width": w, "height": h, "format": fmt})
    return records


def execute_request(
    engine: ImageEngine,
    request: dict[str, Any],
    store: ArtifactStore,
    budget: Budget,
    cancel: CancelToken,
    *,
    retry_on_timeout: bool = False,
) -> dict[str, Any]:
    schemas.validate(request, "atelier.generation_request.v1")
    caps = engine.capabilities()
    started_at, t0 = utc_now(), time.monotonic()
    attempts: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    status: str | None = None
    seed_used: int | None = None
    model_id = request["engine"]["model_id"]
    metadata: dict[str, Any] = {}
    cost = 0.0

    avail = engine.availability()
    if not avail.ok:
        status = "unavailable"
        errors.append({"code": "engine_unavailable", "message": avail.reason, "retryable": False, "attempt": None})
    else:
        max_attempts = 1 + request["limits"]["max_retries"]
        for attempt in range(1, max_attempts + 1):
            if cancel.is_cancelled():
                status = "cancelled"
                errors.append({"code": "cancelled", "message": "cancelled before attempt", "retryable": False, "attempt": attempt})
                break
            try:
                budget.check_attempt(caps.cost_usd_per_image)
            except BudgetExceeded as exc:
                status = "budget_exceeded"
                errors.append({"code": "budget_exceeded", "message": str(exc), "retryable": False,
                               "attempt": attempt, "details": {"limit": exc.reason}})
                break
            out_dir = store.candidate_image_dir(request["candidate_id"], attempt)
            token = cancel.child()
            a0 = time.monotonic()
            outcome, payload = _call_with_timeout(engine.generate, request["limits"]["timeout_s"], request, out_dir, token)
            runtime = round(time.monotonic() - a0, 4)
            attempt_status, retryable, n_images = "failed", False, 0
            if outcome == "timeout":
                token.cancel()  # cooperative engines stop; non-cooperative threads are abandoned (daemon)
                attempt_status, retryable = "timeout", retry_on_timeout
                errors.append({"code": "timeout", "message": f"no response within {request['limits']['timeout_s']} s",
                               "retryable": retryable, "attempt": attempt})
            elif outcome == "error":
                if isinstance(payload, Cancelled):
                    attempt_status = "cancelled"
                    errors.append({"code": "cancelled", "message": str(payload), "retryable": False, "attempt": attempt})
                elif isinstance(payload, EngineError):
                    retryable = payload.retryable
                    errors.append({"code": payload.code, "message": str(payload), "retryable": retryable,
                                   "attempt": attempt, "details": payload.details})
                else:
                    errors.append({"code": "engine_exception", "message": f"{type(payload).__name__}: {payload}",
                                   "retryable": False, "attempt": attempt})
            else:
                try:
                    images = _check_output(payload, out_dir, store)
                except ValueError as exc:
                    attempt_status = "malformed_response"
                    errors.append({"code": "malformed_response", "message": str(exc), "retryable": False, "attempt": attempt})
                else:
                    attempt_status, n_images = "succeeded", len(images)
                    seed_used, model_id, metadata = payload.seed_used, payload.model_id, payload.metadata
            budget.charge_attempt(caps.cost_usd_per_image, runtime, n_images)
            cost += caps.cost_usd_per_image
            attempts.append({"attempt": attempt, "status": attempt_status, "runtime_s": runtime})
            status = attempt_status
            if attempt_status == "succeeded" or not retryable:
                break

    if status != "succeeded":
        images = []
    result = {
        "schema_version": "atelier.generation_result.v1",
        "result_id": "result_" + request["candidate_id"].split("_", 1)[1],
        "request_id": request["request_id"],
        "candidate_id": request["candidate_id"],
        "status": status,
        "engine": {"engine_id": caps.engine_id, "model_id": model_id},
        "images": images,
        "seed_used": seed_used,
        "runtime_s": round(time.monotonic() - t0, 4),
        "attempts": attempts,
        "errors": errors,
        "cost": {"usd_estimated": round(cost, 6), "basis": caps.cost_basis},
        "started_at": started_at,
        "finished_at": utc_now(),
        "engine_metadata": metadata,
    }
    schemas.validate(result, "atelier.generation_result.v1")
    return result
