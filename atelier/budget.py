"""Budget enforcement and cooperative cancellation.

Every generation attempt is checked *before* it runs and charged *after* it runs.
Limits are hard: exceeding one raises BudgetExceeded; nothing is silently skipped.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(message)


class Cancelled(RuntimeError):
    pass


class CancelToken:
    """Cancellation flag. Checks an in-memory event, a parent token and (optionally) a
    ``CANCEL`` file in the run directory, so another process can cancel a run."""

    def __init__(self, flag_path: Path | None = None, parent: "CancelToken | None" = None):
        self._event = threading.Event()
        self._flag_path = Path(flag_path) if flag_path else None
        self._parent = parent

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        if self._event.is_set():
            return True
        if self._flag_path is not None and self._flag_path.exists():
            self._event.set()
            return True
        return self._parent.is_cancelled() if self._parent else False

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise Cancelled("cancellation requested")

    def child(self) -> "CancelToken":
        return CancelToken(parent=self)


@dataclass
class Budget:
    limits: dict[str, Any]   # run_config["budget"]
    ledger: dict[str, Any]   # run_state["ledger"], mutated in place so it persists with state

    @staticmethod
    def empty_ledger() -> dict[str, Any]:
        return {"images": 0, "attempts": 0, "cost_usd_estimated": 0.0, "engine_seconds": 0.0}

    def remaining(self) -> dict[str, float]:
        return {
            "images": self.limits["max_total_images"] - self.ledger["images"],
            "attempts": self.limits["max_generation_attempts"] - self.ledger["attempts"],
            "cost_usd": round(self.limits["max_cost_usd"] - self.ledger["cost_usd_estimated"], 6),
            "engine_seconds": self.limits["max_engine_seconds"] - self.ledger["engine_seconds"],
        }

    def check_attempt(self, est_cost_usd: float) -> None:
        rem = self.remaining()
        if rem["attempts"] < 1:
            raise BudgetExceeded("max_generation_attempts", f"attempt limit {self.limits['max_generation_attempts']} reached")
        if rem["images"] < 1:
            raise BudgetExceeded("max_total_images", f"image limit {self.limits['max_total_images']} reached")
        if est_cost_usd > rem["cost_usd"] + 1e-12:
            raise BudgetExceeded(
                "max_cost_usd",
                f"estimated cost {est_cost_usd:.4f} USD exceeds remaining {rem['cost_usd']:.4f} USD",
            )
        if rem["engine_seconds"] <= 0:
            raise BudgetExceeded("max_engine_seconds", f"engine time limit {self.limits['max_engine_seconds']} s reached")

    def can_afford_batch(self, n_images: int, est_cost_per_image: float) -> tuple[bool, str | None]:
        rem = self.remaining()
        if rem["images"] < n_images:
            return False, f"only {rem['images']} images left in budget, batch needs {n_images}"
        if rem["attempts"] < n_images:
            return False, f"only {rem['attempts']} generation attempts left, batch needs {n_images}"
        if n_images * est_cost_per_image > rem["cost_usd"] + 1e-12:
            return False, f"batch estimate {n_images * est_cost_per_image:.4f} USD exceeds remaining {rem['cost_usd']:.4f} USD"
        if rem["engine_seconds"] <= 0:
            return False, "engine time budget exhausted"
        return True, None

    def charge_attempt(self, cost_usd: float, engine_seconds: float, images: int) -> None:
        self.ledger["attempts"] += 1
        self.ledger["images"] += images
        self.ledger["cost_usd_estimated"] = round(self.ledger["cost_usd_estimated"] + cost_usd, 6)
        self.ledger["engine_seconds"] = round(self.ledger["engine_seconds"] + engine_seconds, 4)
