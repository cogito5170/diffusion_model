"""5. Budget enforcement and cancellation."""

import unittest

from atelier.budget import Budget, BudgetExceeded, Cancelled, CancelToken
from atelier.engines.mock import MockEngine
from atelier.orchestrator import Orchestrator
from tests.helpers import TempDirTest, config, new_run

LIMITS = {"max_total_images": 3, "max_generation_attempts": 4, "max_cost_usd": 0.10, "max_engine_seconds": 100}


class BudgetUnitTests(unittest.TestCase):
    def test_limits(self):
        b = Budget(LIMITS, Budget.empty_ledger())
        b.check_attempt(0.03)
        b.charge_attempt(0.03, 1.0, 1)
        b.charge_attempt(0.03, 1.0, 1)
        b.charge_attempt(0.03, 1.0, 1)
        with self.assertRaises(BudgetExceeded) as ctx:
            b.check_attempt(0.0)
        self.assertEqual(ctx.exception.reason, "max_total_images")

    def test_cost_limit(self):
        b = Budget(LIMITS, Budget.empty_ledger())
        b.charge_attempt(0.09, 1.0, 1)
        with self.assertRaises(BudgetExceeded) as ctx:
            b.check_attempt(0.02)
        self.assertEqual(ctx.exception.reason, "max_cost_usd")

    def test_attempt_limit_counts_failures(self):
        b = Budget(LIMITS, Budget.empty_ledger())
        for _ in range(4):
            b.charge_attempt(0.0, 1.0, 0)
        with self.assertRaises(BudgetExceeded) as ctx:
            b.check_attempt(0.0)
        self.assertEqual(ctx.exception.reason, "max_generation_attempts")

    def test_engine_seconds_limit(self):
        b = Budget(LIMITS, Budget.empty_ledger())
        b.charge_attempt(0.0, 100.0, 1)
        with self.assertRaises(BudgetExceeded):
            b.check_attempt(0.0)

    def test_can_afford_batch(self):
        b = Budget(LIMITS, Budget.empty_ledger())
        self.assertEqual(b.can_afford_batch(3, 0.0), (True, None))
        ok, why = b.can_afford_batch(4, 0.0)
        self.assertFalse(ok)
        self.assertIn("images", why)
        ok, why = b.can_afford_batch(2, 0.06)
        self.assertFalse(ok)
        self.assertIn("USD", why)

    def test_cancel_token(self):
        parent = CancelToken()
        child = parent.child()
        self.assertFalse(child.is_cancelled())
        parent.cancel()
        self.assertTrue(child.is_cancelled())
        with self.assertRaises(Cancelled):
            child.raise_if_cancelled()


class BudgetInRunTests(TempDirTest):
    def test_budget_stops_generation_mid_batch_and_is_recorded(self):
        cfg = config(budget={"max_total_images": 2, "max_generation_attempts": 10, "max_cost_usd": 0.0, "max_engine_seconds": 600},
                     selection={"shortlist_size": 2, "mmr_lambda": 0.7, "near_duplicate_threshold": 0.08})
        orch = new_run(self.tmp, cfg)
        orch.run()
        cands = orch.state["iterations"][0]["candidates"]
        statuses = [orch.store.get(c["result_id"])["status"] for c in cands]
        self.assertEqual(statuses, ["succeeded", "succeeded", "budget_exceeded", "budget_exceeded"])
        self.assertEqual(orch.state["ledger"]["images"], 2)
        self.assertEqual(len(orch.state["iterations"]), 1, "no refinement batch without budget")
        self.assertEqual(orch.state["state"], "AWAITING_HUMAN_APPROVAL")
        self.assertTrue(orch.state["stop_reason"].startswith(("budget_insufficient", "curator", "target", "plateau")))

    def test_iteration_not_started_when_batch_unaffordable(self):
        cfg = config(budget={"max_total_images": 6, "max_generation_attempts": 10, "max_cost_usd": 0.0, "max_engine_seconds": 600},
                     stopping={"target_weighted_score": 5.0, "min_dimension_for_target": 5, "min_improvement": 0.0})
        orch = new_run(self.tmp, cfg)
        orch.run()
        self.assertEqual(orch.state["ledger"]["images"], 4)
        self.assertTrue(orch.state["stop_reason"].startswith("budget_insufficient_for_next_batch"), orch.state["stop_reason"])

    def test_cost_budget_with_paid_engine(self):
        cfg = config(engine={"engine_id": "mock", "model_id": "mock-procedural-v1", "options": {"cost_usd_per_image": 0.04}},
                     budget={"max_total_images": 12, "max_generation_attempts": 24, "max_cost_usd": 0.10, "max_engine_seconds": 600},
                     selection={"shortlist_size": 2, "mmr_lambda": 0.7, "near_duplicate_threshold": 0.08})
        orch = new_run(self.tmp, cfg)
        orch.run()
        statuses = [orch.store.get(c["result_id"])["status"] for c in orch.state["iterations"][0]["candidates"]]
        self.assertEqual(statuses.count("succeeded"), 2)
        self.assertLessEqual(orch.state["ledger"]["cost_usd_estimated"], 0.10)

    def test_cancel_via_flag_file_between_candidates(self):
        class CancelAfterTwo(MockEngine):
            def generate(self, request, out_dir, cancel):
                out = super().generate(request, out_dir, cancel)
                if len(self.calls) == 2:
                    orch.request_cancel()
                return out

        orch = new_run(self.tmp, engine=CancelAfterTwo())
        orch.run()
        self.assertEqual(orch.state["state"], "CANCELLED")
        done = [c for c in orch.state["iterations"][0]["candidates"] if c["result_id"]]
        self.assertEqual(len(done), 2, "cancellation takes effect at the next candidate boundary")
        self.assertTrue((orch.store.root / "report" / "report.md").exists(), "partial report on cancel")
        self.assertEqual(orch.store.verify(), [])

    def test_cancel_interrupts_running_generation(self):
        class Slow(MockEngine):
            def generate(self, request, out_dir, cancel):
                orch.request_cancel()  # cancellation arrives while the engine is busy
                return super().generate(request, out_dir, cancel)

        orch = new_run(self.tmp, engine=Slow(failures={"*": ["timeout"]}))
        orch.run()
        self.assertEqual(orch.state["state"], "CANCELLED")
        first = orch.store.get(orch.state["iterations"][0]["candidates"][0]["result_id"])
        self.assertEqual(first["status"], "cancelled")

    def test_cancelled_run_is_terminal_on_resume(self):
        orch = new_run(self.tmp)
        orch.request_cancel()
        orch.run()
        self.assertEqual(orch.state["state"], "CANCELLED")
        again = Orchestrator.load(orch.store.root)
        again.run()
        self.assertEqual(again.state["state"], "CANCELLED")
        self.assertEqual(again.state["ledger"]["attempts"], 0)


if __name__ == "__main__":
    unittest.main()
