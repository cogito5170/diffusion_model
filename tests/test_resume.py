"""9. Resuming an interrupted run."""

from atelier.engines.mock import MockEngine, SimulatedCrash
from atelier.orchestrator import Orchestrator
from tests.helpers import TempDirTest, new_run


class ResumeTests(TempDirTest):
    def test_crash_mid_batch_resumes_without_regenerating(self):
        crashing = MockEngine(failures={"2": ["crash"]})
        orch = new_run(self.tmp, engine=crashing)
        with self.assertRaises(SimulatedCrash):
            orch.run()
        on_disk = Orchestrator.load(orch.store.root, engine=MockEngine())
        self.assertEqual(on_disk.state["state"], "PROMPTS_COMPILED")
        done = [c["candidate_id"] for c in on_disk.state["iterations"][0]["candidates"] if c["result_id"]]
        self.assertEqual(len(done), 2, "slots 0 and 1 were persisted before the crash")
        self.assertEqual(on_disk.state["ledger"]["attempts"], 2)

        fresh = MockEngine()
        resumed = Orchestrator.load(orch.store.root, engine=fresh)
        resumed.run()
        first_batch_calls = [c["candidate_id"] for c in fresh.calls if "-i0-" in c["candidate_id"]]
        self.assertNotIn(done[0], first_batch_calls)
        self.assertNotIn(done[1], first_batch_calls)
        self.assertEqual(len(first_batch_calls), 2, "only the two missing candidates were generated")
        self.assertEqual(resumed.state["state"], "AWAITING_HUMAN_APPROVAL")
        self.assertEqual(resumed.store.verify(), [])

    def test_resume_is_idempotent_at_pause(self):
        orch = new_run(self.tmp)
        orch.run()
        snapshot = dict(orch.state)
        ledger = dict(orch.state["ledger"])
        again = Orchestrator.load(orch.store.root)
        again.run()
        self.assertEqual(again.state["state"], snapshot["state"])
        self.assertEqual(again.state["ledger"], ledger, "resuming at the approval gate generates nothing")

    def test_resume_after_approval_from_new_process(self):
        orch = new_run(self.tmp)
        orch.run()
        cid = orch.store.get(orch._it()["selection_id"])["shortlist"][0]
        second = Orchestrator.load(orch.store.root)
        second.approve(cid, "reviewer")
        third = Orchestrator.load(orch.store.root)
        third.run()
        self.assertEqual(third.state["state"], "DONE")
        self.assertEqual(third.state["final_candidate_id"], cid)

    def test_crash_during_critique_step(self):
        from atelier.agents_io import OfflineBackend

        class CrashOnce(OfflineBackend):
            crashed = False

            def obtain(self, req, store, previous_errors, attempt):
                if req.step == "critique" and not CrashOnce.crashed and req.key.endswith("s1"):
                    CrashOnce.crashed = True
                    raise SimulatedCrash("die while critiquing")
                return super().obtain(req, store, previous_errors, attempt)

        orch = new_run(self.tmp, agents=CrashOnce())
        with self.assertRaises(SimulatedCrash):
            orch.run()
        resumed = Orchestrator.load(orch.store.root, agents=OfflineBackend())
        self.assertEqual(resumed.state["state"], "QA_DONE")
        resumed.run()
        self.assertEqual(resumed.state["state"], "AWAITING_HUMAN_APPROVAL")
        crit_ids = [c["critique_id"] for c in resumed.state["iterations"][0]["candidates"]]
        self.assertTrue(all(crit_ids))
