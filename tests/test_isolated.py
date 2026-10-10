"""Engine isolation: a worker process that can die without taking the caller with it (mock engine only)."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from atelier.artifact_store import ArtifactStore
from atelier.budget import Budget, CancelToken
from atelier.engines import create_engine
from atelier.engines.isolated import IsolatedEngine, describe_exit
from atelier.engines.runner import execute_request
from atelier.prompt_compiler import compile_baseline_request
from tests.helpers import ROOT, TempDirTest, config

LIMITS = {"max_total_images": 20, "max_generation_attempts": 40, "max_cost_usd": 1.0, "max_engine_seconds": 600}


def engine_cfg(**options):
    return {"engine_id": "mock", "model_id": "mock-procedural-v1", "options": {"isolate": True, **options}}


class IsolatedEngineTests(TempDirTest):
    def setUp(self):
        super().setUp()
        self.store = ArtifactStore(self.tmp / "run_iso")
        self.budget = Budget(LIMITS, Budget.empty_ledger())

    def req(self, engine, slot=0, timeout_s=60.0):
        return compile_baseline_request("a steel arch", engine.capabilities(), run_id="run_iso", slot=slot,
                                        width=768, height=432, base_seed=1, engine_params={}, timeout_s=timeout_s,
                                        max_retries=0)

    def run_one(self, engine, **kw):
        return execute_request(engine, self.req(engine, **kw), self.store, self.budget, CancelToken())

    def test_factory_wraps_when_isolate_is_set(self):
        engine = create_engine(engine_cfg())
        self.assertIsInstance(engine, IsolatedEngine)
        self.assertEqual(engine.capabilities().model_id, "mock-procedural-v1")
        engine.close()

    def test_generates_in_worker_and_reuses_it(self):
        engine = create_engine(engine_cfg())
        try:
            r1 = self.run_one(engine, slot=0)
            pid = engine._proc.pid
            r2 = self.run_one(engine, slot=1)
            self.assertEqual((r1["status"], r2["status"]), ("succeeded", "succeeded"))
            self.assertTrue(r1["engine_metadata"]["isolated_worker"])
            self.assertEqual(engine._proc.pid, pid, "the loaded model stays in one worker between requests")
        finally:
            engine.close()
        self.assertIsNone(engine._proc)

    def test_worker_killed_by_os_is_reported_and_recovered(self):
        engine = create_engine(engine_cfg(failures={"0": ["sigkill"]}))
        try:
            crashed = self.run_one(engine, slot=0)
            self.assertEqual(crashed["status"], "failed")
            err = crashed["errors"][0]
            self.assertEqual(err["code"], "engine_process_crashed")
            self.assertEqual(err["details"]["exit_code"], -9)
            self.assertIn("memory", err["message"])
            recovered = self.run_one(engine, slot=1)
            self.assertEqual(recovered["status"], "succeeded", "a fresh worker is started after a crash")
        finally:
            engine.close()

    def test_timeout_stops_cooperatively_and_keeps_the_model_loaded(self):
        engine = create_engine(engine_cfg(failures={"0": ["timeout"]}))
        try:
            res = self.run_one(engine, slot=0, timeout_s=1.5)
            self.assertEqual(res["status"], "timeout")
            import time

            time.sleep(1.0)  # the worker acknowledges the stop at its next step
            pid = engine._proc.pid
            again = self.run_one(engine, slot=1)
            self.assertEqual(again["status"], "succeeded")
            self.assertEqual(engine._proc.pid, pid, "same worker, model still loaded")
        finally:
            engine.close()

    def test_unresponsive_worker_is_killed_after_grace(self):
        engine = create_engine(engine_cfg(failures={"0": ["hang"]}))
        engine.CANCEL_GRACE_S = 0.5
        try:
            res = self.run_one(engine, slot=0, timeout_s=1.0)
            self.assertEqual(res["status"], "timeout")
            import time

            deadline = time.time() + 15
            while engine._proc is not None and time.time() < deadline:
                time.sleep(0.1)
            self.assertIsNone(engine._proc, "a worker that ignores the stop request is terminated")
        finally:
            engine.close()

    def test_progress_is_forwarded_from_the_worker(self):
        engine = create_engine(engine_cfg())
        events = []
        engine.progress_callback = lambda stage, step, total: events.append((stage, step, total))
        try:
            req = self.req(engine)
            req["engine_params"]["num_inference_steps"] = 3
            res = execute_request(engine, req, self.store, self.budget, CancelToken())
            self.assertEqual(res["status"], "succeeded")
            self.assertEqual([e for e in events if e[0] == "step"][-1], ("step", 3, 3))
        finally:
            engine.close()

    def test_engine_errors_cross_the_process_boundary(self):
        engine = create_engine(engine_cfg(failures={"0": ["error"]}))
        try:
            res = self.run_one(engine, slot=0)
            self.assertEqual(res["errors"][0]["code"], "backend_error")
        finally:
            engine.close()

    def test_exit_descriptions(self):
        self.assertIn("memory", describe_exit(-9))
        self.assertIn("MPS", describe_exit(-6))


class StudioWorkerCrashTests(unittest.TestCase):
    def test_studio_survives_a_worker_crash(self):
        sys.path.insert(0, str(ROOT))
        import studio

        tmp = Path(tempfile.mkdtemp(prefix="iso-studio-"))
        before = set(studio.RUNS.glob("run_studio-*")) if studio.RUNS.exists() else set()
        cfg = config(engine=engine_cfg(failures={"0": ["sigkill"]}))
        (tmp / "c.json").write_text(json.dumps(cfg))
        s = studio.Studio({"fast": str(tmp / "c.json")})
        try:
            job = s.generate("a steel arch --n 3", {}, model="fast")
            self.assertEqual(len(job["images"]), 1)
            self.assertIn("메모리", job["error"])
            # The studio keeps working: the next request gets a fresh worker (which crashes again here only because
            # the mock is told to crash on slot 0) and is reported, instead of the studio process dying.
            again = s.generate("a steel arch --n 1", {"seed": 5}, model="fast")
            self.assertIn("메모리", again["error"])
        finally:
            s.engine.close()
            for d in set(studio.RUNS.glob("run_studio-*")) - before:
                shutil.rmtree(d, ignore_errors=True)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
