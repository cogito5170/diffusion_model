"""OPTIONAL live integration test: real diffusers inference. Disabled by default.

Enable explicitly (after reading docs/LIVE_GENERATION.md):

    ATELIER_LIVE_TEST=1 ATELIER_LIVE_MODEL=stabilityai/sd-turbo python -m unittest tests.test_live_diffusers

It never downloads weights unless ATELIER_LIVE_ALLOW_DOWNLOAD=1 is also set.
"""

import os
import unittest

from atelier.artifact_store import ArtifactStore
from atelier.budget import Budget, CancelToken
from atelier.engines.diffusers_local import DiffusersLocalEngine
from atelier.engines.runner import execute_request
from atelier.prompt_compiler import compile_request
from tests.helpers import TempDirTest, art_direction, brief

LIVE = os.environ.get("ATELIER_LIVE_TEST") == "1"


@unittest.skipUnless(LIVE, "live test disabled; set ATELIER_LIVE_TEST=1 to enable (see docs/LIVE_GENERATION.md)")
class LiveDiffusersTest(TempDirTest):
    def test_single_real_generation(self):
        engine = DiffusersLocalEngine(
            model_id=os.environ.get("ATELIER_LIVE_MODEL", "stabilityai/sd-turbo"),
            allow_download=os.environ.get("ATELIER_LIVE_ALLOW_DOWNLOAD") == "1",
        )
        av = engine.availability()
        self.assertTrue(av.ok, av.reason)
        req = compile_request(art_direction(), brief(), engine.capabilities(), run_id="run_live", iteration=0, slot=0,
                              base_seed=1234, keep_seeds=True, engine_params={}, timeout_s=float(os.environ.get("ATELIER_LIVE_TIMEOUT", "900")),
                              max_retries=0)
        store = ArtifactStore(self.tmp / "run_live")
        budget = Budget({"max_total_images": 1, "max_generation_attempts": 1, "max_cost_usd": 0.0, "max_engine_seconds": 1800},
                        Budget.empty_ledger())
        res = execute_request(engine, req, store, budget, CancelToken())
        print("\nlive result:", res["status"], res["runtime_s"], "s", res["engine_metadata"], res["errors"])
        self.assertEqual(res["status"], "succeeded", res["errors"])
        self.assertEqual((res["images"][0]["width"], res["images"][0]["height"]), (req["width"], req["height"]))


if __name__ == "__main__":
    unittest.main()
