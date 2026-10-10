"""4. Engine failures, timeouts, malformed responses, retries; diffusers adapter without torch."""

import types
import unittest
from pathlib import Path

from PIL import Image

from atelier import schemas
from atelier.artifact_store import ArtifactStore
from atelier.budget import Budget, CancelToken
from atelier.engines.base import EngineError
from atelier.engines.diffusers_local import DiffusersLocalEngine
from atelier.engines.mock import MockEngine
from atelier.engines.runner import execute_request
from atelier.prompt_compiler import compile_request
from atelier.util import sha256_file
from tests.helpers import TempDirTest, art_direction, brief

LIMITS = {"max_total_images": 20, "max_generation_attempts": 40, "max_cost_usd": 1.0, "max_engine_seconds": 600}


def make_request(engine, slot=0, timeout_s=5.0, max_retries=1, engine_params=None):
    return compile_request(art_direction(), brief(), engine.capabilities(), run_id="run_eng", iteration=0, slot=slot,
                           base_seed=7, keep_seeds=True, engine_params=engine_params or {}, timeout_s=timeout_s,
                           max_retries=max_retries)


class RunnerTests(TempDirTest):
    def setUp(self):
        super().setUp()
        self.store = ArtifactStore(self.tmp / "run_eng")
        self.budget = Budget(LIMITS, Budget.empty_ledger())

    def run_req(self, engine, **kw):
        res = execute_request(engine, make_request(engine, **kw), self.store, self.budget, CancelToken())
        schemas.validate(res, "atelier.generation_result.v1")
        return res

    def test_success_records_real_artifacts(self):
        res = self.run_req(MockEngine())
        self.assertEqual(res["status"], "succeeded")
        img = res["images"][0]
        p = self.store.abs(img["path"])
        self.assertTrue(p.is_file())
        self.assertEqual(sha256_file(p), img["sha256"])
        with Image.open(p) as im:
            self.assertEqual(im.size, (img["width"], img["height"]))
        self.assertEqual(res["seed_used"], make_request(MockEngine())["seed"])
        self.assertEqual(res["engine"], {"engine_id": "mock", "model_id": "mock-procedural-v1"})
        self.assertEqual(self.budget.ledger["images"], 1)

    def test_mock_is_deterministic(self):
        a = self.run_req(MockEngine())
        store2 = ArtifactStore(self.tmp / "run_eng2")
        b = execute_request(MockEngine(), make_request(MockEngine()), store2, Budget(LIMITS, Budget.empty_ledger()), CancelToken())
        self.assertEqual(a["images"][0]["sha256"], b["images"][0]["sha256"])

    def test_transient_error_is_retried(self):
        res = self.run_req(MockEngine(failures={"0": ["transient", "ok"]}), max_retries=1)
        self.assertEqual(res["status"], "succeeded")
        self.assertEqual([a["status"] for a in res["attempts"]], ["failed", "succeeded"])
        self.assertEqual(res["errors"][0]["code"], "transient_backend_error")
        self.assertTrue(res["errors"][0]["retryable"])

    def test_retry_limit_enforced(self):
        res = self.run_req(MockEngine(failures={"0": ["transient"] * 5}), max_retries=2)
        self.assertEqual(res["status"], "failed")
        self.assertEqual(len(res["attempts"]), 3)
        self.assertEqual(res["images"], [])

    def test_permanent_error_not_retried(self):
        res = self.run_req(MockEngine(failures={"0": ["error", "ok"]}), max_retries=3)
        self.assertEqual(res["status"], "failed")
        self.assertEqual(len(res["attempts"]), 1)
        self.assertFalse(res["errors"][0]["retryable"])

    def test_timeout(self):
        res = self.run_req(MockEngine(failures={"0": ["timeout", "ok"]}), timeout_s=0.2, max_retries=1)
        self.assertEqual(res["status"], "timeout")
        self.assertEqual(len(res["attempts"]), 1, "timeouts are not retried unless retry_on_timeout")
        self.assertEqual(res["errors"][0]["code"], "timeout")

    def test_timeout_retry_when_enabled(self):
        eng = MockEngine(failures={"0": ["timeout", "ok"]})
        res = execute_request(eng, make_request(eng, timeout_s=2.0, max_retries=1), self.store, self.budget,
                              CancelToken(), retry_on_timeout=True)
        self.assertEqual(res["status"], "succeeded")
        self.assertEqual([a["status"] for a in res["attempts"]], ["timeout", "succeeded"])

    def test_malformed_file_detected(self):
        res = self.run_req(MockEngine(failures={"0": ["malformed"]}))
        self.assertEqual(res["status"], "malformed_response")
        self.assertIn("cannot be decoded", res["errors"][0]["message"])
        self.assertEqual(self.budget.ledger["images"], 0, "malformed output is not counted as an image")
        self.assertEqual(self.budget.ledger["attempts"], 1, "but the attempt is charged")

    def test_wrong_return_type_detected(self):
        res = self.run_req(MockEngine(failures={"0": ["not_an_object"]}))
        self.assertEqual(res["status"], "malformed_response")

    def test_image_outside_attempt_dir_rejected(self):
        class Escaping(MockEngine):
            def generate(self, request, out_dir, cancel):
                out = super().generate(request, out_dir, cancel)
                outside = Path(out_dir).parents[2] / "escape.png"
                Path(out.images[0]).replace(outside)
                out.images = [outside]
                return out

        res = self.run_req(Escaping())
        self.assertEqual(res["status"], "malformed_response")
        self.assertIn("outside", res["errors"][0]["message"])

    def test_unexpected_exception_is_recorded(self):
        class Boom(MockEngine):
            def generate(self, request, out_dir, cancel):
                raise ZeroDivisionError("bad math")

        res = self.run_req(Boom())
        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["errors"][0]["code"], "engine_exception")
        self.assertIn("ZeroDivisionError", res["errors"][0]["message"])

    def test_unavailable_engine(self):
        eng = DiffusersLocalEngine()  # torch/diffusers are not installed in the test environment
        if eng.availability().ok:
            self.skipTest("diffusers stack is installed here")
        res = execute_request(eng, make_request(eng), self.store, self.budget, CancelToken())
        self.assertEqual(res["status"], "unavailable")
        self.assertEqual(res["attempts"], [])
        self.assertEqual(self.budget.ledger["attempts"], 0, "nothing ran, nothing charged")

    def test_cancelled_before_attempt(self):
        tok = CancelToken()
        tok.cancel()
        eng = MockEngine()
        res = execute_request(eng, make_request(eng), self.store, self.budget, tok)
        self.assertEqual(res["status"], "cancelled")
        self.assertEqual(eng.calls, [])


class _FakeImage:
    def __init__(self, size):
        self.size = size

    def save(self, path, format="PNG"):
        Image.new("RGB", self.size, (90, 90, 90)).save(path, format=format)


class _FakePipe:
    def __init__(self):
        self.calls = []
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def set_progress_bar_config(self, **kw):
        pass

    def __call__(self, **kw):
        self.calls.append(kw)
        kw["callback_on_step_end"](self, 0, 0, {})
        return types.SimpleNamespace(images=[_FakeImage((kw["width"], kw["height"]))])


class _FakeGenerator:
    def __init__(self, device):
        self.seed = None

    def manual_seed(self, s):
        self.seed = s
        return self


def fake_torch():
    return types.SimpleNamespace(
        Generator=_FakeGenerator, float32="float32", float16="float16",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )


class DiffusersAdapterTests(TempDirTest):
    """Exercises the real adapter code path with an injected fake pipeline (no torch needed)."""

    def make(self, **kw):
        self.pipe = _FakePipe()
        self.loaded_with = {}

        def factory(model_id, **kwargs):
            self.loaded_with = {"model_id": model_id, **kwargs}
            return self.pipe

        return DiffusersLocalEngine(_pipeline_factory=factory, _torch=fake_torch(), **kw)

    def test_parameters_reach_pipeline(self):
        eng = self.make()
        store = ArtifactStore(self.tmp / "run_d")
        req = make_request(eng, engine_params={"num_inference_steps": 3})
        res = execute_request(eng, req, store, Budget(LIMITS, Budget.empty_ledger()), CancelToken())
        self.assertEqual(res["status"], "succeeded", res["errors"])
        call = self.pipe.calls[0]
        self.assertEqual(call["prompt"], req["prompt"]["positive"])
        self.assertEqual((call["width"], call["height"]), (req["width"], req["height"]))
        self.assertEqual(call["num_inference_steps"], 3)
        self.assertEqual(call["guidance_scale"], 0.0)
        self.assertNotIn("negative_prompt", call, "negative prompt is not sent when compiled to None")
        self.assertEqual(call["generator"].seed, req["seed"])
        self.assertEqual(res["seed_used"], req["seed"])
        self.assertEqual(res["engine"]["model_id"], "stabilityai/sd-turbo")
        self.assertEqual(res["engine_metadata"]["device"], "cpu")
        self.assertTrue(self.loaded_with["local_files_only"], "downloads are off by default")
        self.assertEqual(self.pipe.device, "cpu")

    def test_allow_download_must_be_explicit(self):
        eng = self.make(allow_download=True)
        store = ArtifactStore(self.tmp / "run_d2")
        execute_request(eng, make_request(eng), store, Budget(LIMITS, Budget.empty_ledger()), CancelToken())
        self.assertFalse(self.loaded_with["local_files_only"])

    def test_oom_is_structured_error(self):
        eng = self.make()

        def oom(**kw):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        self.pipe.__class__ = type("OOMPipe", (_FakePipe,), {"__call__": lambda self, **kw: oom(**kw)})
        store = ArtifactStore(self.tmp / "run_d3")
        res = execute_request(eng, make_request(eng), store, Budget(LIMITS, Budget.empty_ledger()), CancelToken())
        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["errors"][0]["code"], "out_of_memory")

    def test_load_failure_is_structured_error(self):
        def factory(model_id, **kw):
            raise OSError("model not found in local cache")

        eng = DiffusersLocalEngine(_pipeline_factory=factory, _torch=fake_torch())
        store = ArtifactStore(self.tmp / "run_d4")
        res = execute_request(eng, make_request(eng), store, Budget(LIMITS, Budget.empty_ledger()), CancelToken())
        self.assertEqual(res["errors"][0]["code"], "model_load_failed")

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            DiffusersLocalEngine(model_id="someone/unknown-model")

    def test_capabilities(self):
        c = DiffusersLocalEngine().capabilities()
        self.assertEqual(c.max_prompt_tokens, 75)
        self.assertEqual(c.max_reference_images, 0)
        self.assertFalse(c.negative_prompt_effective({"guidance_scale": 0.0}))
        sd15 = DiffusersLocalEngine(model_id="stable-diffusion-v1-5/stable-diffusion-v1-5").capabilities()
        self.assertTrue(sd15.negative_prompt_effective({"guidance_scale": 7.0}))


if __name__ == "__main__":
    unittest.main()
