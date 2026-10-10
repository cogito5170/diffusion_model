"""Midjourney-style local studio: flag parsing, model switching and the HTTP API (mock engines only)."""

import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from tests.helpers import ROOT, config

sys.path.insert(0, str(ROOT))
import studio  # noqa: E402


class FlagParsingTests(unittest.TestCase):
    def test_flags_are_stripped_and_applied(self):
        o = studio.parse_prompt("steel arch under glowing ceiling --ar 16:9 --n 2 --seed 42 --crop 574x172", {})
        self.assertEqual(o["prompt"], "steel arch under glowing ceiling")
        self.assertEqual((o["ar"], o["n"], o["seed"], o["crop"]), ("16:9", 2, 42, (574, 172)))
        self.assertEqual((o["width"], o["height"]), (768, 432))

    def test_negative_flag_takes_a_list(self):
        o = studio.parse_prompt("a steel arch --no people, text, logo --seed 5 --ar 16:9", {})
        self.assertEqual((o["prompt"], o["negative"], o["seed"], o["ar"]), ("a steel arch", "people, text, logo", 5, "16:9"))
        o = studio.parse_prompt("a steel arch --no people", {})
        self.assertEqual(o["negative"], "people")

    def test_size_follows_the_model(self):
        self.assertEqual(studio.size_for_ratio("16:9", 1024 * 1024), (1368, 768))
        self.assertEqual(studio.size_for_ratio("1:1", 1024 * 1024), (1024, 1024))
        self.assertEqual(studio.size_for_ratio("16:9", 768 * 432), (768, 432))

    def test_defaults_and_random_seed(self):
        o = studio.parse_prompt("a chair", {})
        self.assertEqual((o["ar"], o["n"]), ("1:1", 4))
        self.assertIsInstance(o["seed"], int)

    def test_page_controls_are_overridden_by_prompt_flags(self):
        o = studio.parse_prompt("a chair --ar 9:16", {"ar": "16:9", "n": 1})
        self.assertEqual((o["ar"], o["n"]), ("9:16", 1))
        self.assertLess(o["width"], o["height"])

    def test_bad_values_rejected(self):
        for text in ("--ar 16:9", "x --n 9", "x --seed abc", "x --ar 10:3", "x --crop 574by172", "x --ar 0:1"):
            with self.subTest(text=text), self.assertRaises(studio.PromptError):
                studio.parse_prompt(text, {})


class StudioServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="studio-test-"))
        quality = config(engine={"engine_id": "mock", "model_id": "mock-sdxl-like",
                                 "options": {"profile": "stabilityai/stable-diffusion-xl-base-1.0"}},
                         engine_params={})
        qpath = cls.tmp / "quality.json"
        qpath.write_text(json.dumps(quality))
        cls.before = set(studio.RUNS.glob("run_studio-*")) if studio.RUNS.exists() else set()
        cls.server = studio.serve({"fast": "configs/mock_offline.json", "quality": str(qpath)}, port=0, open_browser=False)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        for d in set(studio.RUNS.glob("run_studio-*")) - cls.before:  # only what these tests created
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def post(self, body):
        req = urllib.request.Request(self.base + "/api/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_page_and_models(self):
        status, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("이미지 스튜디오".encode(), body)
        status, body = self.get("/api/models")
        models = {m["key"]: m for m in json.loads(body)}
        self.assertEqual(set(models), {"fast", "quality"})
        self.assertTrue(all(m["available"] for m in models.values()))

    def test_generate_grid_with_crop_and_history(self):
        status, job = self.post({"prompt": "a quiet concrete room --ar 16:9 --n 2 --seed 7 --crop 574x172"})
        self.assertEqual(status, 200, job)
        self.assertEqual([i["seed"] for i in job["images"]], [7, 7 + 7919])
        self.assertTrue(all(i["status"] == "succeeded" for i in job["images"]))
        status, png = self.get(job["images"][0]["url"])
        self.assertEqual(status, 200)
        self.assertTrue(png.startswith(b"\x89PNG"))
        status, _ = self.get(job["images"][0]["crop_url"])
        self.assertEqual(status, 200)
        status, hist = self.get("/api/history")
        self.assertIn(job["name"], [j["name"] for j in json.loads(hist)])

    def test_quality_model_uses_negative_and_larger_size(self):
        status, job = self.post({"prompt": "a steel arch --ar 16:9 --n 1 --no people, text", "model": "quality"})
        self.assertEqual(status, 200, job)
        self.assertEqual((job["width"], job["height"]), (1368, 768))
        self.assertEqual(job["negative_sent"], "people, text")
        status, job = self.post({"prompt": "a steel arch --n 1 --no people", "model": "fast"})
        self.assertEqual(status, 200, job)
        self.assertIsNone(job["negative_sent"], "the fast model ignores negatives; nothing is sent")
        self.assertTrue(any("negative" in w for w in job["warnings"]))

    def test_same_seed_same_image(self):
        _, a = self.post({"prompt": "steel bench", "n": 1, "seed": 99})
        _, b = self.post({"prompt": "steel bench", "n": 1, "seed": 99})
        _, pa = self.get(a["images"][0]["url"])
        _, pb = self.get(b["images"][0]["url"])
        self.assertEqual(pa, pb)

    def test_errors_are_reported_not_crashing(self):
        status, body = self.post({"prompt": "   "})
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        status, body = self.post({"prompt": "x", "model": "nope"})
        self.assertEqual(status, 400)
        status, _ = self.get("/runs/../../atelier/schemas/common.v1.json")
        self.assertEqual(status, 404)
        status, _ = self.get("/runs/../studio.py")
        self.assertEqual(status, 404)


class ModelSwitchTests(unittest.TestCase):
    def test_switching_models_frees_the_previous_one(self):
        s = studio.Studio({"fast": "configs/mock_offline.json", "other": "configs/mock_offline.json"})
        first = s._use("fast")
        closed = []
        first.close = lambda: closed.append(True)
        self.assertIs(s._use("fast"), first, "same model is reused, not reloaded")
        s._use("other")
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()


class StudioFailureTests(unittest.TestCase):
    def setUp(self):
        self.before = set(studio.RUNS.glob("run_studio-*")) if studio.RUNS.exists() else set()

    def tearDown(self):
        for d in set(studio.RUNS.glob("run_studio-*")) - self.before:
            shutil.rmtree(d, ignore_errors=True)

    def test_model_load_failure_stops_after_first_image(self):
        from atelier.engines.base import EngineError

        s = studio.Studio({"fast": "configs/mock_offline.json"})
        engine = s._use("fast")

        def broken(request, out_dir, cancel):
            raise EngineError("model_files_incomplete", "missing vae_1_0 file", retryable=False,
                              details={"fix": "python scripts/download_model.py sdxl"})

        engine.generate = broken
        job = s.generate("a steel arch --n 4", {}, model="fast")
        self.assertEqual(len(job["images"]), 1, "no point trying three more images with a broken model")
        self.assertIn("download_model.py sdxl", job["error"])

    def test_long_prompt_is_flagged_and_used_part_is_shown(self):
        s = studio.Studio({"fast": "configs/mock_offline.json"})
        long_prompt = " ".join(["monumental brushed steel sculpture bursting with white plaster"] * 20)
        job = s.generate(long_prompt + " --n 1", {}, model="fast")
        self.assertTrue(job["truncated"])
        self.assertLess(len(job["prompt_sent"].split()), len(long_prompt.split()))


class StudioProgressAndCancelTests(unittest.TestCase):
    def test_progress_is_reported_and_stop_works(self):
        import time

        tmp = Path(tempfile.mkdtemp(prefix="studio-prog-"))
        before = set(studio.RUNS.glob("run_studio-*")) if studio.RUNS.exists() else set()
        cfg = config(engine={"engine_id": "mock", "model_id": "mock-slow",
                             "options": {"isolate": True, "step_latency_s": 0.4}},
                     engine_params={"num_inference_steps": 4})
        (tmp / "slow.json").write_text(json.dumps(cfg))
        server = studio.serve({"fast": str(tmp / "slow.json")}, port=0, open_browser=False)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        result = {}

        def post():
            req = urllib.request.Request(base + "/api/generate", data=json.dumps({"prompt": "slow steel --n 4"}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                result["job"] = json.loads(r.read())

        worker = threading.Thread(target=post)
        try:
            worker.start()
            seen = None
            deadline = time.time() + 60
            while time.time() < deadline:
                with urllib.request.urlopen(base + "/api/progress", timeout=10) as r:
                    p = json.loads(r.read())
                if p.get("active") and p.get("stage") == "step" and p.get("step", 0) >= 2:
                    seen = p
                    break
                time.sleep(0.2)
            self.assertIsNotNone(seen, "progress should report steps while generating")
            self.assertEqual((seen["images"], seen["steps"]), (4, 4))
            self.assertIn("elapsed_s", seen)
            with urllib.request.urlopen(urllib.request.Request(base + "/api/cancel", data=b"", method="POST"), timeout=10) as r:
                self.assertTrue(json.loads(r.read())["cancelled"])
            worker.join(90)
            job = result["job"]
            self.assertIn("멈췄습니다", job["error"])
            self.assertLess(len([i for i in job["images"] if i["url"]]), 4)
            with urllib.request.urlopen(base + "/api/progress", timeout=10) as r:
                self.assertFalse(json.loads(r.read())["active"])
        finally:
            server.shutdown()
            for d in set(studio.RUNS.glob("run_studio-*")) - before:
                shutil.rmtree(d, ignore_errors=True)
            shutil.rmtree(tmp, ignore_errors=True)


class StudioWarmupAndTimeoutTests(unittest.TestCase):
    def _studio(self, tmp, options, timeout_s, steps=4):
        cfg = config(engine={"engine_id": "mock", "model_id": "mock-slow", "options": {"isolate": True, **options}},
                     engine_params={"num_inference_steps": steps})
        cfg["generation"]["timeout_s"] = timeout_s
        (tmp / "c.json").write_text(json.dumps(cfg))
        return studio.Studio({"fast": str(tmp / "c.json")})

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="studio-warm-"))
        self.before = set(studio.RUNS.glob("run_studio-*")) if studio.RUNS.exists() else set()

    def tearDown(self):
        for d in set(studio.RUNS.glob("run_studio-*")) - self.before:
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_model_loading_is_not_counted_against_the_image_timeout(self):
        s = self._studio(self.tmp, {"load_latency_s": 2.5}, timeout_s=2.0)
        try:
            job = s.generate("a steel arch --n 1", {}, model="fast")
        finally:
            s.engine.close()
        self.assertNotIn("error", job, job.get("error"))
        self.assertEqual(job["images"][0]["status"], "succeeded")
        self.assertGreaterEqual(job["load_s"], 2.0)

    def test_timeout_message_says_where_it_stopped(self):
        s = self._studio(self.tmp, {"step_latency_s": 0.6}, timeout_s=1.5)
        try:
            job = s.generate("a steel arch --n 2", {}, model="fast")
        finally:
            s.engine.close()
        self.assertEqual(len(job["images"]), 1)
        self.assertIn("초 안에 끝나지 않아", job["error"])
        self.assertIn("단계", job["error"])
