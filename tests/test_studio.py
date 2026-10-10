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
