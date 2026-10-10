"""`imagine` terminal command: inline and pasted prompts, --hq, failures (mock engine only)."""

import io
import os
import shutil
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import imagine  # noqa: E402


class ImagineTests(unittest.TestCase):
    def setUp(self):
        self.before = set(imagine.studio.RUNS.glob("run_studio-*")) if imagine.studio.RUNS.exists() else set()
        self.env = mock.patch.dict(os.environ, {"ATELIER_IMAGINE_CONFIG": "configs/mock_offline.json"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        for d in set(imagine.studio.RUNS.glob("run_studio-*")) - self.before:
            shutil.rmtree(d, ignore_errors=True)

    def test_hq_flag(self):
        self.assertEqual(imagine.split_model("a chair --hq --ar 16:9"), ("a chair  --ar 16:9", "quality"))
        self.assertEqual(imagine.split_model("a chair"), ("a chair", "fast"))

    def test_inline_prompt(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = imagine.main(["a", "quiet", "concrete", "room", "--n", "2", "--seed", "3", "--crop", "574x172"])
        self.assertEqual(code, 0, out.getvalue())
        self.assertEqual(out.getvalue().count("시드 "), 2)

    def test_pasted_prompt_with_shell_unfriendly_characters(self):
        answers = iter(["a machine's skeleton (steel!) bursting with plaster? --n 1", ""])
        out = io.StringIO()
        with mock.patch("builtins.input", lambda *_: next(answers)), redirect_stdout(out):
            code = imagine.main([])
        self.assertEqual(code, 0, out.getvalue())

    def test_empty_prompt(self):
        with mock.patch("builtins.input", side_effect=EOFError), redirect_stdout(io.StringIO()):
            self.assertEqual(imagine.main([]), 1)


if __name__ == "__main__":
    unittest.main()
