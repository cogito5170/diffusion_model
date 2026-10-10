"""The non-developer menu (easy.py): input parsing and one end-to-end pass with the mock engine."""

import io
import os
import shutil
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT))
import easy  # noqa: E402


class EasyMenuTests(unittest.TestCase):
    def test_parse_slot(self):
        self.assertEqual(easy.parse_slot("574x172"), (574, 172))
        self.assertEqual(easy.parse_slot(" 574 × 172 "), (574, 172))
        self.assertIsNone(easy.parse_slot("574 by 172"))

    def test_korean_detection(self):
        self.assertTrue(easy.has_korean("콘크리트 방"))
        self.assertFalse(easy.has_korean("a concrete room"))

    def test_prompt_flow_with_mock_engine(self):
        answers = iter(["1", "a quiet concrete room with one brushed steel bench", "", "1", "574x172", "q"])
        with mock.patch.dict(os.environ, {"ATELIER_EASY_CONFIG": "configs/mock_offline.json"}), \
                mock.patch("builtins.input", lambda *_: next(answers)), \
                mock.patch.object(easy, "new_name", return_value="easy-unittest"), \
                mock.patch.object(easy, "CONFIG", "configs/mock_offline.json"), \
                mock.patch.object(easy, "open_paths") as opened, redirect_stdout(io.StringIO()):
            try:
                self.assertEqual(easy.main(), 0)
            finally:
                out = ROOT / "runs" / "run_easy-unittest"
                paths = opened.call_args[0][0] if opened.call_args else []
                prompt_saved = (out / "prompt.txt").exists()
                shutil.rmtree(out, ignore_errors=True)
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].endswith("slot_574x172.png"))
        self.assertTrue(prompt_saved)


if __name__ == "__main__":
    unittest.main()


class InstallerScriptTests(unittest.TestCase):
    def test_install_script_parses(self):
        import subprocess

        self.assertEqual(subprocess.call(["bash", "-n", str(ROOT / "install_mac.sh")]), 0)
        text = (ROOT / "install_mac.sh").read_text()
        self.assertIn("easy.py", text)
        self.assertIn("download_model.py sd-turbo", text)
        self.assertIn("download_model.py sdxl", text)
        self.assertIn("studio.py", text)
        self.assertIn("pkgutil --check-signature", text, "the downloaded Python installer must be signature-checked")
        self.assertNotIn("brew install", text, "must not depend on Homebrew")
        self.assertNotIn("rsync -", text, "macOS 15 ships openrsync; the update step uses plain mv instead")
        self.assertNotIn("/usr/bin/python3", text.split("pick_python()")[1].split("}")[0],
                         "never invoke Apple's python3 stub (it pops up a developer-tools dialog)")
