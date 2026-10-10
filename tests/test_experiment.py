"""Experiment harness: A/B/C generation, blind packet, analysis (synthetic ratings, temp dir only)."""

import csv
import io
import json
import sys
from contextlib import redirect_stdout

from tests.helpers import ROOT, TempDirTest

sys.path.insert(0, str(ROOT / "experiments"))
import run_comparison  # noqa: E402


class ExperimentHarnessTests(TempDirTest):
    def test_run_and_analyze(self):
        out = self.tmp / "exp1"
        args = ["run", "--brief", str(ROOT / "examples/briefs/metallic_identity.json"),
                "--config", str(ROOT / "configs/mock_offline.json"), "--out", str(out), "--images-per-condition", "3"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(run_comparison.main(args), 0)
        metrics = json.loads((out / "automatic_metrics.json").read_text())
        self.assertEqual(set(metrics) - {"_labels", "design"}, {"A", "B", "C"})
        self.assertEqual(metrics["A"]["n_requested"], 3)
        self.assertEqual(metrics["B"]["n_requested"], 3)
        key = json.loads((out / "blind" / "key.json").read_text())
        files = sorted(p.name for p in (out / "blind" / "images").iterdir())
        self.assertEqual(len(files), len(key))
        self.assertTrue(all("cand_" not in f and "-a" not in f for f in files), "blind file names leak nothing")

        with redirect_stdout(io.StringIO()):
            self.assertEqual(run_comparison.main(["analyze", "--out", str(out)]), 1, "no ratings -> no claims")
        with open(out / "blind" / "ratings.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for i, r in enumerate(rows):  # synthetic ratings purely to exercise the code path
            base = {"A": 2, "B": 3, "C": 3}[key[r["blind_id"]]["condition"]]
            r["overall_1to5"] = str(base + i % 2)
        with open(out / "blind" / "ratings.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=run_comparison.RATING_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(run_comparison.main(["analyze", "--out", str(out)]), 0)
        res = json.loads((out / "human_results.json").read_text())
        self.assertIn("overall_1to5:B-A", res["comparisons"])
        self.assertTrue(res["caveats"])

    def test_out_of_range_rating_rejected(self):
        out = self.tmp / "exp2"
        (out / "blind").mkdir(parents=True)
        (out / "blind" / "key.json").write_text(json.dumps({"img001": {"condition": "A", "candidate_id": "c", "sha256": "x"}}))
        with open(out / "blind" / "ratings.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(run_comparison.RATING_COLUMNS)
            w.writerow(["img001", "images/img001.png", "7", "", "", ""])
        with self.assertRaises(SystemExit):
            run_comparison.cmd_analyze(type("A", (), {"out": str(out)})())
