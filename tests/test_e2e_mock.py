"""10. End-to-end workflow with the mock engine (offline agents), including the CLI."""

import io
import json
from contextlib import redirect_stdout

from atelier import schemas
from atelier.cli import main
from atelier.engines.mock import MockEngine
from tests.helpers import ROOT, TempDirTest, new_run


class EndToEndTests(TempDirTest):
    def test_full_workflow_to_done(self):
        orch = new_run(self.tmp)
        orch.run()
        st = orch.state
        visited = [h["to"] for h in st["history"]]
        for s in ("BRIEF_ACCEPTED", "RESEARCHED", "CONCEPTS_READY", "ART_DIRECTION_DRAFTED", "SPEC_VALIDATED",
                  "PROMPTS_COMPILED", "GENERATED", "QA_DONE", "CRITIQUED", "SELECTED", "AWAITING_HUMAN_APPROVAL"):
            self.assertIn(s, visited)
        self.assertEqual(st["state"], "AWAITING_HUMAN_APPROVAL", "human approval is the default gate")
        self.assertIsNone(st["final_candidate_id"])
        self.assertTrue((orch.store.root / "report" / "review.md").exists())
        self.assertTrue((orch.store.root / "report" / "contact_sheet.png").exists())

        sel = orch.store.get(orch._it()["selection_id"])
        orch.approve(sel["shortlist"][0], "e2e-test", "top of shortlist")
        orch.run()
        self.assertEqual(orch.state["state"], "DONE")
        report = json.loads((orch.store.root / "report" / "report.json").read_text())
        schemas.validate(report, "atelier.run_report.v1")
        self.assertTrue(report["integrity"]["verified"])
        self.assertEqual(report["final_candidate_id"], sel["shortlist"][0])
        self.assertEqual(report["critic_kinds"], ["heuristic"])
        self.assertIn("HEURISTIC", report["evidence_note"])
        md = (orch.store.root / "report" / "report.md").read_text()
        self.assertIn("Original intent (verbatim", md)
        self.assertIn(orch.store.get("brief_metallic-identity-01")["original_intent"]["text"], md)
        for kind in ("creative_brief", "research_notes", "concept_set", "art_direction_draft", "material_spec",
                     "art_direction", "generation_request", "generation_result", "qa_report", "critique",
                     "selection", "curation", "approval"):
            self.assertTrue(orch.store.ids(kind), kind)

    def test_partial_failures_do_not_stop_the_run(self):
        engine = MockEngine(failures={"1": ["error"], "2": ["blank"], "3": ["malformed"]})
        orch = new_run(self.tmp, engine=engine)
        orch.run()
        cands = orch.state["iterations"][0]["candidates"]
        statuses = [orch.store.get(c["result_id"])["status"] for c in cands]
        self.assertEqual(statuses, ["succeeded", "failed", "succeeded", "malformed_response"])
        qa_blank = orch.store.get(cands[2]["qa_id"])
        self.assertFalse(qa_blank["passed"], "a blank frame fails technical QA")
        self.assertIsNone(cands[2]["critique_id"], "QA failures are not sent to the critic")
        sel = orch.store.get(orch.state["iterations"][0]["selection_id"])
        self.assertEqual(sel["shortlist"][:1], [cands[0]["candidate_id"]])
        self.assertNotEqual(orch.state["state"], "FAILED")

    def test_all_generations_fail_gracefully(self):
        orch = new_run(self.tmp, engine=MockEngine(failures={"*": ["error"]}))
        orch.run()
        self.assertEqual(orch.state["state"], "FAILED")
        self.assertEqual(orch.state["failure"]["code"], "no_images_generated")
        self.assertTrue(orch.state["failure"]["details"]["errors"])
        self.assertTrue((orch.store.root / "report" / "report.md").exists())

    def test_unavailable_live_engine_fails_honestly(self):
        from atelier.engines.diffusers_local import DiffusersLocalEngine

        eng = DiffusersLocalEngine()
        if eng.availability().ok:
            self.skipTest("diffusers stack installed")
        orch = new_run(self.tmp, engine=eng)
        orch.run()
        self.assertEqual(orch.state["state"], "FAILED")
        res = orch.store.get(orch.state["iterations"][0]["candidates"][0]["result_id"])
        self.assertEqual(res["status"], "unavailable")
        self.assertEqual(orch.state["ledger"]["images"], 0)

    def test_auto_approval_only_when_configured(self):
        from tests.helpers import config

        orch = new_run(self.tmp, config(require_human_approval=False))
        orch.run()
        self.assertEqual(orch.state["state"], "DONE")
        ap = orch.store.get(orch.state["pointers"]["approval"])
        self.assertEqual(ap["reviewer"]["kind"], "config_auto")

    def test_cli_run_approve_verify(self):
        runs = str(self.tmp / "runs")
        brief = str(ROOT / "examples" / "briefs" / "metallic_identity.json")
        cfg = str(ROOT / "configs" / "mock_offline.json")
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["--runs-dir", runs, "run", "--brief", brief, "--config", cfg, "--run-id", "run_cli"]), 0)
        self.assertIn("WAITING for human approval", buf.getvalue())
        sel_dir = self.tmp / "runs" / "run_cli" / "artifacts" / "selection"
        shortlist = json.loads(sorted(sel_dir.glob("*.json"))[-1].read_text())["shortlist"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--runs-dir", runs, "approve", "run_cli", "--candidate", shortlist[0], "--reviewer", "cli"]), 0)
            self.assertEqual(main(["--runs-dir", runs, "verify", "run_cli"]), 0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--runs-dir", runs, "status", "run_cli"])
        self.assertIn("DONE", buf.getvalue())

    def test_cli_validate(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["validate", str(ROOT / "examples" / "briefs" / "metallic_identity.json")]), 0)
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps({"schema_version": "atelier.creative_brief.v1"}))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["validate", str(bad)]), 1)
