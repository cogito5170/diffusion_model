"""Claude Code subagent hand-off: task files, strict ingest, rejection loop."""

import json

from atelier import schemas
from atelier.agents_io import FileDropBackend, OfflineBackend
from atelier.orchestrator import Orchestrator
from tests.helpers import TempDirTest, config, example, new_run


class FileDropTests(TempDirTest):
    def setUp(self):
        super().setUp()
        self.orch = new_run(self.tmp, config(agent_backend="claude_code"))

    def task(self):
        w = self.orch.state["waiting_for"]
        self.assertEqual(w["kind"], "agent")
        task = json.loads((self.orch.store.root / w["task_paths"][0]).read_text())
        schemas.validate(task, "atelier.agent_task.v1")
        return task

    def drop(self, task, artifact):
        (self.orch.store.root / task["output_path"]).write_text(json.dumps(artifact))

    def resume(self):
        self.orch = Orchestrator.load(self.orch.store.root)
        self.orch.run()

    def test_pauses_with_task_for_named_agent(self):
        self.orch.run()
        t = self.task()
        self.assertEqual((t["agent"], t["step"]), ("concept-researcher", "research"))
        self.assertEqual(t["expected_schema_version"], "atelier.research_notes.v1")
        self.assertEqual(t["required_values"]["research_id"], "research_t")
        self.assertTrue((self.orch.store.root / t["inputs"]["brief"]).exists())

    def test_valid_artifact_is_ingested(self):
        self.orch.run()
        t = self.task()
        art = example("research_notes")
        art.update(t["required_values"])
        self.drop(t, art)
        self.resume()
        self.assertEqual(self.orch.state["state"], "RESEARCHED")
        self.assertEqual(self.task()["agent"], "concept-generator")

    def test_invalid_artifact_rejected_and_task_reissued_with_errors(self):
        self.orch.run()
        t = self.task()
        art = example("research_notes")
        art.update(t["required_values"])
        art["unexpected"] = True
        self.drop(t, art)
        self.resume()
        self.assertEqual(self.orch.state["state"], "BRIEF_ACCEPTED")
        t2 = self.task()
        self.assertEqual(t2["attempt"], 2)
        self.assertTrue(any("unexpected" in e for e in t2["previous_errors"]))
        self.assertTrue(list((self.orch.store.root / "rejected").glob("research__brief.1.json")))

    def test_wrong_required_id_rejected(self):
        self.orch.run()
        t = self.task()
        art = example("research_notes")  # keeps the example's ids instead of the required ones
        self.drop(t, art)
        self.resume()
        self.assertTrue(any("research_id" in e for e in self.task()["previous_errors"]))

    def test_unparseable_json_rejected(self):
        self.orch.run()
        t = self.task()
        (self.orch.store.root / t["output_path"]).write_text("{not json")
        self.resume()
        self.assertTrue(any("not valid JSON" in e for e in self.task()["previous_errors"]))

    def test_repeated_rejection_fails_run(self):
        self.orch.run()
        for _ in range(3):
            t = self.task()
            self.drop(t, {"schema_version": "atelier.research_notes.v1"})
            self.resume()
        self.assertEqual(self.orch.state["state"], "FAILED")
        self.assertEqual(self.orch.state["failure"]["code"], "agent_output_rejected")
        self.assertTrue((self.orch.store.root / "report" / "report.md").exists())

    def test_full_run_with_file_drop_protocol(self):
        """Drive every agent step through files, answering each task with the offline stub's output."""
        stub = OfflineBackend()
        from atelier.agents_io import AgentRequest

        seen = []
        for _ in range(60):
            self.orch.run()
            w = self.orch.state["waiting_for"]
            if not w or w["kind"] != "agent":
                break
            seen.append(w["agent"])
            # Rebuild the request the orchestrator made, then answer it with the offline stub (stands in for Claude).
            offline = Orchestrator.load(self.orch.store.root, agents=_Recorder(stub))
            offline.step()
            for req, art in offline.agents.answers:
                (self.orch.store.root / "inbox" / f"{req.slug}.json").write_text(json.dumps(art))
            self.orch = Orchestrator.load(self.orch.store.root)
        self.assertEqual(self.orch.state["state"], "AWAITING_HUMAN_APPROVAL")
        for agent in ("concept-researcher", "concept-generator", "art-director", "material-specialist", "art-critic", "curator"):
            self.assertIn(agent, seen)


class _Recorder(OfflineBackend):
    """Offline backend that records answers instead of letting the orchestrator ingest them."""

    def __init__(self, inner):
        self.inner = inner
        self.answers = []

    def obtain(self, req, store, previous_errors, attempt):
        inbox = store.root / "inbox" / f"{req.slug}.json"
        if inbox.exists():
            return json.loads(inbox.read_text())
        self.answers.append((req, self.inner.obtain(req, store, previous_errors, attempt)))
        return None
