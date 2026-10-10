"""6. Artifact integrity and metadata persistence."""

import json

from atelier import schemas
from atelier.artifact_store import ArtifactStore, IntegrityError
from atelier.orchestrator import Orchestrator, RunError
from atelier.util import sha256_json
from tests.helpers import TempDirTest, brief, clone, config, example, new_run


class StoreTests(TempDirTest):
    def test_put_get_roundtrip_and_manifest(self):
        store = ArtifactStore(self.tmp / "run_s")
        ad = example("art_direction")
        art_id = store.put(ad)
        self.assertEqual(art_id, "ad_example-v1")
        self.assertEqual(store.get(art_id), ad)
        reopened = ArtifactStore(self.tmp / "run_s")
        self.assertEqual(reopened.get(art_id), ad, "manifest persists across instances")
        self.assertEqual(reopened.ids("art_direction"), [art_id])

    def test_invalid_artifact_never_written(self):
        store = ArtifactStore(self.tmp / "run_s")
        bad = example("critique")
        bad["ratings"]["originality"] = 9
        with self.assertRaises(schemas.SchemaValidationError):
            store.put(bad)
        self.assertEqual(store.ids(), [])
        self.assertFalse(list((self.tmp / "run_s" / "artifacts").rglob("*.json")))

    def test_ids_are_immutable(self):
        store = ArtifactStore(self.tmp / "run_s")
        ad = example("art_direction")
        store.put(ad)
        store.put(clone(ad))  # identical content: idempotent
        ad["mood"] = ["different"]
        with self.assertRaises(IntegrityError):
            store.put(ad)

    def test_tampering_detected(self):
        store = ArtifactStore(self.tmp / "run_s")
        art_id = store.put(example("art_direction"))
        path = store.path_of(art_id)
        data = json.loads(path.read_text())
        data["mood"] = ["tampered"]
        path.write_text(json.dumps(data))
        with self.assertRaises(IntegrityError):
            store.get(art_id)
        self.assertTrue(store.verify())

    def test_path_escape_rejected(self):
        store = ArtifactStore(self.tmp / "run_s")
        with self.assertRaises(IntegrityError):
            store.abs("../../etc/passwd")

    def test_no_temp_files_left_behind(self):
        store = ArtifactStore(self.tmp / "run_s")
        store.put(example("art_direction"))
        self.assertFalse([p for p in (self.tmp / "run_s").rglob("*.tmp")])


class RunMetadataTests(TempDirTest):
    def test_run_metadata_and_image_integrity(self):
        orch = new_run(self.tmp)
        orch.run()
        self.assertEqual(orch.store.verify(), [])
        res_id = orch.state["iterations"][0]["candidates"][0]["result_id"]
        res = orch.store.get(res_id)
        req = orch.store.get(res["request_id"])
        self.assertEqual(res["engine"]["model_id"], "mock-procedural-v1")
        self.assertEqual(res["seed_used"], req["seed"])
        self.assertGreater(res["runtime_s"], 0)
        # tamper with an image: verify must notice
        img = orch.store.abs(res["images"][0]["path"])
        img.write_bytes(img.read_bytes() + b"x")
        problems = orch.store.verify()
        self.assertTrue(any("sha256 mismatch" in p for p in problems), problems)

    def test_state_and_events_persist(self):
        orch = new_run(self.tmp)
        orch.run()
        reloaded = Orchestrator.load(orch.store.root)
        self.assertEqual(reloaded.state, orch.state)
        events = [e["event"] for e in orch.store.events()]
        self.assertIn("run_created", events)
        self.assertIn("generation_result", events)
        self.assertEqual([h["to"] for h in orch.state["history"]][:3], ["BRIEF_ACCEPTED", "RESEARCHED", "CONCEPTS_READY"])

    def test_original_intent_preserved_and_interpretation_separate(self):
        orch = new_run(self.tmp)
        orch.run()
        original = orch.store.get("brief_metallic-identity-01")
        interpreted = orch.store.get("brief_metallic-identity-01.interpreted")
        self.assertIsNone(original["system_interpretation"])
        self.assertEqual(original["original_intent"], brief()["original_intent"])
        self.assertEqual(interpreted["original_intent"], original["original_intent"])
        self.assertEqual(interpreted["system_interpretation"]["source_artifact_id"], orch.state["pointers"]["research"])
        self.assertEqual(orch.state["original_intent_sha256"], sha256_json(brief()["original_intent"]))

    def test_brief_with_prefilled_interpretation_rejected(self):
        b = brief()
        b["system_interpretation"] = {
            "interpreted_by": {"name": "x", "kind": "human"}, "interpreted_at": "2026-01-01T00:00:00Z",
            "source_artifact_id": "research_x", "summary": "s", "themes": [], "tensions": [], "open_questions": []}
        with self.assertRaises(RunError):
            new_run(self.tmp, b=b)

    def test_config_frozen(self):
        orch = new_run(self.tmp)
        cfg_path = orch.store.root / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["batch_size"] = 2
        cfg_path.write_text(json.dumps(cfg))
        with self.assertRaises(RunError):
            Orchestrator.load(orch.store.root)

    def test_duplicate_run_id_rejected(self):
        new_run(self.tmp)
        with self.assertRaises(RunError):
            new_run(self.tmp)

    def test_invalid_run_id_rejected(self):
        with self.assertRaises(RunError):
            new_run(self.tmp, run_id="../escape")
        with self.assertRaises(RunError):
            new_run(self.tmp, run_id="run_with_underscore")


if __name__ == "__main__":
    import unittest

    unittest.main()
