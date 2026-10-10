"""1-2. JSON Schema validation, missing and unknown fields, strict typing."""

import unittest

from atelier import schemas
from tests.helpers import ROOT, brief, clone, config, example, load


class SpecCheckTests(unittest.TestCase):
    def test_negated_list_is_not_a_conflict(self):
        from atelier.spec_checks import check_art_direction

        ad = example("art_direction")
        ad["semantic_constraints"].append({"constraint": "no text, logos, signage or robot", "priority": "must"})
        errs, _ = check_art_direction(ad, brief())
        self.assertEqual(errs, [])
        ad["semantic_constraints"].append({"constraint": "a robot assembles the frame", "priority": "must"})
        errs, _ = check_art_direction(ad, brief())
        self.assertTrue(any("robot" in e for e in errs), "an affirmative mention still conflicts")


class SchemaTests(unittest.TestCase):
    def test_all_schemas_are_valid_draft_2020_12(self):
        checked = schemas.check_all_schemas()
        self.assertEqual(len(checked), len(schemas.ARTIFACT_TYPES))

    def test_examples_validate(self):
        schemas.validate(brief())
        for name in ("research_notes", "art_direction_draft", "material_spec", "art_direction", "critique",
                     "iteration_plan", "curation"):
            with self.subTest(name=name):
                schemas.validate(example(name))
        for cfg in sorted((ROOT / "configs").glob("*.json")):
            with self.subTest(cfg=cfg.name):
                schemas.validate(load(cfg))

    def test_every_artifact_declares_version_and_id(self):
        for version, (_, id_field) in schemas.ARTIFACT_TYPES.items():
            schema = load(schemas.SCHEMA_DIR / schemas.ARTIFACT_TYPES[version][0])
            self.assertIn("schema_version", schema["required"], version)
            self.assertIn(id_field, schema["required"], version)
            self.assertFalse(schema.get("additionalProperties", True), f"{version} must reject unknown fields")

    def test_missing_required_field_rejected(self):
        b = brief()
        del b["original_intent"]
        errs = schemas.validation_errors(b)
        self.assertTrue(any("original_intent" in e for e in errs), errs)

    def test_unknown_top_level_field_rejected(self):
        b = brief()
        b["surprise"] = 1
        self.assertTrue(any("surprise" in e for e in schemas.validation_errors(b)))

    def test_unknown_nested_field_rejected(self):
        ad = example("art_direction")
        ad["materials"][0]["glossiness"] = 0.9
        self.assertTrue(any("glossiness" in e for e in schemas.validation_errors(ad)))

    def test_unknown_schema_version_rejected(self):
        b = brief()
        b["schema_version"] = "atelier.creative_brief.v9"
        self.assertEqual(schemas.validation_errors(b), ["unknown schema_version 'atelier.creative_brief.v9'"])

    def test_expected_version_mismatch_rejected(self):
        self.assertTrue(schemas.validation_errors(brief(), "atelier.critique.v1"))

    def test_no_type_coercion(self):
        c = example("critique")
        c["ratings"]["composition"] = "4"
        self.assertTrue(schemas.validation_errors(c))
        c["ratings"]["composition"] = 4.0  # JSON number 4.0 is an integer in JSON Schema; 4.5 is not
        self.assertFalse(schemas.validation_errors(c))
        c["ratings"]["composition"] = 4.5
        self.assertTrue(schemas.validation_errors(c))

    def test_rating_range(self):
        c = example("critique")
        for bad in (0, 6):
            c["ratings"]["originality"] = bad
            self.assertTrue(schemas.validation_errors(c), bad)

    def test_critique_requires_all_six_ratings(self):
        c = example("critique")
        del c["ratings"]["technical_quality"]
        self.assertTrue(schemas.validation_errors(c))

    def test_failed_result_cannot_claim_images(self):
        r = {
            "schema_version": "atelier.generation_result.v1", "result_id": "result_x", "request_id": "req_x",
            "candidate_id": "cand_x", "status": "failed", "engine": {"engine_id": "mock", "model_id": "m"},
            "images": [{"path": "a.png", "sha256": "0" * 64, "width": 1, "height": 1, "format": "PNG"}],
            "seed_used": None, "runtime_s": 0, "attempts": [], "errors": [{"code": "x", "message": "m", "retryable": False}],
            "cost": {"usd_estimated": 0, "basis": "b"}, "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:00Z", "engine_metadata": {},
        }
        self.assertTrue(schemas.validation_errors(r))
        r["images"] = []
        self.assertFalse(schemas.validation_errors(r))
        r["errors"] = []
        self.assertTrue(schemas.validation_errors(r), "a failure must carry a structured error")

    def test_succeeded_result_needs_image(self):
        r = {
            "schema_version": "atelier.generation_result.v1", "result_id": "result_x", "request_id": "req_x",
            "candidate_id": "cand_x", "status": "succeeded", "engine": {"engine_id": "mock", "model_id": "m"},
            "images": [], "seed_used": 1, "runtime_s": 0, "attempts": [], "errors": [],
            "cost": {"usd_estimated": 0, "basis": "b"}, "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:00Z", "engine_metadata": {},
        }
        self.assertTrue(schemas.validation_errors(r))

    def test_curation_iterate_requires_plan_and_stop_requires_reason(self):
        cur = example("curation")
        cur["iteration_plan"] = None
        self.assertTrue(schemas.validation_errors(cur))
        cur["recommendation"] = "stop"
        self.assertTrue(schemas.validation_errors(cur), "stop needs stop_reason")
        cur["stop_reason"] = "good enough"
        self.assertFalse(schemas.validation_errors(cur))

    def test_brief_keeps_original_intent_separate_from_interpretation(self):
        schema = load(schemas.SCHEMA_DIR / "creative_brief.v1.json")
        self.assertIn("original_intent", schema["properties"])
        self.assertIn("system_interpretation", schema["properties"])
        b = brief()
        b["original_intent"]["summary"] = "system words"  # interpretation may not be smuggled into the intent
        self.assertTrue(schemas.validation_errors(b))

    def test_draft_schema_reuses_art_direction_sections(self):
        draft = load(schemas.SCHEMA_DIR / "art_direction_draft.v1.json")
        full = load(schemas.SCHEMA_DIR / "art_direction.v1.json")
        for section in ("subject", "composition", "palette", "mood", "semantic_constraints", "avoidance_constraints", "output"):
            self.assertEqual(draft["properties"][section]["$ref"], f"art_direction.v1.json#/$defs/{section}")
            self.assertEqual(full["properties"][section]["$ref"], f"#/$defs/{section}")
        self.assertNotIn("materials", draft["properties"])
        self.assertNotIn("lighting", draft["properties"])

    def test_config_rejects_bad_weights_and_limits(self):
        from atelier.config import ConfigError

        with self.assertRaises(ConfigError):
            config(weights={"intent_alignment": 0.5, "material_plausibility": 0.5, "composition": 0.5,
                            "visual_coherence": 0, "originality": 0, "technical_quality": 0})
        with self.assertRaises(ConfigError):
            config(batch_size=9)
        with self.assertRaises(ConfigError):
            config(max_iterations=6)
        cfg = config(batch_size=8)
        self.assertEqual(cfg["batch_size"], 8)


if __name__ == "__main__":
    unittest.main()
