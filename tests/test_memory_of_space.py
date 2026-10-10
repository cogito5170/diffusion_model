"""The 'Memory of Space' brief/art direction stay valid and compile for the configured engine."""

import unittest

from atelier import schemas
from atelier.engines.diffusers_local import DiffusersLocalEngine
from atelier.prompt_compiler import compile_request
from atelier.spec_checks import check_art_direction
from tests.helpers import EXAMPLES, ROOT, load


class MemoryOfSpaceTests(unittest.TestCase):
    def setUp(self):
        self.brief = load(EXAMPLES / "briefs" / "memory_of_space.json")
        self.ad = load(EXAMPLES / "artifacts" / "memory_of_space.art_direction.json")

    def test_valid_and_checked(self):
        for name in ("art_direction_draft", "material_spec", "art_direction"):
            schemas.validate(load(EXAMPLES / "artifacts" / f"memory_of_space.{name}.json"))
        schemas.validate(self.brief)
        self.assertEqual(check_art_direction(self.ad, self.brief), ([], []))

    def test_original_intent_is_verbatim_source(self):
        src = (ROOT / "docs" / "prompts" / "memory_of_space.source.md").read_text().strip()
        self.assertEqual(self.brief["original_intent"]["text"], src)

    def test_compiles_for_sd_turbo_and_records_losses(self):
        req = compile_request(self.ad, self.brief, DiffusersLocalEngine().capabilities(), run_id="run_mos", iteration=0,
                              slot=0, base_seed=20261010, keep_seeds=True, engine_params={}, timeout_s=300, max_retries=0)
        self.assertEqual((req["width"], req["height"]), (768, 432))
        dropped = {d["field"] for d in req["compilation"]["dropped"]}
        self.assertIn("palette", dropped)
        self.assertTrue(any(f.startswith("reference_images") for f in dropped))
        self.assertNotIn("people", req["prompt"]["positive"])


if __name__ == "__main__":
    unittest.main()
