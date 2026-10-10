"""3. Prompt compilation and capability negotiation."""

import dataclasses
import unittest

from atelier import schemas
from atelier.engines.base import ParamSpec
from atelier.engines.mock import MockEngine
from atelier.prompt_compiler import (CompilationError, compile_baseline_request, compile_request, engine_input,
                                     estimate_tokens, negotiate_size)
from tests.helpers import art_direction, brief, clone


def caps(**changes):
    return dataclasses.replace(MockEngine().capabilities(), **changes)


def compile_(ad=None, b=None, c=None, **kw):
    args = dict(run_id="run_t", iteration=0, slot=0, base_seed=100, keep_seeds=True, engine_params={},
                timeout_s=30, max_retries=1)
    args.update(kw)
    return compile_request(ad or art_direction(), b or brief(), c or caps(), **args)


class CompilerTests(unittest.TestCase):
    def test_output_is_schema_valid_and_deterministic(self):
        r1, r2 = compile_(), compile_()
        schemas.validate(r1, "atelier.generation_request.v1")
        self.assertEqual(r1, r2)

    def test_respects_token_budget_and_records_drops(self):
        r = compile_()
        self.assertLessEqual(estimate_tokens(r["prompt"]["positive"]), 75)
        dropped = {d["field"] for d in r["compilation"]["dropped"]}
        self.assertTrue(dropped, "the example spec is larger than the CLIP budget; something must be dropped")
        self.assertIn("prompt_segments_dropped", {w["code"] for w in r["compilation"]["warnings"]})

    def test_priority_keeps_subject_and_primary_material(self):
        r = compile_(c=caps(max_prompt_tokens=30))
        self.assertIn("fine-art editorial photograph", r["prompt"]["positive"])
        self.assertIn("liquid chrome", r["prompt"]["positive"])
        self.assertLessEqual(estimate_tokens(r["prompt"]["positive"]), 30)

    def test_unlimited_budget_includes_everything(self):
        r = compile_(c=caps(max_prompt_tokens=None))
        p = r["prompt"]["positive"]
        for frag in ("graphite", "85mm lens", "austere", "brushed steel", "industrial hall"):
            self.assertIn(frag, p)

    def test_avoidance_terms_never_in_positive_prompt(self):
        r = compile_(c=caps(max_prompt_tokens=None))
        for term in ("robot", "science fiction", "cartoon"):
            self.assertNotIn(term, r["prompt"]["positive"].lower())

    def test_negative_prompt_dropped_when_cfg_disabled(self):
        r = compile_()  # sd-turbo profile: guidance 0 -> negatives have no effect
        self.assertIsNone(r["prompt"]["negative"])
        self.assertIn("negative_prompt_unavailable", {w["code"] for w in r["compilation"]["warnings"]})

    def test_negative_prompt_used_when_effective(self):
        c = caps(params={"num_inference_steps": ParamSpec("int", 25, 5, 50),
                         "guidance_scale": ParamSpec("float", 7.0, 1.0, 15.0)})
        r = compile_(c=c)
        self.assertIn("robot", r["prompt"]["negative"])
        self.assertIn("watermark", r["prompt"]["negative"])

    def test_negative_prompt_dropped_when_unsupported(self):
        r = compile_(c=caps(supports_negative_prompt=False))
        self.assertIsNone(r["prompt"]["negative"])
        self.assertIn("prompt.negative", {d["field"] for d in r["compilation"]["dropped"]})

    def test_unknown_engine_param_rejected(self):
        with self.assertRaises(CompilationError):
            compile_(engine_params={"cfg_rescale": 0.7})

    def test_out_of_range_param_rejected_not_clamped(self):
        with self.assertRaises(Exception) as ctx:
            compile_(engine_params={"num_inference_steps": 50})  # turbo profile allows 1-4
        self.assertIn("above maximum", str(ctx.exception))

    def test_wrong_param_type_rejected(self):
        with self.assertRaises(Exception):
            compile_(engine_params={"num_inference_steps": 2.5})
        with self.assertRaises(Exception):
            compile_(engine_params={"num_inference_steps": True})

    def test_size_negotiation(self):
        c = caps()
        self.assertEqual(negotiate_size(512, 640, c), (512, 640, False))
        w, h, adj = negotiate_size(1600, 2000, c)
        self.assertTrue(adj)
        self.assertLessEqual(max(w, h), c.max_side)
        self.assertEqual((w % 8, h % 8), (0, 0))
        self.assertAlmostEqual(w / h, 0.8, delta=0.02)
        ad = art_direction()
        ad["output"].update(width=1600, height=2000)
        r = compile_(ad)
        self.assertTrue(r["compilation"]["size_adjusted"])
        self.assertIn("size_adjusted", {x["code"] for x in r["compilation"]["warnings"]})

    def test_size_negotiation_never_breaks_limits(self):
        import itertools

        from atelier.engines.diffusers_local import DiffusersLocalEngine

        for c in (caps(), DiffusersLocalEngine(model_id="stabilityai/stable-diffusion-xl-base-1.0").capabilities()):
            for w, h in itertools.product(range(64, 2049, 56), repeat=2):
                try:
                    nw, nh, _ = negotiate_size(w, h, c)
                except CompilationError:
                    self.assertGreater(max(w, h) / min(w, h), 0.97 * c.max_side / c.min_side, (w, h))
                    continue
                self.assertLessEqual(nw * nh, c.max_pixels, (w, h))
                self.assertLessEqual(max(nw, nh), c.max_side, (w, h))
                self.assertGreaterEqual(min(nw, nh), c.min_side, (w, h))
                self.assertEqual((nw % 8, nh % 8), (0, 0))

    def test_required_reference_image_unsupported_is_error(self):
        b = brief()
        b["constraints"]["reference_images"] = [{"path": "ref.png", "role": "composition", "required": True}]
        with self.assertRaises(CompilationError):
            compile_(b=b)

    def test_optional_reference_image_dropped_with_warning(self):
        b = brief()
        b["constraints"]["reference_images"] = [{"path": "ref.png", "role": "mood", "required": False}]
        r = compile_(b=b)
        self.assertEqual(r["reference_images"], [])
        self.assertIn("reference_images_dropped", {w["code"] for w in r["compilation"]["warnings"]})

    def test_reference_image_kept_when_supported(self):
        b = brief()
        b["constraints"]["reference_images"] = [{"path": "missing.png", "role": "mood", "required": True}]
        r = compile_(b=b, c=caps(max_reference_images=1))
        self.assertEqual(r["reference_images"], [{"path": "missing.png", "role": "mood", "sha256": None}])

    def test_seeds_fixed_across_iterations_unless_requested(self):
        a = compile_(iteration=0, slot=2)
        b = compile_(iteration=1, slot=2)
        c = compile_(iteration=1, slot=2, keep_seeds=False)
        self.assertEqual(a["seed"], b["seed"])
        self.assertNotEqual(a["seed"], c["seed"])
        self.assertNotEqual(compile_(slot=0)["seed"], compile_(slot=1)["seed"])

    def test_seed_unsupported(self):
        r = compile_(c=caps(supports_seed=False))
        self.assertIsNone(r["seed"])

    def test_spec_change_reaches_engine_input(self):
        ad = art_direction()
        ad2 = clone(ad)
        ad2["materials"][0]["roughness"] = 0.25
        self.assertNotEqual(engine_input(compile_(ad)), engine_input(compile_(ad2)))
        ad3 = clone(ad)
        ad3["composition"]["focal_hierarchy"] = ["floor seam"]  # used by critique, never by the compiler
        self.assertEqual(engine_input(compile_(ad)), engine_input(compile_(ad3)))

    def test_subject_clause_cut_at_comma_not_mid_phrase(self):
        from atelier.prompt_compiler import _short

        text = ("one low bent brushed stainless steel arch sculpture alone on concrete under a luminous ceiling, "
                "formed from a single continuous sheet folded once")
        clause, shortened = _short(text, 16)
        self.assertTrue(shortened)
        self.assertEqual(clause, "one low bent brushed stainless steel arch sculpture alone on concrete under a luminous ceiling")

    def test_lighting_outranks_surface_imperfections(self):
        ad = art_direction()
        ad["materials"][0]["imperfections"] = ["faint longitudinal grain variation along the whole sheet",
                                               "slightly softer burnished patch on the crown of the fold"]
        r = compile_(ad)
        self.assertIn("key light", r["prompt"]["positive"])
        fields = [d["field"] for d in r["compilation"]["dropped"]]
        self.assertNotIn("lighting", fields)

    def test_trimmed_segment_keeps_leading_parts_and_is_recorded(self):
        ad = art_direction()
        ad["lighting"]["key"]["direction"] = "directly overhead from the full ceiling plane, broad and even, " * 2
        r = compile_(ad, c=caps(max_prompt_tokens=60))
        self.assertLessEqual(estimate_tokens(r["prompt"]["positive"]), 60)
        fields = [d["field"] for d in r["compilation"]["dropped"]]
        self.assertTrue("lighting (tail)" in fields or "lighting" in fields)

    def test_material_names_its_object(self):
        from atelier.prompt_compiler import describe_material

        m = dict(art_direction()["materials"][0], material_id="tool_head", type="brushed_steel", finish="dust-dulled")
        self.assertTrue(describe_material(m).startswith("brushed steel tool head (dust-dulled)"))
        m["material_id"] = "primary"
        self.assertTrue(describe_material(m).startswith("brushed steel (dust-dulled)"))

    def test_baseline_request(self):
        r = compile_baseline_request(brief()["original_intent"]["text"], caps(), run_id="run_t", slot=1, width=512,
                                     height=640, base_seed=100, engine_params={}, timeout_s=10, max_retries=0)
        schemas.validate(r)
        self.assertEqual(r["prompt_source"], "baseline_raw")
        self.assertIsNone(r["art_direction_id"])
        self.assertLessEqual(estimate_tokens(r["prompt"]["positive"]), 75)
        self.assertEqual(r["seed"], compile_(slot=1)["seed"], "baseline uses the same seeds as structured runs")


if __name__ == "__main__":
    unittest.main()
