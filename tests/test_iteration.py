"""8. Iteration stopping conditions and bounded-plan validation."""

import unittest

from atelier.iteration import apply_plan, major_sections, stopping_decision, validate_plan
from atelier.prompt_compiler import compile_request, engine_input
from atelier.engines.mock import MockEngine
from tests.helpers import TempDirTest, art_direction, brief, clone, config, example, new_run

STOP = {"target_weighted_score": 4.2, "min_dimension_for_target": 3, "min_improvement": 0.1}


def decide(**kw):
    args = dict(iteration=0, max_iterations=2, best_overall=3.5, best_overall_min_rating=3,
                best_before_this_iteration=None, best_this_iteration=3.5, stopping_cfg=STOP,
                curator_recommendation="iterate", curator_stop_reason=None, can_afford_next_batch=(True, None))
    args.update(kw)
    return stopping_decision(**args)


class StoppingTests(unittest.TestCase):
    def test_continue(self):
        self.assertIsNone(decide())

    def test_max_iterations(self):
        self.assertTrue(decide(iteration=2).startswith("max_iterations_reached"))

    def test_zero_iterations_means_single_batch(self):
        self.assertTrue(decide(iteration=0, max_iterations=0).startswith("max_iterations_reached"))

    def test_target_reached_requires_min_dimension(self):
        self.assertTrue(decide(best_overall=4.3).startswith("target_reached"))
        self.assertIsNone(decide(best_overall=4.3, best_overall_min_rating=2))

    def test_plateau(self):
        r = decide(iteration=1, best_before_this_iteration=3.6, best_this_iteration=3.65)
        self.assertTrue(r.startswith("plateau"), r)
        self.assertIsNone(decide(iteration=1, best_before_this_iteration=3.4, best_this_iteration=3.6))

    def test_curator_stop(self):
        self.assertTrue(decide(curator_recommendation="stop", curator_stop_reason="enough").startswith("curator"))

    def test_budget(self):
        self.assertTrue(decide(can_afford_next_batch=(False, "x")).startswith("budget_insufficient"))

    def test_no_eligible(self):
        self.assertTrue(decide(best_overall=None).startswith("no_eligible"))


def plan_for(ad, path, to, **kw):
    from atelier.iteration import pointer_get

    p = clone(example("iteration_plan"))
    p["base_art_direction_id"] = ad["art_direction_id"]
    p["what_will_change"] = [{"path": path, "from": pointer_get(ad, path), "to": to, "rationale": "test"}]
    p.update(kw)
    return p


class PlanValidationTests(unittest.TestCase):
    def setUp(self):
        self.ad = art_direction()
        caps = MockEngine().capabilities()

        def ei(ad):
            return engine_input(compile_request(ad, brief(), caps, run_id="run_p", iteration=0, slot=0, base_seed=1,
                                                keep_seeds=True, engine_params={}, timeout_s=10, max_retries=0))

        self.ei = ei

    def check(self, plan, **kw):
        args = dict(current_iteration=0, selection_id="sel_example-i0", max_major_variables=1, max_iterations=2,
                    brief=brief(), engine_input_fn=self.ei)
        args.update(kw)
        return validate_plan(plan, self.ad, **args)

    def test_example_plan_valid(self):
        self.assertEqual(self.check(example("iteration_plan")), [])

    def test_too_many_major_variables(self):
        p = plan_for(self.ad, "/lighting/contrast_ratio", "4:1")
        p["what_will_change"].append({"path": "/materials/0/roughness", "from": 0.04, "to": 0.1, "rationale": "x"})
        self.assertEqual(major_sections(p), ["lighting", "materials"])
        errs = self.check(p)
        self.assertTrue(any("major variables" in e for e in errs), errs)
        self.assertEqual(self.check(p, max_major_variables=2), [])

    def test_several_paths_in_one_section_count_once(self):
        p = plan_for(self.ad, "/lighting/contrast_ratio", "4:1")
        p["what_will_change"].append({"path": "/lighting/key/hardness", "from": "hard", "to": "medium", "rationale": "x"})
        self.assertEqual(self.check(p), [])

    def test_frozen_sections_rejected(self):
        for path, to in (("/concept_thesis", "another concept entirely, different"), ("/output/width", 640)):
            errs = self.check(plan_for(self.ad, path, to))
            self.assertTrue(any("may not be changed" in e for e in errs), (path, errs))

    def test_stale_from_value_rejected(self):
        p = plan_for(self.ad, "/lighting/contrast_ratio", "4:1")
        p["what_will_change"][0]["from"] = "2:1"
        self.assertTrue(any("stale" in e for e in self.check(p)))

    def test_noop_change_rejected(self):
        p = plan_for(self.ad, "/lighting/contrast_ratio", "8:1")
        self.assertTrue(any("no change" in e for e in self.check(p)))

    def test_nonexistent_path_rejected(self):
        p = plan_for(self.ad, "/lighting/contrast_ratio", "4:1")
        p["what_will_change"][0]["path"] = "/lighting/haze"
        self.assertTrue(any("does not exist" in e for e in self.check(p)))

    def test_change_that_never_reaches_engine_rejected(self):
        p = plan_for(self.ad, "/composition/focal_hierarchy", ["floor seam"])
        errs = self.check(p)
        self.assertTrue(any("does not change the compiled engine input" in e for e in errs), errs)

    def test_resulting_spec_must_stay_valid(self):
        p = plan_for(self.ad, "/materials/0/roughness", 0.9)  # chrome at roughness 0.9 is implausible
        self.assertTrue(any("implausible" in e for e in self.check(p)))
        p = plan_for(self.ad, "/materials/0/roughness", 1.7)
        self.assertTrue(any("invalid" in e for e in self.check(p)))

    def test_iteration_bound(self):
        p = example("iteration_plan")
        p["iteration"] = 3
        errs = self.check(p, current_iteration=2, max_iterations=2)
        self.assertTrue(any("beyond max_iterations" in e for e in errs), errs)

    def test_unattainable_criterion_rejected(self):
        p = example("iteration_plan")
        p["improvement_criterion"] = {"metric": "composition", "aggregate": "mean", "comparator": ">", "threshold": 5}
        self.assertTrue(any("never be met" in e for e in self.check(p)))

    def test_apply_plan_versions_and_provenance(self):
        new = apply_plan(example("iteration_plan"), self.ad)
        self.assertEqual(new["version"], 2)
        self.assertEqual(new["art_direction_id"], "ad_example-v2")
        self.assertEqual(new["parent_id"], "ad_example-v1")
        self.assertEqual(new["provenance"]["derived_from_plan_id"], "plan_example-i1")
        self.assertEqual(new["materials"], self.ad["materials"], "untouched sections stay fixed")
        self.assertNotEqual(new["subject"], self.ad["subject"])


class LoopBoundTests(TempDirTest):
    def test_loop_never_exceeds_max_iterations(self):
        for max_it in (0, 1, 2):
            with self.subTest(max_iterations=max_it):
                cfg = config(max_iterations=max_it,
                             stopping={"target_weighted_score": 5.0, "min_dimension_for_target": 5, "min_improvement": 0.0},
                             selection={"shortlist_size": 3, "mmr_lambda": 0.7, "near_duplicate_threshold": 0.08})
                orch = new_run(self.tmp, cfg, run_id=f"run_bound{max_it}")
                orch.run()
                self.assertLessEqual(len(orch.state["iterations"]), max_it + 1)
                self.assertLessEqual(orch.state["ledger"]["images"], 4 * (max_it + 1))
                self.assertEqual(orch.state["state"], "AWAITING_HUMAN_APPROVAL")

    def test_every_executed_iteration_has_five_part_plan(self):
        cfg = config(stopping={"target_weighted_score": 5.0, "min_dimension_for_target": 5, "min_improvement": 0.0})
        orch = new_run(self.tmp, cfg)
        orch.run()
        for it in orch.state["iterations"][1:]:
            ad = orch.store.get(it["art_direction_id"])
            plan = orch.store.get(ad["provenance"]["derived_from_plan_id"])
            for key in ("what_failed", "what_will_change", "what_remains_fixed", "why_it_should_help", "improvement_criterion"):
                self.assertTrue(plan[key], key)
            self.assertIsNotNone(it["plan_outcome"], "each refinement batch is evaluated against its criterion")


if __name__ == "__main__":
    unittest.main()
