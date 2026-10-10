"""7. Candidate ranking and diversity preservation."""

import shutil
import unittest

from PIL import Image

from atelier import schemas
from atelier.engines.mock import render
from atelier.evaluator import weighted_score
from atelier.selector import CandidateView, distance, image_features, select
from tests.helpers import TempDirTest, config

CFG = config()
R = {"intent_alignment": 3, "material_plausibility": 3, "composition": 3, "visual_coherence": 3, "originality": 3,
     "technical_quality": 3}


def ratings(**kw):
    return dict(R, **kw)


class SelectorTests(TempDirTest):
    def img(self, name, seed, prompt="liquid chrome, graphite and chrome palette"):
        p = self.tmp / f"{name}.png"
        render(prompt, 128, 160, seed).save(p)
        return p

    def view(self, cid, path, r, qa=True, it=0):
        return CandidateView(cid, it, path, qa, r, "heuristic")

    def sel(self, views, **kw):
        cfg = dict(CFG["selection"], **kw)
        s = select(views, CFG["weights"], cfg, run_id="run_sel", iteration=0, requires_human_approval=True)
        schemas.validate(s, "atelier.selection.v1")
        return s

    def test_weighted_score_uses_configured_weights(self):
        self.assertEqual(weighted_score(ratings(intent_alignment=5), CFG["weights"]), round(3 + 2 * 0.25, 4))
        self.assertEqual(weighted_score(ratings(technical_quality=5), CFG["weights"]), round(3 + 2 * 0.10, 4))

    def test_ranking_order_and_ineligibility(self):
        a, b, c = self.img("a", 1), self.img("b", 2), self.img("c", 3)
        s = self.sel([
            self.view("cand_a", a, ratings(intent_alignment=4)),
            self.view("cand_b", b, ratings(intent_alignment=5)),
            self.view("cand_c", c, ratings(intent_alignment=5), qa=False),
            self.view("cand_d", None, None),
            self.view("cand_e", a, None),
        ])
        order = [r["candidate_id"] for r in s["ranking"]]
        self.assertEqual(order[:2], ["cand_b", "cand_a"])
        reasons = {r["candidate_id"]: r["ineligible_reason"] for r in s["ranking"]}
        self.assertEqual(reasons["cand_c"], "failed technical QA")
        self.assertIn("no image", reasons["cand_d"])
        self.assertEqual(reasons["cand_e"], "no critique")
        self.assertNotIn("cand_c", s["shortlist"])

    def test_near_duplicates_are_demoted_not_both_shortlisted(self):
        a = self.img("a", 1)
        dup = self.tmp / "dup.png"
        shutil.copy(a, dup)
        b = self.img("b", 99, prompt="brushed steel, cobalt and amber palette, monochrome")
        views = [self.view("cand_a", a, ratings(intent_alignment=5)),
                 self.view("cand_dup", dup, ratings(intent_alignment=5, composition=4)),
                 self.view("cand_b", b, ratings())]
        s = self.sel(views, shortlist_size=2)
        self.assertIn(["cand_a", "cand_dup"], s["diversity"]["near_duplicate_pairs"])
        self.assertEqual(len(s["shortlist"]), 2)
        self.assertFalse({"cand_a", "cand_dup"} <= set(s["shortlist"]), s["shortlist"])
        self.assertIn("cand_b", s["shortlist"], "a lower-scoring but different image is preserved")
        self.assertTrue(set(s["diversity"]["demoted_near_duplicates"]) & {"cand_a", "cand_dup"})

    def test_mmr_lambda_one_is_pure_score(self):
        imgs = [self.img(f"i{k}", k) for k in range(4)]
        views = [self.view(f"cand_{k}", p, ratings(composition=1 + k)) for k, p in enumerate(imgs)]
        s = self.sel(views, mmr_lambda=1.0, shortlist_size=3, near_duplicate_threshold=0.0)
        self.assertEqual(s["shortlist"], ["cand_3", "cand_2", "cand_1"])

    def test_low_lambda_prefers_diversity(self):
        a = self.img("a", 1)
        a2 = self.img("a2", 1, prompt="liquid chrome, graphite and chrome palette, soft focus")
        far = self.img("far", 50, prompt="oxidized steel, rust and amber palette")
        self.assertLess(distance(image_features(a), image_features(a2)), distance(image_features(a), image_features(far)))
        views = [self.view("cand_a", a, ratings(intent_alignment=5)),
                 self.view("cand_a2", a2, ratings(intent_alignment=5)),
                 self.view("cand_far", far, ratings(intent_alignment=4))]
        s = self.sel(views, mmr_lambda=0.2, shortlist_size=2, near_duplicate_threshold=0.0)
        self.assertEqual(s["shortlist"], ["cand_a", "cand_far"])

    def test_diversity_metrics_reported(self):
        views = [self.view(f"cand_{k}", self.img(f"i{k}", k), ratings()) for k in range(3)]
        s = self.sel(views)
        self.assertIsNotNone(s["diversity"]["mean_pairwise_distance"])
        self.assertTrue(s["requires_human_approval"])

    def test_features_ignore_file_encoding(self):
        a = self.img("a", 5)
        jpg = self.tmp / "a.jpg"
        Image.open(a).save(jpg, quality=95)
        self.assertLess(distance(image_features(a), image_features(jpg)), 0.05)


class HumanOverrideTests(TempDirTest):
    def test_human_can_override_shortlist_and_it_is_recorded(self):
        from tests.helpers import new_run

        orch = new_run(self.tmp)
        orch.run()
        self.assertEqual(orch.state["state"], "AWAITING_HUMAN_APPROVAL")
        sel = orch.store.get(orch._it()["selection_id"])
        all_ids = [c["candidate_id"] for _, c in orch._all_candidates()]
        outsider = next(c for c in all_ids if c not in sel["shortlist"])
        approval = orch.approve(outsider, "reviewer-1", "prefer the quieter frame")
        self.assertTrue(approval["override"]["is_override"])
        self.assertIn("not in system shortlist", approval["override"]["reasons"])
        orch.run()
        self.assertEqual(orch.state["state"], "DONE")
        self.assertEqual(orch.state["final_candidate_id"], outsider)

    def test_cannot_approve_candidate_without_image(self):
        from atelier.orchestrator import RunError
        from tests.helpers import new_run

        from atelier.engines.mock import MockEngine

        orch = new_run(self.tmp, engine=MockEngine(failures={"3": ["error"]}))
        orch.run()
        failed = orch.state["iterations"][0]["candidates"][3]["candidate_id"]
        with self.assertRaises(RunError):
            orch.approve(failed, "reviewer-1")
        with self.assertRaises(RunError):
            orch.approve("cand_nonexistent", "reviewer-1")


if __name__ == "__main__":
    unittest.main()
