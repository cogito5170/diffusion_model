"""Candidate ranking with diversity preservation.

The shortlist is built with Maximal Marginal Relevance (MMR) over cheap perceptual
features: a 16x16 luminance thumbnail plus a hue/saturation histogram. It trades
weighted score against similarity to candidates already picked. Near-duplicates are
demoted, not deleted. The final choice is left to a human by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from atelier.evaluator import weighted_score

FEATURE_DESCRIPTION = ("16x16 zero-mean luma thumbnail (L2, weight 0.7) + 8x3 hue/sat histogram of chromatic pixels "
                       "and 4-bin luma histogram of grey pixels (L2, weight 0.3); distance = L2 / sqrt(2)")


@dataclass
class CandidateView:
    candidate_id: str
    iteration: int
    image_path: Path | None
    qa_passed: bool | None
    ratings: dict[str, int] | None
    critic_kind: str | None


def image_features(path: Path) -> np.ndarray:
    st = Path(path).stat()
    return _features_cached(str(Path(path).resolve()), st.st_size, st.st_mtime_ns).copy()


@lru_cache(maxsize=512)
def _features_cached(path: str, size: int, mtime_ns: int) -> np.ndarray:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        thumb = np.asarray(rgb.convert("L").resize((16, 16), Image.BILINEAR), dtype=np.float32).ravel()
        hsv = np.asarray(rgb.resize((64, 64), Image.BILINEAR).convert("HSV"), dtype=np.float32) / 255.0
    thumb = thumb - thumb.mean()
    thumb /= (np.linalg.norm(thumb) or 1.0)
    h, s, v = (hsv[..., i].ravel() for i in range(3))
    chromatic = s >= 0.15  # hue is meaningless (and compression-noisy) for near-grey pixels
    hue_hist, _, _ = np.histogram2d(h[chromatic], s[chromatic], bins=(8, 3), range=((0, 1), (0.15, 1)))
    grey_hist, _ = np.histogram(v[~chromatic], bins=4, range=(0, 1))
    hist = np.concatenate([hue_hist.ravel(), grey_hist]).astype(np.float32)
    hist /= (np.linalg.norm(hist) or 1.0)
    return np.concatenate([thumb * np.sqrt(0.7), hist * np.sqrt(0.3)])


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.sqrt(2.0))


def mean_pairwise_distance(features: dict[str, np.ndarray], ids: list[str] | None = None) -> float | None:
    keys = ids if ids is not None else list(features)
    pairs = list(combinations(keys, 2))
    if not pairs:
        return None
    return round(sum(distance(features[a], features[b]) for a, b in pairs) / len(pairs), 6)


def select(
    views: list[CandidateView],
    weights: dict[str, float],
    selection_cfg: dict[str, Any],
    *,
    run_id: str,
    iteration: int,
    requires_human_approval: bool,
    features: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    ranking: list[dict[str, Any]] = []
    eligible: dict[str, float] = {}
    for v in views:
        reason = None
        if v.image_path is None:
            reason = "no image (generation did not succeed)"
        elif not v.qa_passed:
            reason = "failed technical QA"
        elif v.ratings is None:
            reason = "no critique"
        score = weighted_score(v.ratings, weights) if v.ratings is not None else None
        if reason is None:
            eligible[v.candidate_id] = score
        ranking.append({"candidate_id": v.candidate_id, "iteration": v.iteration, "eligible": reason is None,
                        "ineligible_reason": reason, "weighted_score": score, "score_rank": None,
                        "critic_kind": v.critic_kind})

    order = sorted(eligible, key=lambda c: (-eligible[c], c))
    for rank, cid in enumerate(order, start=1):
        next(r for r in ranking if r["candidate_id"] == cid)["score_rank"] = rank
    ranking.sort(key=lambda r: (r["score_rank"] is None, r["score_rank"] or 0, r["candidate_id"]))

    paths = {v.candidate_id: v.image_path for v in views if v.candidate_id in eligible}
    feats = dict(features or {})
    for cid, p in paths.items():
        if cid not in feats:
            feats[cid] = image_features(p)

    thr = selection_cfg["near_duplicate_threshold"]
    lam = selection_cfg["mmr_lambda"]
    k = selection_cfg["shortlist_size"]
    near_dupes = [[a, b] for a, b in combinations(sorted(eligible), 2) if distance(feats[a], feats[b]) < thr]

    shortlist: list[str] = []
    demoted: list[str] = []
    remaining = list(order)
    while remaining and len(shortlist) < k:
        def mmr(cid: str) -> tuple[float, str]:
            rel = (eligible[cid] - 1.0) / 4.0
            sim = max((1.0 - distance(feats[cid], feats[s]) for s in shortlist), default=0.0)
            return (lam * rel - (1 - lam) * sim, cid)

        fresh = [c for c in remaining if all(distance(feats[c], feats[s]) >= thr for s in shortlist)]
        pool = fresh or remaining
        if not fresh:
            demoted.extend(c for c in remaining if c not in demoted)
            break  # every remaining candidate is a near-duplicate of the shortlist; keep it short
        best = max(pool, key=lambda c: (mmr(c)[0], -order.index(c)))
        shortlist.append(best)
        remaining.remove(best)
    for c in remaining:
        if any(distance(feats[c], feats[s]) < thr for s in shortlist) and c not in demoted:
            demoted.append(c)

    return {
        "schema_version": "atelier.selection.v1",
        "selection_id": f"sel_{run_id.split('_', 1)[1]}-i{iteration}",
        "run_id": run_id,
        "iteration": iteration,
        "weights": dict(weights),
        "method": {"name": "weighted_score_mmr", "mmr_lambda": lam, "shortlist_size": k,
                   "near_duplicate_threshold": thr, "feature": FEATURE_DESCRIPTION},
        "ranking": ranking,
        "shortlist": shortlist,
        "diversity": {
            "mean_pairwise_distance": mean_pairwise_distance(feats, sorted(eligible)),
            "shortlist_mean_pairwise_distance": mean_pairwise_distance(feats, shortlist),
            "near_duplicate_pairs": near_dupes,
            "demoted_near_duplicates": sorted(demoted),
        },
        "requires_human_approval": requires_human_approval,
    }
