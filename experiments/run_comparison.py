"""Reproducible A/B/C comparison.

  A  baseline: the user's raw intent text as the only prompt (no art direction)
  B  structured: run C's iteration-0 art direction + deterministic prompt compiler, no critique/refinement
  C  full workflow: critic + curator + bounded refinement (the orchestrator run)

B reuses C's v1 art direction, so B vs C isolates the refinement loop and A vs B isolates
structured art direction. A and B use the same engine and the same per-slot seeds. By
default every condition gets the same number of images (the number C actually generated).

Usage:
  python experiments/run_comparison.py run     --brief B.json --config C.json --out experiments/out/NAME
  python experiments/run_comparison.py analyze --out experiments/out/NAME      # after filling blind/ratings.csv

Nothing here claims that any condition is better. `analyze` reports measured differences
in blinded human ratings with bootstrap intervals, and labels every other number as
measured, estimated or proxy.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atelier.artifact_store import ArtifactStore  # noqa: E402
from atelier.budget import Budget, CancelToken  # noqa: E402
from atelier.config import load_config  # noqa: E402
from atelier.engines import create_engine  # noqa: E402
from atelier.engines.runner import execute_request  # noqa: E402
from atelier.offline_agents import heuristic_ratings  # noqa: E402
from atelier.orchestrator import Orchestrator  # noqa: E402
from atelier.prompt_compiler import compile_baseline_request, compile_request  # noqa: E402
from atelier.selector import distance, image_features  # noqa: E402
from atelier.util import atomic_write_json, read_json  # noqa: E402

RATING_COLUMNS = ["blind_id", "file", "overall_1to5", "intent_alignment_1to5", "material_plausibility_1to5", "notes"]


def _generate(engine, requests, store, budget):
    out = []
    for req in requests:
        store.put(req)
        res = execute_request(engine, req, store, budget, CancelToken())
        store.put(res)
        out.append(res)
    return out


def cmd_run(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    brief, cfg = read_json(Path(a.brief)), load_config(a.config)
    stem = out.name.replace("_", "-")
    run_id = f"run_{stem}-c"
    runs = out / "runs"
    if (runs / run_id / "state.json").exists():
        orch = Orchestrator.load(runs / run_id)
    else:
        orch = Orchestrator.create(runs, run_id, brief, cfg)
    orch.run()
    if orch.state["state"] not in ("AWAITING_HUMAN_APPROVAL", "DONE", "APPROVED"):
        print(f"condition C is not finished (state {orch.state['state']}); waiting_for={orch.state['waiting_for']}")
        print("complete the pending agent tasks, then re-run this command (it resumes).")
        return 2

    c_rows = []
    for itn, cand in orch._all_candidates():
        res = orch.store.get(cand["result_id"]) if cand["result_id"] else None
        if res:
            c_rows.append(("C", res, orch.store))
    n_images = a.images_per_condition or sum(1 for _, r, _ in c_rows if r["status"] == "succeeded")
    ad0 = orch.store.get(orch.state["iterations"][0]["art_direction_id"])
    engine = create_engine(cfg["engine"])
    caps = engine.capabilities()
    g = cfg["generation"]
    limits = dict(cfg["budget"], max_total_images=n_images, max_generation_attempts=n_images * (1 + g["max_retries"]))

    rows = list(c_rows)
    for cond in ("A", "B"):
        store = ArtifactStore(out / f"condition_{cond}")
        existing = [store.get(i) for i in store.ids("generation_result")]
        if not existing:
            reqs = []
            for slot in range(n_images):
                if cond == "A":
                    reqs.append(compile_baseline_request(
                        brief["original_intent"]["text"], caps, run_id=f"run_{stem}-a", slot=slot,
                        width=ad0["output"]["width"], height=ad0["output"]["height"], base_seed=g["base_seed"],
                        engine_params=cfg["engine_params"], timeout_s=g["timeout_s"], max_retries=g["max_retries"]))
                else:
                    reqs.append(compile_request(
                        ad0, brief, caps, run_id=f"run_{stem}-b", iteration=0, slot=slot, base_seed=g["base_seed"],
                        keep_seeds=True, engine_params=cfg["engine_params"], timeout_s=g["timeout_s"],
                        max_retries=g["max_retries"]))
            existing = _generate(engine, reqs, store, Budget(limits, Budget.empty_ledger()))
        rows += [(cond, r, store) for r in existing]

    summary = summarize(rows, ad0)
    summary["design"] = {
        "images_per_condition_target": n_images,
        "engine": {"engine_id": caps.engine_id, "model_id": caps.model_id},
        "c_run": str(runs / run_id),
        "b_art_direction_id": ad0["art_direction_id"],
        "critic_kinds_in_C": sorted({orch.store.get(c["critique_id"])["critic"]["kind"]
                                     for _, c in orch._all_candidates() if c["critique_id"]}),
    }
    atomic_write_json(out / "automatic_metrics.json", summary)
    make_blind_packet(rows, out / "blind", seed=a.seed)
    print(json.dumps(summary, indent=2))
    print(f"\nblind rating sheet: {out / 'blind' / 'ratings.csv'}  (key in blind/key.json - do not open before rating)")
    return 0


def summarize(rows, ad) -> dict:
    by = {}
    for cond, res, store in rows:
        by.setdefault(cond, []).append((res, store))
    out = {}
    for cond, items in sorted(by.items()):
        ok = [(r, s) for r, s in items if r["status"] == "succeeded"]
        paths = [s.abs(r["images"][0]["path"]) for r, s in ok]
        feats = {str(p): image_features(p) for p in paths}
        pairs = list(combinations(list(feats), 2))
        proxies = [heuristic_ratings(p, ad, {"passed": True}, [q for q in paths if q != p])[0] for p in paths]
        out[cond] = {
            "n_requested": len(items),
            "n_succeeded": len(ok),
            "success_rate": round(len(ok) / len(items), 4) if items else None,
            "runtime_s_per_candidate_mean__measured": round(float(np.mean([r["runtime_s"] for r, _ in items])), 4) if items else None,
            "cost_usd_per_candidate__estimated": round(float(np.mean([r["cost"]["usd_estimated"] for r, _ in items])), 6) if items else None,
            "cost_basis": items[0][0]["cost"]["basis"] if items else None,
            "diversity_mean_pairwise_feature_distance__measured_proxy":
                round(float(np.mean([distance(feats[a], feats[b]) for a, b in pairs])), 4) if pairs else None,
            "heuristic_intent_alignment_mean__proxy": round(float(np.mean([p["intent_alignment"] for p in proxies])), 3) if proxies else None,
            "heuristic_material_plausibility_mean__proxy": round(float(np.mean([p["material_plausibility"] for p in proxies])), 3) if proxies else None,
        }
    out["_labels"] = {
        "measured": "observed in this run (wall-clock, counts, feature distances)",
        "estimated": "derived from the engine's declared cost model, not an invoice",
        "proxy": "heuristic pixel statistics; not evidence of aesthetic quality",
    }
    return out


def make_blind_packet(rows, blind: Path, seed: int) -> None:
    if (blind / "ratings.csv").exists():
        return  # never overwrite ratings in progress
    (blind / "images").mkdir(parents=True, exist_ok=True)
    items = [(cond, res, store) for cond, res, store in rows if res["status"] == "succeeded"]
    rnd = random.Random(seed)
    rnd.shuffle(items)
    key = {}
    with open(blind / "ratings.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(RATING_COLUMNS)
        for i, (cond, res, store) in enumerate(items, start=1):
            bid = f"img{i:03d}"
            src = store.abs(res["images"][0]["path"])
            shutil.copy(src, blind / "images" / f"{bid}.png")
            key[bid] = {"condition": cond, "candidate_id": res["candidate_id"], "sha256": res["images"][0]["sha256"]}
            w.writerow([bid, f"images/{bid}.png", "", "", "", ""])
    atomic_write_json(blind / "key.json", key)


def _bootstrap_diff(x, y, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    diffs = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(n)]
    return float(x.mean() - y.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def cmd_analyze(a) -> int:
    out = Path(a.out)
    key = read_json(out / "blind" / "key.json")
    ratings: dict[str, dict[str, list[float]]] = {}
    with open(out / "blind" / "ratings.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cond = key[row["blind_id"]]["condition"]
            for col in ("overall_1to5", "intent_alignment_1to5", "material_plausibility_1to5"):
                if row[col].strip():
                    v = float(row[col])
                    if not 1 <= v <= 5:
                        raise SystemExit(f"{row['blind_id']} {col}={v} is outside 1-5")
                    ratings.setdefault(col, {}).setdefault(cond, []).append(v)
    if not ratings:
        print("no ratings filled in yet; nothing to analyze (no claims made)")
        return 1
    result = {"metrics": {}, "comparisons": {}}
    for col, by in ratings.items():
        result["metrics"][col] = {c: {"n": len(v), "mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v, ddof=1)), 3) if len(v) > 1 else None}
                                  for c, v in sorted(by.items())}
        for hi, lo in (("B", "A"), ("C", "B"), ("C", "A")):
            if len(by.get(hi, [])) >= 2 and len(by.get(lo, [])) >= 2:
                d, l, u = _bootstrap_diff(by[hi], by[lo])
                result["comparisons"][f"{col}:{hi}-{lo}"] = {
                    "mean_diff": round(d, 3), "ci95": [round(l, 3), round(u, 3)],
                    "finding": "measured difference (CI excludes 0)" if l > 0 or u < 0 else "no supported difference (CI includes 0)",
                }
    result["caveats"] = ["single rater unless ratings.csv aggregates several raters",
                         "images within a condition share seeds/prompts and are not independent samples",
                         "bootstrap CI over images, not over briefs; one brief does not generalize"]
    atomic_write_json(out / "human_results.json", result)
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--brief", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--images-per-condition", type=int, default=0, help="default: number of images C generated")
    r.add_argument("--seed", type=int, default=7, help="shuffle seed for the blind packet")
    an = sub.add_parser("analyze")
    an.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_analyze(a)


if __name__ == "__main__":
    raise SystemExit(main())
