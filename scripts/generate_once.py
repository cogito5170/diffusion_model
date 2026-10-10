"""Generate N real images from one brief + art direction with the configured engine (no agents, no loop).

Images and validated request/result records are kept under runs/<name>/ so you can look at them.

    python scripts/generate_once.py --config configs/live_sd_turbo_mac.json \
        --brief examples/briefs/memory_of_space.json \
        --art-direction examples/artifacts/memory_of_space.art_direction.json --count 2 --name mos-try1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atelier import schemas  # noqa: E402
from atelier.artifact_store import ArtifactStore  # noqa: E402
from atelier.budget import Budget, CancelToken  # noqa: E402
from atelier.config import load_config  # noqa: E402
from atelier.engines import create_engine  # noqa: E402
from atelier.engines.runner import execute_request  # noqa: E402
from atelier.evaluator import run_qa  # noqa: E402
from atelier.prompt_compiler import compile_request  # noqa: E402
from atelier.util import read_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--brief", required=True)
    ap.add_argument("--art-direction", required=True)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--name", required=True, help="output folder name under runs/")
    a = ap.parse_args()
    cfg = load_config(a.config)
    brief, ad = read_json(Path(a.brief)), read_json(Path(a.art_direction))
    schemas.validate(brief, "atelier.creative_brief.v1")
    schemas.validate(ad, "atelier.art_direction.v1")
    if not 1 <= a.count <= cfg["batch_size"]:
        raise SystemExit(f"--count must be 1..{cfg['batch_size']} (batch_size in config)")
    engine = create_engine(cfg["engine"])
    av = engine.availability()
    if not av.ok:
        raise SystemExit(f"engine unavailable: {av.reason}")
    caps, g = engine.capabilities(), cfg["generation"]
    store = ArtifactStore(Path("runs") / f"run_{a.name}")
    budget = Budget(dict(cfg["budget"], max_total_images=a.count), Budget.empty_ledger())
    for slot in range(a.count):
        req = compile_request(ad, brief, caps, run_id=f"run_{a.name}", iteration=0, slot=slot, base_seed=g["base_seed"],
                              keep_seeds=True, engine_params=cfg["engine_params"], timeout_s=g["timeout_s"],
                              max_retries=g["max_retries"])
        if slot == 0:
            print("prompt:", req["prompt"]["positive"])
            print("dropped:", [d["field"] for d in req["compilation"]["dropped"]])
            print(f"size {req['width']}x{req['height']}, params {req['engine_params']}")
        store.put(req)
        res = execute_request(engine, req, store, budget, CancelToken())
        store.put(res)
        line = f"[{slot}] {res['status']} in {res['runtime_s']:.1f}s"
        if res["status"] == "succeeded":
            qa = run_qa(res, req, store, cfg["qa"])
            store.put(qa)
            line += f"  QA {'passed' if qa['passed'] else 'FAILED'}  -> {store.abs(res['images'][0]['path'])}"
            line += f"  ({res['engine_metadata'].get('device')}, {res['engine_metadata'].get('dtype')})"
        else:
            line += "  " + "; ".join(f"{e['code']}: {e['message']}" for e in res["errors"])
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
