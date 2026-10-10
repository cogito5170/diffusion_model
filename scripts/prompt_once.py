"""Generate images straight from a prompt string (no art direction, no agents) and optionally crop them to a
layout slot such as 574x172.

The prompt is passed through the same engine runner, budget, QA and records as a pipeline run
(prompt_source = "baseline_raw"). Prompts longer than the engine's token budget are truncated and the drop is
recorded. Cropping keeps the vertical centre band, so compose the image around a horizontal middle band.

    python scripts/prompt_once.py --config configs/live_sd_turbo_mac.json --name slot-try1 --count 4 \
        --crop 574x172 --prompt "cinematic night photograph, ..."
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atelier.artifact_store import ArtifactStore  # noqa: E402
from atelier.budget import Budget, CancelToken  # noqa: E402
from atelier.config import load_config  # noqa: E402
from atelier.engines import create_engine  # noqa: E402
from atelier.engines.runner import execute_request  # noqa: E402
from atelier.evaluator import run_qa  # noqa: E402
from atelier.prompt_compiler import compile_baseline_request  # noqa: E402


def crop_to_slot(src: Path, slot_w: int, slot_h: int) -> Path:
    """Centre-crop to the slot's aspect ratio, then resize to the slot size (never upscales silently)."""
    with Image.open(src) as im:
        w, h = im.size
        target = slot_w / slot_h
        if w / h > target:
            cw, ch = round(h * target), h
        else:
            cw, ch = w, round(w / target)
        left, top = (w - cw) // 2, (h - ch) // 2
        band = im.crop((left, top, left + cw, top + ch))
        if cw < slot_w:
            print(f"  note: crop is {cw}x{ch}, smaller than the slot; it is upscaled to {slot_w}x{slot_h}")
        out = src.with_name(f"slot_{slot_w}x{slot_h}.png")
        band.resize((slot_w, slot_h), Image.LANCZOS).save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--name", required=True, help="output folder name under runs/")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=432)
    ap.add_argument("--crop", help="layout slot WxH, e.g. 574x172")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if not 1 <= a.count <= cfg["batch_size"]:
        raise SystemExit(f"--count must be 1..{cfg['batch_size']} (batch_size in config)")
    slot = tuple(int(v) for v in a.crop.lower().split("x")) if a.crop else None
    engine = create_engine(cfg["engine"])
    av = engine.availability()
    if not av.ok:
        raise SystemExit(f"engine unavailable: {av.reason}")
    caps, g = engine.capabilities(), cfg["generation"]
    store = ArtifactStore(Path("runs") / f"run_{a.name}")
    budget = Budget(dict(cfg["budget"], max_total_images=a.count), Budget.empty_ledger())
    for slot_i in range(a.count):
        req = compile_baseline_request(a.prompt, caps, run_id=f"run_{a.name}", slot=slot_i, width=a.width,
                                       height=a.height, base_seed=g["base_seed"], engine_params=cfg["engine_params"],
                                       timeout_s=g["timeout_s"], max_retries=g["max_retries"])
        if slot_i == 0:
            print("prompt sent:", req["prompt"]["positive"])
            if req["compilation"]["warnings"]:
                print("warnings:", [w["code"] for w in req["compilation"]["warnings"]])
            print(f"size {req['width']}x{req['height']}, params {req['engine_params']}")
        store.put(req)
        res = execute_request(engine, req, store, budget, CancelToken())
        store.put(res)
        line = f"[{slot_i}] seed {req['seed']} {res['status']} in {res['runtime_s']:.1f}s"
        if res["status"] == "succeeded":
            qa = run_qa(res, req, store, cfg["qa"])
            store.put(qa)
            img = store.abs(res["images"][0]["path"])
            line += f"  QA {'passed' if qa['passed'] else 'FAILED'}  -> {img}"
            if slot:
                line += f"\n     slot crop -> {crop_to_slot(img, *slot)}"
        else:
            line += "  " + "; ".join(f"{e['code']}: {e['message']}" for e in res["errors"])
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
