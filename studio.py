"""이미지 스튜디오: a Midjourney-style local web page. Type a prompt, get a grid, re-roll or reuse a seed.

    .venv/bin/python studio.py          # opens http://127.0.0.1:7860 in the browser

Prompt flags (parsed out, never sent to the model):
    --ar 16:9      aspect ratio (1:1 default; up to 3:1)
    --n 4          number of images (1-4)
    --seed 1234    first seed (images use seed, seed+7919, ...); omitted = random
    --crop 574x172 also save a centre crop resized to a layout slot

The page binds to 127.0.0.1 only. Every image goes through the same engine runner, budget, QA and records as the
rest of atelier (runs/run_studio-*/). The model is loaded once and reused, so only the first image is slow.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import secrets
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from atelier.artifact_store import ArtifactStore  # noqa: E402
from atelier.budget import Budget, CancelToken  # noqa: E402
from atelier.config import load_config  # noqa: E402
from atelier.engines import create_engine  # noqa: E402
from atelier.engines.runner import execute_request  # noqa: E402
from atelier.evaluator import run_qa  # noqa: E402
from atelier.prompt_compiler import CompilationError, compile_baseline_request  # noqa: E402

RUNS = ROOT / "runs"
MAX_N = 4
LONG_SIDE = 768
SQUARE_SIDE = 512  # sd-turbo's native resolution
FLAG_RE = re.compile(r"--(ar|n|seed|crop)\s+(\S+)", re.IGNORECASE)


class PromptError(ValueError):
    pass


def size_for_ratio(ar: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{1,2}):(\d{1,2})", ar)
    if not m or int(m.group(1)) == 0 or int(m.group(2)) == 0:
        raise PromptError(f"--ar must look like 16:9 (got {ar!r})")
    a, b = int(m.group(1)), int(m.group(2))
    if max(a, b) / min(a, b) > 3:
        raise PromptError("--ar can be at most 3:1. For wider layout slots, use --crop (e.g. --crop 574x172)")
    if a == b:
        return SQUARE_SIDE, SQUARE_SIDE
    if a > b:
        return LONG_SIDE, max(256, round(LONG_SIDE * b / a / 8) * 8)
    return max(256, round(LONG_SIDE * a / b / 8) * 8), LONG_SIDE


def parse_prompt(text: str, defaults: dict) -> dict:
    """Split Midjourney-style flags off the prompt. Flags written in the prompt override the page's controls."""
    opts = {"ar": defaults.get("ar") or "1:1", "n": defaults.get("n") or MAX_N,
            "seed": defaults.get("seed"), "crop": defaults.get("crop") or None}
    for key, value in FLAG_RE.findall(text):
        opts[key.lower()] = value
    prompt = " ".join(FLAG_RE.sub(" ", text).split())
    if not prompt:
        raise PromptError("프롬프트가 비어 있습니다 (the prompt is empty)")
    try:
        opts["n"] = int(opts["n"])
    except (TypeError, ValueError):
        raise PromptError("--n must be a number from 1 to 4") from None
    if not 1 <= opts["n"] <= MAX_N:
        raise PromptError("--n must be from 1 to 4")
    if opts["seed"] in (None, ""):
        opts["seed"] = secrets.randbelow(2**31)
    else:
        try:
            opts["seed"] = int(opts["seed"])
        except ValueError:
            raise PromptError("--seed must be a whole number") from None
        if not 0 <= opts["seed"] < 2**32:
            raise PromptError("--seed must be between 0 and 4294967295")
    if opts["crop"]:
        m = re.fullmatch(r"(\d{2,5})[xX×](\d{2,5})", str(opts["crop"]).strip())
        if not m:
            raise PromptError("--crop must look like 574x172")
        opts["crop"] = (int(m.group(1)), int(m.group(2)))
    opts["width"], opts["height"] = size_for_ratio(str(opts["ar"]))
    opts["prompt"] = prompt
    return opts


def crop_to_slot(src: Path, slot_w: int, slot_h: int) -> Path:
    from PIL import Image

    with Image.open(src) as im:
        w, h = im.size
        target = slot_w / slot_h
        cw, ch = (round(h * target), h) if w / h > target else (w, round(w / target))
        left, top = (w - cw) // 2, (h - ch) // 2
        out = src.with_name(f"slot_{slot_w}x{slot_h}.png")
        im.crop((left, top, left + cw, top + ch)).resize((slot_w, slot_h), Image.LANCZOS).save(out)
    return out


class Studio:
    def __init__(self, config_path: str):
        self.cfg = load_config(config_path)
        self.engine = create_engine(self.cfg["engine"])
        self.lock = threading.Lock()  # one generation at a time; the model is not shared across threads

    def generate(self, raw_prompt: str, defaults: dict) -> dict:
        opts = parse_prompt(raw_prompt, defaults)
        av = self.engine.availability()
        if not av.ok:
            raise PromptError(f"engine unavailable: {av.reason}")
        caps, g = self.engine.capabilities(), self.cfg["generation"]
        name = "studio-" + _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        store = ArtifactStore(RUNS / f"run_{name}")
        budget = Budget(dict(self.cfg["budget"], max_total_images=opts["n"]), Budget.empty_ledger())
        job = {"name": name, "prompt": raw_prompt, "prompt_text": opts["prompt"], "ar": opts["ar"],
               "width": None, "height": None, "first_seed": opts["seed"], "crop": opts["crop"],
               "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
               "engine": f"{caps.engine_id}/{caps.model_id}", "warnings": [], "images": []}
        with self.lock:
            for slot in range(opts["n"]):
                try:
                    req = compile_baseline_request(opts["prompt"], caps, run_id=f"run_{name}", slot=slot,
                                                   width=opts["width"], height=opts["height"], base_seed=opts["seed"],
                                                   engine_params=self.cfg["engine_params"], timeout_s=g["timeout_s"],
                                                   max_retries=g["max_retries"])
                except CompilationError as exc:
                    raise PromptError(str(exc)) from None
                if slot == 0:
                    job["prompt_sent"] = req["prompt"]["positive"]
                    job["width"], job["height"] = req["width"], req["height"]
                    job["warnings"] = [w["message"] for w in req["compilation"]["warnings"]]
                store.put(req)
                res = execute_request(self.engine, req, store, budget, CancelToken())
                store.put(res)
                item = {"seed": req["seed"], "status": res["status"], "runtime_s": res["runtime_s"],
                        "url": None, "crop_url": None, "qa_passed": None,
                        "error": "; ".join(e["message"] for e in res["errors"]) if res["status"] != "succeeded" else None}
                if res["status"] == "succeeded":
                    qa = run_qa(res, req, store, self.cfg["qa"])
                    store.put(qa)
                    img = store.abs(res["images"][0]["path"])
                    item["qa_passed"] = qa["passed"]
                    item["url"] = "/runs/" + img.relative_to(RUNS).as_posix()
                    if opts["crop"]:
                        item["crop_url"] = "/runs/" + crop_to_slot(img, *opts["crop"]).relative_to(RUNS).as_posix()
                job["images"].append(item)
        (store.root / "job.json").write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
        return job


def history(limit: int = 30) -> list[dict]:
    jobs = []
    for path in sorted(RUNS.glob("run_studio-*/job.json"), reverse=True)[:limit]:
        try:
            jobs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return jobs


def make_handler(studio: Studio):
    page = (ROOT / "atelier" / "studio.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            if self.path == "/api/history":
                return self._json(HTTPStatus.OK, history())
            if self.path.startswith("/runs/"):
                rel = self.path[len("/runs/"):].split("?", 1)[0]
                target = (RUNS / rel).resolve()
                if RUNS.resolve() not in target.parents or target.suffix.lower() != ".png" or not target.is_file():
                    return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return self._send(HTTPStatus.OK, target.read_bytes(), "image/png")
            return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/generate":
                return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            length = int(self.headers.get("Content-Length") or 0)
            if length > 20000:
                return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                job = studio.generate(str(body.get("prompt", "")),
                                      {"ar": body.get("ar"), "n": body.get("n"), "seed": body.get("seed"),
                                       "crop": body.get("crop")})
            except PromptError as exc:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - report instead of crashing the page
                return self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})
            return self._json(HTTPStatus.OK, job)

    return Handler


def serve(config_path: str, port: int = 7860, open_browser: bool = True) -> ThreadingHTTPServer:
    studio = Studio(config_path)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(studio))
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"이미지 스튜디오가 열렸습니다: {url}")
    print("끝내려면 이 창을 닫거나 Control + C 를 누르세요.")
    if open_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    return server


def main() -> int:
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    config = os.environ.get("ATELIER_STUDIO_CONFIG", "configs/live_sd_turbo_mac.json")
    port = int(os.environ.get("ATELIER_STUDIO_PORT", "7860"))
    server = serve(config, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n스튜디오를 닫았습니다.")
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
