"""`imagine`: Midjourney-style generation from the terminal.

    imagine                                  # then paste the prompt, empty line + Enter
    imagine a brushed steel arch --ar 16:9   # or give it inline

Same flags as the studio page (--ar, --n, --seed, --crop, --no) plus --hq for the high-quality model (SDXL).
Results are saved as studio jobs (runs/run_studio-*), so they also appear in the studio page's history.
Reading the prompt inside the program (instead of as shell words) avoids zsh quoting, globbing and `!` issues.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import studio  # noqa: E402

HQ_RE = re.compile(r"(^|\s)--hq(?=\s|$)", re.IGNORECASE)


def read_prompt() -> str:
    print("프롬프트를 붙여 넣고 Enter. 다 넣었으면 빈 줄에서 Enter를 한 번 더 누르세요.")
    lines: list[str] = []
    while True:
        try:
            line = input("> " if not lines else "  ")
        except EOFError:
            break
        if not line.strip():
            if lines:
                break
            continue
        lines.append(line.strip())
    return " ".join(lines)


def split_model(text: str) -> tuple[str, str]:
    if HQ_RE.search(text):
        return HQ_RE.sub(" ", text).strip(), "quality"
    return text.strip(), "fast"


def main(argv: list[str]) -> int:
    text = " ".join(argv).strip() or read_prompt()
    if not text:
        print("프롬프트가 비어 있습니다.")
        return 1
    text, model = split_model(text)
    configs = None
    if os.environ.get("ATELIER_IMAGINE_CONFIG"):  # tests / dry runs with the mock engine
        configs = {"fast": os.environ["ATELIER_IMAGINE_CONFIG"], "quality": os.environ["ATELIER_IMAGINE_CONFIG"]}
    s = studio.Studio(configs)
    label = studio.MODELS.get(model, {}).get("label", model)
    print(f"\n{label} 로 만드는 중... (처음엔 모델을 불러오느라 " + ("1–2분" if model == "quality" else "10초쯤") + " 더 걸립니다)")
    try:
        job = s.generate(text, {}, model=model)
    except studio.PromptError as exc:
        print(f"\n만들지 못했습니다: {exc}")
        return 1
    if job.get("truncated"):
        print("\n참고: 문장이 길어서 앞부분만 사용했습니다. 실제로 사용된 문장:\n  " + job["prompt_sent"])
    for w in job.get("warnings", []):
        if "only the beginning was used" not in w:
            print("참고:", w)
    paths = []
    for img in job["images"]:
        if img["url"]:
            p = studio.RUNS / img["url"][len("/runs/"):]
            paths.append(str(p))
            if img.get("crop_url"):
                paths.append(str(studio.RUNS / img["crop_url"][len("/runs/"):]))
            print(f"  시드 {img['seed']}  {img['runtime_s']:.1f}초  {p}")
        else:
            print(f"  시드 {img['seed']}  실패: {img['error']}")
    if job.get("error"):
        print("\n" + job["error"])
    if paths and sys.platform == "darwin":
        subprocess.call(["open", *paths])
    return 0 if paths else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\n중단했습니다.")
        raise SystemExit(130)
