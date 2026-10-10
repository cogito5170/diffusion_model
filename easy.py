"""'이미지 만들기': a menu for non-developers. Ask a few questions, generate, and open the results.

Launched by the desktop icon that install_mac.sh creates. Everything still goes through the same engine
runner, budget, QA and records (scripts/prompt_once.py and scripts/generate_once.py).
Set ATELIER_EASY_CONFIG to use another run config (the tests use the mock engine).
"""

from __future__ import annotations

import datetime as _dt
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = os.environ.get("ATELIER_EASY_CONFIG", "configs/live_sd_turbo_mac.json")
MAX_COUNT = 4

EXAMPLES = {
    "1": ("A: One Sheet, Folded (브러시드 스틸 + 빛나는 천장)", "steel_light"),
    "2": ("B: Seat for No One (깎인 흰 폼 의자)", "foam_vitrine"),
    "3": ("The Memory of Space (로봇 팔 + 투명 구조물 + 물)", "memory_of_space"),
}


def has_korean(text: str) -> bool:
    return re.search(r"[가-힣]", text) is not None


def parse_slot(text: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"\s*(\d{2,5})\s*[xX×*]\s*(\d{2,5})\s*", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def new_name() -> str:
    return "easy-" + _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def ask(question: str, default: str = "") -> str:
    hint = f" [엔터 = {default}]" if default else ""
    answer = input(f"\n{question}{hint}\n> ").strip()
    return answer or default


def ask_count() -> int:
    while True:
        raw = ask(f"몇 장 만들까요? (1~{MAX_COUNT})", "4")
        if raw.isdigit() and 1 <= int(raw) <= MAX_COUNT:
            return int(raw)
        print(f"  1부터 {MAX_COUNT} 사이 숫자를 입력해 주세요.")


def ask_prompt() -> str:
    print("\n만들고 싶은 장면을 영어로 적어 주세요. 여러 줄로 붙여 넣어도 됩니다.")
    print("다 적었으면 빈 줄에서 엔터를 한 번 더 누르세요.")
    lines: list[str] = []
    while True:
        line = input("> " if not lines else "  ")
        if not line.strip():
            if lines:
                break
            continue
        lines.append(line.strip())
    return " ".join(lines)


def env() -> dict[str, str]:
    e = dict(os.environ)
    e.setdefault("DIFFUSERS_VERBOSITY", "error")    # hide library notices (e.g. the safety-checker notice)
    e.setdefault("TRANSFORMERS_VERBOSITY", "error")
    e.setdefault("PYTHONWARNINGS", "ignore")
    return e


def run(cmd: list[str]) -> int:
    print("\n생성 중입니다... (첫 장은 모델을 불러오느라 조금 더 걸립니다)\n")
    return subprocess.call(cmd, cwd=ROOT, env=env())


def open_paths(paths: list[str]) -> None:
    if not paths:
        print("열 이미지가 없습니다. 위의 메시지를 확인해 주세요.")
        return
    if sys.platform == "darwin":
        subprocess.call(["open", *paths])
    else:
        print("\n".join(paths))


def make_from_prompt() -> None:
    prompt = ask_prompt()
    if has_korean(prompt):
        print("\n참고: 이 모델은 한국어를 거의 이해하지 못합니다. 영어로 쓰면 결과가 훨씬 좋아집니다.")
        if ask("그래도 이대로 만들까요? (y/n)", "n").lower() != "y":
            return
    count = ask_count()
    slot = None
    while True:
        raw = ask("특정 크기로 잘라 둘까요? 예: 574x172 (필요 없으면 그냥 엔터)", "")
        if not raw:
            break
        slot = parse_slot(raw)
        if slot:
            break
        print("  '가로x세로' 형식으로 입력해 주세요. 예: 574x172")
    name = new_name()
    cmd = [sys.executable, "scripts/prompt_once.py", "--config", CONFIG, "--name", name,
           "--count", str(count), "--prompt", prompt]
    if slot:
        cmd += ["--crop", f"{slot[0]}x{slot[1]}"]
    if run(cmd) != 0:
        print("\n생성에 실패했습니다. 위 메시지를 그대로 복사해서 물어봐 주세요.")
        return
    out = ROOT / "runs" / f"run_{name}"
    (out / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    pattern = f"slot_{slot[0]}x{slot[1]}.png" if slot else "image_0.png"
    open_paths(sorted(glob.glob(str(out / "images" / "*" / "a1" / pattern))))
    print(f"\n결과 폴더: {out}")


def make_from_example() -> None:
    print()
    for key, (label, _) in EXAMPLES.items():
        print(f"  {key}. {label}")
    choice = ask("어떤 작품을 만들까요?", "1")
    if choice not in EXAMPLES:
        print("  목록에 있는 번호를 골라 주세요.")
        return
    stem = EXAMPLES[choice][1]
    count = ask_count()
    name = new_name()
    cmd = [sys.executable, "scripts/generate_once.py", "--config", CONFIG, "--name", name, "--count", str(count),
           "--brief", f"examples/briefs/{stem}.json", "--art-direction", f"examples/artifacts/{stem}.art_direction.json"]
    if run(cmd) != 0:
        print("\n생성에 실패했습니다. 위 메시지를 그대로 복사해서 물어봐 주세요.")
        return
    out = ROOT / "runs" / f"run_{name}"
    open_paths(sorted(glob.glob(str(out / "images" / "*" / "a1" / "image_0.png"))))
    print(f"\n결과 폴더: {out}")


def open_results() -> None:
    folder = ROOT / "runs"
    folder.mkdir(exist_ok=True)
    if sys.platform == "darwin":
        subprocess.call(["open", str(folder)])
    else:
        print(folder)


def main() -> int:
    print("=" * 52)
    print("  이미지 만들기  ·  atelier")
    print("=" * 52)
    actions = {"1": make_from_prompt, "2": make_from_example, "3": open_results}
    while True:
        print("\n  1. 내 문장(프롬프트)으로 이미지 만들기")
        print("  2. 미리 설계된 작품 만들기")
        print("  3. 지금까지 만든 결과 폴더 열기")
        print("  q. 끝내기")
        choice = ask("번호를 입력하세요", "1").lower()
        if choice in ("q", "quit", "exit", "종료"):
            print("안녕히 가세요.")
            return 0
        action = actions.get(choice)
        if action is None:
            print("  1, 2, 3, q 중에서 골라 주세요.")
            continue
        try:
            action()
        except KeyboardInterrupt:
            print("\n중단했습니다.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n끝냈습니다.")
