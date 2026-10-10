#!/bin/bash
# atelier 설치 / 업데이트 (macOS). 여러 번 실행해도 안전합니다. 다시 실행하면 최신으로 업데이트됩니다.
set -e

REPO_URL="https://github.com/cogito5170/diffusion_model.git"
BRANCH="claude/code-agent-image-gen-cz6kfx"
DEST="$HOME/diffusion_model"
SHORTCUT="$HOME/Desktop/이미지 만들기.command"

step() { printf "\n\033[1m[%s] %s\033[0m\n" "$1" "$2"; }
fail() { printf "\n\033[31m%s\033[0m\n" "$1"; exit 1; }

[ "$(uname)" = "Darwin" ] || fail "이 설치 스크립트는 맥(macOS) 전용입니다."
if [ "$(uname -m)" != "arm64" ]; then
  echo "참고: Apple 실리콘(M1 이후) 맥이 아니라서 생성이 많이 느릴 수 있습니다 (CPU 사용)."
fi

step 1/6 "기본 개발 도구 확인"
if ! xcode-select -p >/dev/null 2>&1; then
  xcode-select --install || true
  fail "방금 뜬 창에서 '설치'를 눌러 주세요. 설치가 끝나면 이 명령을 한 번 더 실행하면 됩니다."
fi

step 2/6 "프로그램 받기 / 업데이트 → $DEST"
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" fetch -q origin "$BRANCH"
  git -C "$DEST" checkout -q "$BRANCH"
  git -C "$DEST" pull -q --ff-only origin "$BRANCH" || fail "업데이트 중 충돌이 있습니다. $DEST 폴더 안의 파일을 직접 고친 적이 있다면 알려 주세요."
else
  git clone -q -b "$BRANCH" "$REPO_URL" "$DEST"
fi
cd "$DEST"

step 3/6 "Python 3.10 이상 찾기"
pick_python() {
  for c in python3.13 python3.12 python3.11 python3.10 \
           /opt/homebrew/bin/python3 /usr/local/bin/python3 \
           /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
           /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 python3; do
    p=$(command -v "$c" 2>/dev/null) || continue
    "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null && { echo "$p"; return 0; }
  done
  return 1
}
PY=$(pick_python) || {
  if command -v brew >/dev/null 2>&1; then
    echo "Homebrew로 Python 3.12를 설치합니다..."
    brew install python@3.12
    PY=$(pick_python) || fail "Python 설치 후에도 찾지 못했습니다. 터미널을 닫았다 열고 다시 실행해 주세요."
  else
    open "https://www.python.org/downloads/macos/"
    fail "Python 3.12가 필요합니다. 방금 열린 python.org 페이지에서 'macOS 64-bit universal2 installer'를 받아 설치한 뒤, 이 명령을 다시 실행해 주세요."
  fi
}
echo "사용할 Python: $PY ($("$PY" --version))"

step 4/6 "전용 환경(.venv) 만들고 필요한 프로그램 설치 (처음엔 수 분 ~ 수십 분)"
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  rm -rf .venv
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements-live.txt

step 5/6 "그림 모델 받기 (sd-turbo, 약 2.6GB, 처음 한 번만)"
.venv/bin/python scripts/download_sd_turbo.py

step 6/6 "바탕화면에 '이미지 만들기' 아이콘 만들기"
cat > "$SHORTCUT" <<LAUNCH
#!/bin/bash
cd "$DEST" && exec .venv/bin/python easy.py
LAUNCH
chmod +x "$SHORTCUT"

printf "\n\033[32m설치 완료!\033[0m 바탕화면의 '이미지 만들기'를 더블클릭하세요.\n"
printf "업데이트가 필요할 때는 이 설치 명령을 다시 실행하면 됩니다.\n"
