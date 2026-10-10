#!/bin/bash
# atelier 설치 / 업데이트 (macOS). Python · Homebrew · 개발자 도구가 없어도 됩니다.
# 여러 번 실행해도 안전하고, 다시 실행하면 최신 버전으로 업데이트됩니다.
set -eo pipefail

REPO="cogito5170/diffusion_model"
BRANCH="claude/code-agent-image-gen-cz6kfx"
DEST="$HOME/diffusion_model"
LOG="$HOME/diffusion_model_install.log"
PY_VERSION="3.12.10"
PY_PKG_URL="https://www.python.org/ftp/python/${PY_VERSION}/python-${PY_VERSION}-macos11.pkg"

WITH_SDXL=0
for arg in "$@"; do
  [ "$arg" = "--sdxl" ] && WITH_SDXL=1
done

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date) ==="

step() { printf "\n\033[1m[%s] %s\033[0m\n" "$1" "$2"; }
fail() { printf "\n\033[31m%s\033[0m\n문제가 계속되면 이 파일을 보내 주세요: %s\n" "$1" "$LOG"; exit 1; }

[ "$(uname)" = "Darwin" ] || fail "이 설치 프로그램은 맥(macOS) 전용입니다."
[ -n "$HOME" ] && [ -d "$HOME" ] || fail "홈 폴더를 찾지 못했습니다."
if [ "$(uname -m)" != "arm64" ]; then
  echo "참고: Apple 실리콘(M1 이후) 맥이 아닙니다. 인텔 맥은 확인하지 않았고, 된다고 해도 많이 느립니다."
fi

# ---------------------------------------------------------------- 1. Python
step 1/5 "Python 확인 (3.10 이상)"
pick_python() {
  # Apple's /usr/bin/python3 is skipped on purpose: without developer tools it pops up an install dialog,
  # and with them it is 3.9 (too old).
  for p in /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
           /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
           /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \
           /usr/local/bin/python3.13 /usr/local/bin/python3.12 /usr/local/bin/python3.11; do
    [ -x "$p" ] && "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null && { echo "$p"; return 0; }
  done
  return 1
}
if ! PY=$(pick_python); then
  echo "Python이 없어서 python.org 공식 설치 파일(Python ${PY_VERSION}, 약 45MB)을 받습니다."
  PKG="$(mktemp -d)/python.pkg"
  curl -fL --progress-bar -o "$PKG" "$PY_PKG_URL" || fail "Python 설치 파일을 받지 못했습니다. 인터넷 연결을 확인해 주세요."
  pkgutil --check-signature "$PKG" | grep -q "Python Software Foundation" \
    || fail "받은 파일의 서명이 Python 공식 서명이 아닙니다. 설치를 멈췄습니다."
  echo ""
  echo "이제 Python을 설치합니다. 'Password:'가 나오면 맥 로그인 비밀번호를 입력하고 Enter를 누르세요."
  echo "(입력하는 동안 화면에 아무것도 보이지 않는 것이 정상입니다)"
  sudo installer -pkg "$PKG" -target / || fail "Python 설치가 끝나지 않았습니다."
  CERT="/Applications/Python ${PY_VERSION%.*}/Install Certificates.command"
  [ -f "$CERT" ] && bash "$CERT" >/dev/null 2>&1 || true
  PY=$(pick_python) || fail "Python을 설치했지만 찾지 못했습니다. 터미널을 닫았다 다시 열고 같은 명령을 실행해 주세요."
fi
echo "Python: $PY ($("$PY" --version))"

# ---------------------------------------------------------------- 2. code
step 2/5 "프로그램 받기 / 업데이트 → $DEST"
if [ -d "$DEST/.git" ] && xcode-select -p >/dev/null 2>&1; then
  git -C "$DEST" fetch -q origin "$BRANCH"
  git -C "$DEST" checkout -q "$BRANCH"
  git -C "$DEST" pull -q --ff-only origin "$BRANCH" || fail "업데이트 중 충돌이 있습니다. 프로그램 폴더의 파일을 직접 고친 적이 있나요?"
else
  TMP="$(mktemp -d)"
  curl -fL --progress-bar -o "$TMP/code.zip" "https://codeload.github.com/${REPO}/zip/refs/heads/${BRANCH}" \
    || fail "프로그램을 받지 못했습니다. 인터넷 연결을 확인해 주세요."
  unzip -q "$TMP/code.zip" -d "$TMP/unzipped"
  SRC="$(find "$TMP/unzipped" -mindepth 1 -maxdepth 1 -type d | head -1)"
  [ -n "$SRC" ] && [ -f "$SRC/studio.py" ] || fail "받은 프로그램 파일이 올바르지 않습니다. 다시 실행해 주세요."
  # Fresh code, but keep the user's results, environment and reference images (plain mv: no rsync needed).
  for keep in runs .venv examples/references .git; do
    if [ -e "$DEST/$keep" ]; then
      rm -rf "${SRC:?}/$keep"
      mkdir -p "$(dirname "$SRC/$keep")"
      mv "$DEST/$keep" "$SRC/$keep"
    fi
  done
  if [ -e "$DEST" ]; then
    mv "$DEST" "$TMP/previous"
  fi
  mv "$SRC" "$DEST"
  rm -rf "${TMP:?}"
fi
cd "$DEST"

# ---------------------------------------------------------------- 3. packages
step 3/5 "필요한 프로그램 설치 (처음에는 1–2GB, 수 분 ~ 수십 분)"
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  rm -rf .venv
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements-live.txt || fail "프로그램 설치 중 오류가 났습니다."

# ---------------------------------------------------------------- 4. model
step 4/5 "그림 모델 받기 (빠른 모델 sd-turbo 약 2.6GB, 처음 한 번만)"
.venv/bin/python scripts/download_model.py sd-turbo || fail "그림 모델을 받지 못했습니다. 인터넷 연결을 확인하고 다시 실행해 주세요."
if [ "$WITH_SDXL" = "1" ]; then
  echo "고화질 모델(SDXL, 약 7GB)도 받습니다. 시간이 꽤 걸립니다."
  .venv/bin/python scripts/download_model.py sdxl || fail "고화질 모델을 받지 못했습니다. 인터넷 연결을 확인하고 다시 실행해 주세요."
fi

# ---------------------------------------------------------------- 5. icons
step 5/5 "바탕화면에 아이콘 만들기"
mkdir -p "$HOME/Desktop"
make_icon() {
  printf '#!/bin/bash\ncd "%s" && exec .venv/bin/python %s\n' "$DEST" "$2" > "$HOME/Desktop/$1.command"
  chmod +x "$HOME/Desktop/$1.command"
}
make_icon "이미지 스튜디오" studio.py
make_icon "이미지 만들기" easy.py

# `imagine` in new terminal windows (zsh is the macOS default). noglob keeps * and ? in prompts literal.
chmod +x "$DEST/imagine"
ZRC="$HOME/.zshrc"
if ! grep -q ">>> atelier imagine >>>" "$ZRC" 2>/dev/null; then
  {
    echo ""
    echo "# >>> atelier imagine >>>"
    echo "alias imagine='noglob \"\$HOME/diffusion_model/imagine\"'"
    echo "# <<< atelier imagine <<<"
  } >> "$ZRC"
fi

printf "\n\033[32m설치 완료!\033[0m\n"
printf "바탕화면의 '이미지 스튜디오'를 더블클릭하면 브라우저에서 바로 쓸 수 있습니다.\n"
printf "터미널에서는 새 창을 열고 imagine 을 입력하세요 (프롬프트를 붙여 넣는 칸이 나옵니다).\n"
printf "업데이트가 필요할 때는 같은 설치 명령을 다시 실행하면 됩니다.\n"
if [ "$WITH_SDXL" != "1" ]; then
  printf "고화질 모델(SDXL)이 필요하면 설명서의 '고화질 모델 받기' 한 줄을 실행하세요.\n"
fi
