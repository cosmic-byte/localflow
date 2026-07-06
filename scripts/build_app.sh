#!/usr/bin/env bash
# Build the macOS app bundle and install it so localflow launches from
# Spotlight or the Dock with no terminal involved.
#
# The bundle wraps this checkout's virtualenv. A copy of the interpreter
# lives inside the bundle (Contents/MacOS) so macOS permission prompts and
# grants (microphone, accessibility, input monitoring) show and track the
# app's name instead of the underlying "python3.x" binary. The interpreter
# resolves this checkout's .venv through Contents/pyvenv.cfg and a lib
# symlink, so re-run this script after moving or re-cloning the repository
# or after upgrading python.
#
# On a fresh machine the script also walks through the runtime dependencies
# interactively: Homebrew, ffmpeg (audio capture), ollama (text cleanup),
# starting the ollama service, and pulling the cleanup model.
#
# Usage:
#   ./scripts/build_app.sh                     # deps (interactive) + build + install
#   ./scripts/build_app.sh --uninstall         # interactively undo the install
#   ASSUME_YES=1 ./scripts/build_app.sh        # accept every prompt automatically
#   BUILD_ONLY=1 ./scripts/build_app.sh        # build dist/<name>.app only, skip deps
#   INSTALL_DIR=~/Applications ./scripts/build_app.sh
#   APP_NAME=LocalFlow BUNDLE_ID=com.example.localflow ./scripts/build_app.sh
set -euo pipefail

APP_NAME="${APP_NAME:-Whisper}"
BUNDLE_ID="${BUNDLE_ID:-com.whisper.widget}"
APP_VERSION="${APP_VERSION:-2.0}"
INSTALL_DIR="${INSTALL_DIR:-/Applications}"
BUILD_ONLY="${BUILD_ONLY:-0}"
ASSUME_YES="${ASSUME_YES:-0}"
CLEANUP_MODEL="${CLEANUP_MODEL:-qwen2.5:7b}"
ASR_MODEL="${ASR_MODEL:-large-v3-turbo}"
OLLAMA_URL="http://localhost:11434"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_DIR/.venv"
DIST_APP="$REPO_DIR/dist/$APP_NAME.app"
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"

UNINSTALL=0
for arg in "$@"; do
  case "$arg" in
    --uninstall) UNINSTALL=1 ;;
    *) printf 'unknown argument: %s (supported: --uninstall)\n' "$arg" >&2; exit 1 ;;
  esac
done

log()  { printf '\033[0;32m[build-app]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[build-app]\033[0m %s\n' "$*"; }

# Yes/no prompt, default yes. ASSUME_YES=1 answers yes to everything; a
# non-interactive shell answers no (and says how to override) instead of
# hanging or silently installing things.
confirm() {
  local question="$1" reply
  if [ "$ASSUME_YES" = "1" ]; then
    return 0
  fi
  if [ ! -t 0 ]; then
    warn "non-interactive shell, skipping: $question (use ASSUME_YES=1 to accept)"
    return 1
  fi
  printf '\033[0;36m[build-app]\033[0m %s [Y/n] ' "$question"
  read -r reply
  case "$reply" in
    n|N|no|NO) return 1 ;;
    *) return 0 ;;
  esac
}

# Yes/no prompt, default no, for steps that can affect other software.
# ASSUME_YES=1 still answers yes; a non-interactive shell answers no.
confirm_no() {
  local question="$1" reply
  if [ "$ASSUME_YES" = "1" ]; then
    return 0
  fi
  if [ ! -t 0 ]; then
    warn "non-interactive shell, skipping: $question (use ASSUME_YES=1 to accept)"
    return 1
  fi
  printf '\033[0;36m[build-app]\033[0m %s [y/N] ' "$question"
  read -r reply
  case "$reply" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

ollama_running() {
  curl -s --max-time 2 "$OLLAMA_URL/api/version" >/dev/null 2>&1
}

ensure_runtime_deps() {
  if ! command -v brew >/dev/null 2>&1; then
    if { ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ollama >/dev/null 2>&1; } \
        && confirm "Homebrew is needed to install ffmpeg/ollama. Install Homebrew now?"; then
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null)" || \
        eval "$(/usr/local/bin/brew shellenv 2>/dev/null)" || true
    fi
  fi

  if ! command -v ffmpeg >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1 && \
        confirm "ffmpeg is required for audio capture. Install it now?"; then
      brew install ffmpeg
    else
      warn "ffmpeg missing - recording will not work until you run: brew install ffmpeg"
    fi
  fi

  if ! command -v ollama >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1 && \
        confirm "ollama powers the text-cleanup pass (optional). Install it now?"; then
      brew install ollama
    else
      warn "ollama missing - transcripts will be raw Whisper output (no cleanup pass)"
      return 0
    fi
  fi

  if ! ollama_running; then
    if confirm "ollama is installed but not running. Start it now?"; then
      if command -v brew >/dev/null 2>&1 && brew list ollama >/dev/null 2>&1; then
        brew services start ollama
      else
        nohup ollama serve >/dev/null 2>&1 &
      fi
      local i
      for i in $(seq 1 15); do
        ollama_running && break
        sleep 1
      done
      ollama_running || warn "ollama did not come up after 15s - check 'ollama serve' manually"
    fi
  fi

  if ollama_running && ! ollama list 2>/dev/null | grep -qF "$CLEANUP_MODEL"; then
    if confirm "Cleanup model $CLEANUP_MODEL is not downloaded yet (~4.7 GB). Pull it now?"; then
      ollama pull "$CLEANUP_MODEL"
    else
      warn "model missing - pull it later with: ollama pull $CLEANUP_MODEL"
    fi
  fi
}

# The first dictation triggers the Whisper model download inside the app,
# where the wait is invisible; offering it here keeps it in the installer.
ensure_asr_model() {
  local repo hf_dir
  repo="$("$VENV/bin/python" -c \
    "from localflow.asr import _mlx_repo_for_model; print(_mlx_repo_for_model('$ASR_MODEL'))" \
    2>/dev/null)" || { warn "unknown ASR model $ASR_MODEL - skipping pre-download"; return 0; }
  hf_dir="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${repo//\//--}"
  if [ -d "$hf_dir" ]; then
    return 0
  fi
  if confirm "Pre-download the Whisper speech model $ASR_MODEL now? (~1.6 GB; otherwise it downloads on the first dictation)"; then
    "$VENV/bin/python" -c \
      "from localflow.asr import create_transcriber; create_transcriber('mlx', '$ASR_MODEL').load()" \
      || warn "model pre-download failed - it will download on first use"
  fi
}

# Undo everything the installer set up. One default-yes prompt covers the
# app's own resources; a second, default-no prompt covers the shared
# Homebrew tools (ollama, ffmpeg) since other software may depend on them.
# Homebrew itself and this cloned repository are always left in place.
uninstall() {
  local app_path="$INSTALL_DIR/$APP_NAME.app"
  local hf_hub="${HF_HOME:-$HOME/.cache/huggingface}/hub"

  log "Uninstall removes: $app_path, the $CLEANUP_MODEL ollama model, the"
  log "downloaded Whisper models, ~/.config/localflow (settings, dictionary,"
  log "logs), the build virtualenv, and the app's recorded privacy permissions."
  if confirm "Remove all of the above?"; then
    pkill -f -- "-m localflow" 2>/dev/null || true
    if [ -d "$app_path" ]; then
      if [ -x "$LSREGISTER" ]; then
        "$LSREGISTER" -u "$app_path" 2>/dev/null || true
      fi
      rm -rf "$app_path"
    fi
    if command -v ollama >/dev/null 2>&1 && ollama_running && \
        ollama list 2>/dev/null | grep -qF "$CLEANUP_MODEL"; then
      ollama rm "$CLEANUP_MODEL" || warn "could not remove the $CLEANUP_MODEL model"
    fi
    rm -rf "$hf_hub"/models--mlx-community--whisper-* \
      "$HOME/.config/localflow" "$VENV" "$REPO_DIR/dist"
    tccutil reset All "$BUNDLE_ID" >/dev/null 2>&1 || warn \
      "could not reset permissions - remove the $APP_NAME entries in System Settings -> Privacy & Security"
    log "core resources removed"
  fi

  if command -v brew >/dev/null 2>&1; then
    local shared=()
    brew list ollama >/dev/null 2>&1 && shared+=(ollama)
    brew list ffmpeg >/dev/null 2>&1 && shared+=(ffmpeg)
    if [ ${#shared[@]} -gt 0 ] && \
        confirm_no "Also uninstall the shared tools (${shared[*]})? Other apps may use them"; then
      if brew list ollama >/dev/null 2>&1; then
        brew services stop ollama 2>/dev/null || true
        brew uninstall ollama
        rm -rf "$HOME/.ollama"
      fi
      if brew list ffmpeg >/dev/null 2>&1; then
        brew uninstall ffmpeg
      fi
    fi
  fi

  log "uninstall finished (Homebrew itself and this repository were left in place)"
}

[ "$(uname)" = "Darwin" ] || { warn "macOS only"; exit 1; }

if [ "$UNINSTALL" = "1" ]; then
  uninstall
  exit 0
fi

if [ "$BUILD_ONLY" != "1" ]; then
  ensure_runtime_deps
fi

# Virtualenv with localflow installed (idempotent).
if [ ! -x "$VENV/bin/python" ]; then
  log "creating virtualenv"
  if command -v uv >/dev/null 2>&1; then
    (cd "$REPO_DIR" && uv venv)
  else
    python3 -m venv "$VENV"
  fi
fi
log "installing localflow into the virtualenv"
if command -v uv >/dev/null 2>&1; then
  (cd "$REPO_DIR" && uv pip install -q -e .)
else
  "$VENV/bin/pip" install -q -e "$REPO_DIR"
fi

if [ "$BUILD_ONLY" != "1" ]; then
  ensure_asr_model
fi

log "building $DIST_APP"
rm -rf "$DIST_APP"
mkdir -p "$DIST_APP/Contents/MacOS" "$DIST_APP/Contents/Resources"

# Bundle a copy of the venv's real interpreter so macOS permission prompts
# and grants name and track this app instead of "python3.x". The copy still
# resolves the repo virtualenv: Contents/pyvenv.cfg marks Contents as the
# venv prefix and the lib symlink points its site-packages at the checkout.
PYTHON_REAL="$("$VENV/bin/python" -c 'import os, sys; print(os.path.realpath(sys.executable))')"
cp "$PYTHON_REAL" "$DIST_APP/Contents/MacOS/$APP_NAME-python"
cp "$VENV/pyvenv.cfg" "$DIST_APP/Contents/pyvenv.cfg"
ln -s "$VENV/lib" "$DIST_APP/Contents/lib"

cat > "$DIST_APP/Contents/MacOS/$APP_NAME" <<EOF
#!/bin/zsh
# GUI-launched apps get a minimal PATH; localflow shells out to ffmpeg.
export PATH="/opt/homebrew/bin:/usr/local/bin:\$PATH"
mkdir -p "\$HOME/.config/localflow"
exec >>"\$HOME/.config/localflow/launcher.log" 2>&1
BIN_DIR="\$(cd "\$(dirname "\$0")" && pwd)"
exec "\$BIN_DIR/$APP_NAME-python" -m localflow
EOF
chmod +x "$DIST_APP/Contents/MacOS/$APP_NAME"

cat > "$DIST_APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>
    <string>$APP_NAME</string>
    <key>CFBundleVersion</key>
    <string>$APP_VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>$APP_VERSION</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSUIElement</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>$APP_NAME records your voice to transcribe it into text, entirely on this Mac.</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

cp "$REPO_DIR/assets/AppIcon.icns" "$DIST_APP/Contents/Resources/AppIcon.icns"

if [ "$BUILD_ONLY" = "1" ]; then
  log "done (build only): $DIST_APP"
  exit 0
fi

log "installing to $INSTALL_DIR/$APP_NAME.app"
rm -rf "$INSTALL_DIR/$APP_NAME.app"
cp -R "$DIST_APP" "$INSTALL_DIR/$APP_NAME.app"

if [ -x "$LSREGISTER" ]; then
  "$LSREGISTER" -f "$INSTALL_DIR/$APP_NAME.app"
fi

log "done. Launch \"$APP_NAME\" from Spotlight (Cmd+Space)."
log "First launch: grant Microphone, Accessibility and Input Monitoring to $APP_NAME in System Settings -> Privacy & Security."
