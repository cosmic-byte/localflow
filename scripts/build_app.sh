#!/usr/bin/env bash
# Build the macOS app bundle and install it so localflow launches from
# Spotlight or the Dock with no terminal involved.
#
# The bundle is a thin launcher around this checkout's virtualenv. The
# launcher runs python as a child process (not exec) so macOS attributes the
# microphone, accessibility, and input-monitoring grants to the app bundle
# instead of a terminal. Because the launcher points at this checkout's
# .venv, re-run this script after moving or re-cloning the repository.
#
# Usage:
#   ./scripts/build_app.sh                     # build + install to /Applications
#   BUILD_ONLY=1 ./scripts/build_app.sh        # build dist/<name>.app only
#   INSTALL_DIR=~/Applications ./scripts/build_app.sh
#   APP_NAME=LocalFlow BUNDLE_ID=com.example.localflow ./scripts/build_app.sh
set -euo pipefail

APP_NAME="${APP_NAME:-Whisper}"
BUNDLE_ID="${BUNDLE_ID:-com.whisper.widget}"
APP_VERSION="${APP_VERSION:-2.0}"
INSTALL_DIR="${INSTALL_DIR:-/Applications}"
BUILD_ONLY="${BUILD_ONLY:-0}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_DIR/.venv"
DIST_APP="$REPO_DIR/dist/$APP_NAME.app"

log()  { printf '\033[0;32m[build-app]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[build-app]\033[0m %s\n' "$*"; }

[ "$(uname)" = "Darwin" ] || { warn "macOS only"; exit 1; }

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

command -v ffmpeg >/dev/null 2>&1 || warn "ffmpeg not found: brew install ffmpeg"
command -v ollama >/dev/null 2>&1 || warn "ollama not found (cleanup pass disabled): brew install ollama && ollama pull qwen2.5:7b"

log "building $DIST_APP"
rm -rf "$DIST_APP"
mkdir -p "$DIST_APP/Contents/MacOS" "$DIST_APP/Contents/Resources"

cat > "$DIST_APP/Contents/MacOS/$APP_NAME" <<EOF
#!/bin/zsh
# GUI-launched apps get a minimal PATH; localflow shells out to ffmpeg.
export PATH="/opt/homebrew/bin:/usr/local/bin:\$PATH"
mkdir -p "\$HOME/.config/localflow"
exec >>"\$HOME/.config/localflow/launcher.log" 2>&1
# Run python as a child (not exec) so macOS attributes permission grants
# (microphone, accessibility, input monitoring) to this app bundle.
"$VENV/bin/python" -m localflow
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

LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
[ -x "$LSREGISTER" ] && "$LSREGISTER" -f "$INSTALL_DIR/$APP_NAME.app"

log "done. Launch \"$APP_NAME\" from Spotlight (Cmd+Space)."
log "First launch: grant Microphone, Accessibility and Input Monitoring to $APP_NAME in System Settings -> Privacy & Security."
