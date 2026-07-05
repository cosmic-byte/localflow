# localflow

Local dictation for macOS (Apple Silicon). Hold a key, speak, release — a local Whisper
model transcribes, a local LLM on Ollama removes fillers and fixes punctuation, and the
cleaned text is typed into whatever app is focused. No audio or text ever leaves the
machine.

## How it works

```
hotkey (hold = push-to-talk, double-tap = hands-free)
  → ffmpeg mic capture (16 kHz mono, in memory)
  → MLX Whisper large-v3-turbo (Metal), biased by your dictionary
  → Ollama qwen2.5:7b cleanup (skipped for short utterances)
  → paste into the focused app (clipboard is saved and restored)
```

## Install

Runtime dependencies first (both install options need them):

```sh
brew install ollama ffmpeg
ollama pull qwen2.5:7b
```

### Option A — app bundle (no terminal)

Builds `Whisper.app` and installs it to `/Applications`, so the widget starts
from Spotlight (Cmd+Space) or the Dock like any other Mac app:

```sh
./scripts/build_app.sh
```

The bundle is a thin launcher around this checkout's virtualenv (created
automatically), so keep the cloned folder around and re-run the script if you
move it. Variants: `INSTALL_DIR=~/Applications`, `APP_NAME=LocalFlow`,
`BUILD_ONLY=1` (build `dist/` without installing). Logs from app launches go
to `~/.config/localflow/launcher.log`.

### Option B — CLI

```sh
uv venv && uv pip install -e ".[dev]"
```

## Run

Option A: launch **Whisper** from Spotlight — the floating widget appears and
the global hotkey is live. Option B:

```sh
localflow            # floating widget + global hotkey
localflow --cli      # terminal mode: record until Enter, then transcribe
localflow --cli --duration 5
```

## Window controls

The widget window has standard macOS **traffic light buttons** in the top-left
corner:

- **Red (close)** — quit the app
- **Yellow (minimize)** — minimize the widget to the Dock (click the Dock icon
  to restore it)
- **Green (zoom)** — reserved for the fixed-size widget (size does not change)

**Right-click the pill** to change microphone, ASR model, cleanup settings, and
more. You can also drag the pill to the **bottom-right corner** of the screen to
quit.

## Permissions (one-time)

System Settings → Privacy & Security. With the app bundle the grants attach
to **Whisper** (asked on first launch); with the CLI they attach to your
terminal:

- **Microphone** — for the first recording.
- **Accessibility** — required to paste into other apps and for the global hotkey.
- **Input Monitoring** — required for the global event tap.

If the default `fn` hotkey conflicts with the system dictation shortcut, set
System Settings → Keyboard → "Press fn key to" → "Do Nothing", or pick another hotkey in
`~/.config/localflow/config.json` (e.g. `"ctrl+alt+space"`).

## Config

- `~/.config/localflow/config.json` — settings (models, hotkey, microphone, tone).
  Before switching the cleanup model to `llama3.2:3b` in the menu, pull it first:
  `ollama pull llama3.2:3b`.
- `~/.config/localflow/dictionary.json` — `words` bias recognition (names, jargon);
  `replacements` are deterministic post-fixes applied to every transcript.
- `~/.config/localflow/localflow.log` — log file.

## Development

```sh
uv run ruff check src && uv run ruff format --check src
uv run pytest
uv run python scripts/benchmark_asr.py --record 10   # compare ASR backends
```
