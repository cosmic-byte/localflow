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

### Option A — app bundle (one command, no terminal afterwards)

```sh
./scripts/build_app.sh
```

The script handles the entire first-run setup interactively: it offers to
install anything missing (Homebrew, ffmpeg, ollama), starts the ollama
service, pulls the cleanup model, offers to pre-download the Whisper speech
model (so the first dictation doesn't stall on a ~1.6 GB download), then
builds `Whisper.app` and installs it to `/Applications` so the widget starts
from Spotlight (Cmd+Space). Re-run it any time — completed steps are skipped.

The bundle is a thin launcher around this checkout's virtualenv (created
automatically), so keep the cloned folder around and re-run the script if you
move it. Variants: `ASSUME_YES=1` (accept every prompt, for scripted
installs), `INSTALL_DIR=~/Applications`, `APP_NAME=LocalFlow`, `BUILD_ONLY=1`
(build `dist/` without installing). Logs from app launches go to
`~/.config/localflow/launcher.log`.

### Option B — CLI

```sh
brew install ollama ffmpeg
ollama pull qwen2.5:7b
uv venv && uv pip install -e ".[dev]"
```

## Run

Option A: launch **Whisper** from Spotlight — the floating widget appears
immediately and the global hotkey is live. The speech model loads in the
background (the pill's label reads "loading…" until it is ready). The first
launch opens a short tour of the controls (click through or skip it; replay
it any time via right-click → **Help**). Option B:

```sh
localflow            # floating widget + global hotkey
localflow --cli      # terminal mode: record until Enter, then transcribe
localflow --cli --duration 5
```

## Window controls

The widget is a background overlay: it floats above every app and Space,
never steals focus from the app you are dictating into, and has no Dock icon
or Cmd+Tab entry. Hovering over the pill reveals the **traffic light
buttons** in its top-left corner:

- **Red (close)** — quit the app
- **Green (zoom)** — reserved for the fixed-size widget (size does not change)

**Right-click the pill** to change microphone, ASR model, cleanup settings,
replay the onboarding tour (**Help**), and more. You can also drag the pill to
the **bottom-right corner** of the screen to quit.

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

## Uninstall

```sh
./scripts/build_app.sh --uninstall
```

Undoes everything the installer set up with two questions. The first
(default yes) removes the app's own resources: `Whisper.app`, the pulled
cleanup model, the downloaded Whisper models, `~/.config/localflow`
(settings, dictionary, logs), the repo's virtualenv, and the privacy
permissions macOS recorded for the app. The second (default no) additionally
uninstalls the shared tools ollama and ffmpeg, which other apps may use.
`ASSUME_YES=1` answers yes to both. Homebrew itself and the cloned
repository are left in place.

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
