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

```sh
uv venv && uv pip install -e ".[dev]"
brew install ollama ffmpeg
ollama pull qwen2.5:7b
```

## Run

```sh
localflow            # floating widget + global hotkey
localflow --cli      # terminal mode: record until Enter, then transcribe
localflow --cli --duration 5
```

## Permissions (one-time)

System Settings → Privacy & Security:

- **Microphone** — for your terminal/Python on first recording.
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
