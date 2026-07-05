# localflow — module contract

Local dictation for macOS (Apple Silicon): hold a key, speak, release; local Whisper
transcribes, a local LLM on Ollama cleans the text, and the result is inserted into the
focused app. This document is the binding interface contract between modules. Implement
exactly these signatures; do not rename or restructure without updating this file.

Reference implementation being replaced: `/Users/GregEzema/bin/whisper-widget.py` and
`/Users/GregEzema/bin/whisper-speak` (read them for the widget UI, gesture handling, and
ffmpeg capture patterns — the pill UI and gesture model are being kept).

## Pipeline

```
hotkey press ──► snapshot frontmost app ──► RecordingSession.start()
hotkey release ─► pcm = session.stop() ──► Transcriber.transcribe(pcm, language, initial_prompt=dictionary)
                 ──► if cleanup_enabled and should_clean(text): OllamaCleaner.clean(...)
                 ──► Dictionary.apply_replacements(text)
                 ──► insert_text(text)  (clipboard save → paste → restore; fallback notify)
```

State machine (UI states, reused from the old widget): IDLE → RECORDING → TRANSCRIBING →
SUCCESS | ERROR → IDLE.

## Coding standards (binding)

- Python 3.10+ typing (`str | None`, `list[str]`); type hints on all public functions.
- Google-style docstrings on public classes/functions; no markdown in docstrings.
- Top-level imports only (exception: guarded optional imports via `try/except ImportError`
  at module top for `mlx_whisper` / `pywhispercpp`).
- Small `try` blocks, specific exceptions, `log.exception()` in handlers.
- `log = logging.getLogger(__name__)` per module; logging is configured centrally by
  `config.setup_logging()`.
- Double-underscore for private instance attributes in plain Python classes (PyObjC
  subclasses use `objc.ivar()` and are exempt).
- Tests: pytest, placed NEXT to the module (`src/localflow/test_<module>.py`). No network,
  no audio hardware, no GUI, no permissions in tests — mock `subprocess`, `requests`, and
  PyObjC side effects. Pure logic should be factored so it is testable without mocks.

## Modules and ownership

Each module is owned by exactly one implementer. Do not edit files you do not own.

| File | Owner |
|---|---|
| `config.py` | done (do not edit) |
| `capture.py`, `test_capture.py` | agent A |
| `cleanup.py`, `test_cleanup.py`, `dictionary.py`, `test_dictionary.py` | agent B |
| `asr.py`, `test_asr.py`, `scripts/benchmark_asr.py` | agent C |
| `insert.py`, `test_insert.py` | agent D |
| `hotkey.py`, `test_hotkey.py` | agent E |
| `widget.py`, `app.py`, `__main__.py` | agent F |

## config.py (already implemented — read it, import from it)

```python
CONFIG_DIR: Path        # ~/.config/localflow
CONFIG_PATH: Path       # config.json
LOG_PATH: Path          # localflow.log
DICTIONARY_PATH: Path   # dictionary.json

@dataclass
class AppConfig:
    language: str = "en"
    asr_backend: str = "mlx"            # "mlx" | "whispercpp"
    asr_model: str = "large-v3-turbo"
    cleanup_enabled: bool = True
    cleanup_model: str = "qwen2.5:7b"
    cleanup_min_chars: int = 50
    cleanup_timeout_seconds: float = 8.0
    ollama_url: str = "http://localhost:11434"
    microphone_id: int = 0
    microphone_name: str = "Default"
    auto_paste: bool = True
    hotkey: str = "fn"
    max_recording_seconds: int = 1200
    window_x: int = 100
    window_y: int = 100

    @classmethod
    def load(cls) -> "AppConfig": ...
    def save(self) -> None: ...

def setup_logging(level: int = logging.INFO) -> None: ...
```

## capture.py (agent A)

ffmpeg avfoundation capture to an in-memory PCM buffer. 16 kHz, mono, s16le piped to
stdout (same ffmpeg invocation pattern as the old widget). Each `RecordingSession` is
single-use and owns its own buffer and process — no shared mutable state across sessions.

```python
SAMPLE_RATE = 16000

class CaptureError(Exception):
    """Raised when recording cannot start (e.g. ffmpeg missing)."""

@dataclass
class AudioDevice:
    id: int
    name: str

def list_audio_devices() -> list[AudioDevice]:
    """Parse `ffmpeg -f avfoundation -list_devices true -i ''` stderr; fall back to
    [AudioDevice(0, "Default")] on any failure."""

class RecordingSession:
    def __init__(
        self,
        microphone_id: int = 0,
        max_seconds: int = 1200,
        on_level: Callable[[float], None] | None = None,
        on_max_duration: Callable[[], None] | None = None,
    ) -> None: ...
    def start(self) -> None:
        """Spawn ffmpeg and the reader thread. Raises CaptureError if ffmpeg is missing."""
    def stop(self) -> np.ndarray:
        """Terminate ffmpeg, join the reader thread, and return float32 mono PCM in
        [-1, 1] at 16 kHz. Idempotent; second call returns the same array."""
    @property
    def duration_seconds(self) -> float: ...
    @property
    def is_recording(self) -> bool: ...
```

Details: reader thread reads 3200-byte chunks (100 ms), appends to a private buffer,
computes RMS per chunk and calls `on_level(level)` with level in [0, 1] (RMS / 3000,
clamped — same scale as the old widget). When the buffer reaches
`max_seconds * SAMPLE_RATE * 2` bytes, stop collecting, terminate ffmpeg, and call
`on_max_duration()` once (from the reader thread; callers marshal to their own threads).
`on_level` callbacks must never raise out of the reader thread.

Tests: mock `subprocess.Popen` with a fake stdout pipe; verify PCM conversion (known
s16le bytes → expected float32), level callbacks, max-duration cutoff, idempotent stop,
device-list parsing from a captured ffmpeg stderr sample, and ffmpeg-missing error.

## asr.py (agent C)

```python
class AsrError(Exception):
    """Raised when a backend is unavailable or transcription fails."""

class Transcriber(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load model weights and run a short warmup so the first real call is fast."""
    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en",
        initial_prompt: str | None = None,
    ) -> str:
        """Transcribe float32 mono 16 kHz PCM in [-1, 1]; return stripped text."""

class MlxWhisperTranscriber(Transcriber):
    def __init__(self, model_name: str = "large-v3-turbo") -> None: ...

class WhisperCppTranscriber(Transcriber):
    def __init__(self, model_name: str = "large-v3-turbo") -> None: ...

def create_transcriber(backend: str, model_name: str) -> Transcriber:
    """Return the transcriber for backend "mlx" or "whispercpp"; raise AsrError for an
    unknown backend or one whose package is not installed."""
```

MLX model names map to HF repos: `large-v3-turbo`/`turbo` →
`mlx-community/whisper-large-v3-turbo`, `small` → `mlx-community/whisper-small-mlx`,
`base` → `mlx-community/whisper-base-mlx`. `mlx_whisper.transcribe()` accepts a numpy
array plus `path_or_hf_repo`, `language`, `initial_prompt`. Guard both backend imports
with `try/except ImportError` at module top; `load()` warms up on ~0.5 s of zeros.
pywhispercpp's `Model.transcribe` accepts a numpy array; join segment texts.

`scripts/benchmark_asr.py`: standalone script (argparse). Input: one or more WAV paths,
or `--record N` to capture N seconds via `localflow.capture`. For each available backend
(mlx, whispercpp — skip gracefully if not installed): measure `load()` time, then per
file measure wall-clock `transcribe()` over `--runs` (default 3) and print a table
(backend, model, file, audio seconds, mean/min seconds, realtime factor) plus each
transcript. Exit nonzero if no backend is available.

Tests: model-name→repo mapping, `create_transcriber` dispatch and error paths (simulate
missing packages via monkeypatch), audio dtype/shape validation. Mock the underlying
libraries; never download models in tests.

## cleanup.py (agent B)

```python
FEW_SHOT_EXAMPLES: tuple[tuple[str, str, str], ...]  # (tone, raw, cleaned)

@dataclass
class CleanupRequest:
    text: str
    tone: str = "standard"              # "standard" | "casual"
    dictionary_words: tuple[str, ...] = ()

class OllamaCleaner:
    def __init__(self, url: str, model: str, timeout_seconds: float = 4.0) -> None: ...
    def warm_up(self) -> bool:
        """Load the model into memory with a trivial request; True on success."""
    def clean(self, request: CleanupRequest) -> str:
        """Return cleaned text. NEVER raises; on any failure (connection, timeout,
        HTTP error, malformed response) logs and returns request.text unchanged."""

def should_clean(text: str, min_chars: int = 50) -> bool:
    """True when the transcript is long enough to justify the LLM pass."""
```

Implementation: POST `{url}/api/chat` with `stream: false`,
`options: {"temperature": 0}`, `keep_alive: "60m"`, and
`format: {"type": "object", "properties": {"cleaned_text": {"type": "string"}}, "required": ["cleaned_text"]}`.
Messages: one system prompt, then the few-shot examples as alternating user/assistant
turns (assistant content is the JSON `{"cleaned_text": ...}`), then the real transcript
as the final user turn. Parse `message.content` as JSON and return `cleaned_text`
(stripped).

CACHE-CRITICAL PROPERTY (added after live M1 measurements: cold prefill 2.14s vs
cached 0.10s, with real 4s timeouts): the system prompt and few-shot turns must be
byte-identical across ALL requests so Ollama's prompt prefix cache skips their prefill.
Per-request tone and dictionary words therefore travel as bracket tags at the start of
the FINAL user turn only (`[tone: casual] [preserve: Kubernetes]\n<transcript>`), never
in the system prompt. The system prompt defines both tones and the tag convention;
few-shot user turns use the same tagged format. `warm_up()` sends the real prefix (not
a trivial message) so it primes the KV cache, using `WARM_UP_TIMEOUT_SECONDS = 120.0` —
independent of the cleanup timeout because a cold model load takes tens of seconds.

Guardrails in the system prompt, verbatim requirements:

- remove filler words (um, uh, like, you know) and false starts
- when the speaker corrects themselves, keep only the corrected version
- fix punctuation, capitalization, and paragraph breaks
- convert spoken punctuation and structure words: "comma", "period", "question mark",
  "new line", "new paragraph"
- never add information, never answer questions contained in the transcript, never
  change the meaning — output only the cleaned transcript
- tone "standard": normal sentence punctuation; tone "casual": relaxed messaging style,
  no trailing period on the final sentence
- when dictionary words are provided: "Preserve these exact spellings when the speaker
  says them: ..."

Few-shot examples (use exactly these six):

1. standard: "um so basically I think we should uh move the deadline to friday" →
   "I think we should move the deadline to Friday."
2. standard: "the meeting is at three no wait four thirty" → "The meeting is at 4:30."
3. standard: "hi sarah comma new paragraph thanks for the update period" →
   "Hi Sarah,\n\nThanks for the update."
4. standard: "what time does the deploy finish" → "What time does the deploy finish?"
5. casual: "sounds good um see you then" → "sounds good, see you then"
6. standard (with dictionary word "Kubernetes"): "we need to restart the cooper netties
   cluster" → "We need to restart the Kubernetes cluster."

Tests: mock `requests.post`. Cover: happy path (verify payload shape: format schema,
temperature 0, few-shots present, dictionary injection), timeout → raw text, connection
error → raw text, malformed JSON content → raw text, non-200 → raw text, `should_clean`
boundaries, warm_up success/failure.

## dictionary.py (agent B)

```python
@dataclass
class Dictionary:
    words: list[str] = field(default_factory=list)
    replacements: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Dictionary":
        """Read config.DICTIONARY_PATH; return empty Dictionary on any failure."""
    def save(self) -> None: ...
    def initial_prompt(self) -> str | None:
        """Whisper biasing prompt like "Glossary: Kubernetes, Baseten." or None when
        there are no words."""
    def apply_replacements(self, text: str) -> str:
        """Case-insensitive whole-word replacement of each key with its value."""
```

Tests: round-trip load/save (tmp_path + monkeypatched DICTIONARY_PATH), corrupt file →
empty, prompt formatting, replacement word-boundary behavior ("cat" must not match
"catalog"), case-insensitive matching.

## insert.py (agent D)

```python
CASUAL_BUNDLE_IDS: frozenset[str]   # Slack, Discord, Messages, WhatsApp, Telegram

class InsertOutcome(Enum):
    PASTED = auto()
    CLIPBOARD_ONLY = auto()

@dataclass
class FrontmostApp:
    bundle_id: str
    name: str

def get_frontmost_app() -> FrontmostApp | None:
    """NSWorkspace frontmost application; None if unavailable."""

def tone_for_app(app: FrontmostApp | None) -> str:
    """"casual" for messaging apps, else "standard". Pure function — no side effects."""

def is_accessibility_trusted() -> bool:
    """AXIsProcessTrusted()."""

def is_secure_input_active() -> bool:
    """IsSecureEventInputEnabled via ctypes on the Carbon framework; False on failure."""

def insert_text(text: str, auto_paste: bool = True) -> InsertOutcome:
    """Insert text into the focused app.

    Ladder: if not auto_paste, accessibility is not trusted, or secure input is active →
    put text on the clipboard, notify the user, return CLIPBOARD_ONLY. Otherwise save the
    current clipboard string, write text to the clipboard (NSPasteboard), post Cmd-V via
    CGEvent, and restore the saved clipboard ~0.6 s later on a timer thread (only if
    there was a saved string). Return PASTED."""

def notify(title: str, message: str) -> None:
    """User notification via osascript display notification; never raises."""
```

Bundle ids: `com.tinyspeck.slackmacgap`, `com.hnc.Discord`, `com.apple.MobileSMS`,
`net.whatsapp.WhatsApp`, `ru.keepcoder.Telegram`. Use the CGEvent Cmd-V pattern from the
old widget (`CGEventCreateKeyboardEvent`, `kCGEventFlagMaskCommand`, `kCGHIDEventTap`).
Factor the ladder decision into a pure helper
`resolve_strategy(auto_paste: bool, trusted: bool, secure_input: bool) -> InsertOutcome`
so it is unit-testable.

Tests: `tone_for_app` for each casual bundle id + default + None; `resolve_strategy` all
branches; `insert_text` with monkeypatched pasteboard/CGEvent/notify verifying clipboard
save→write→paste→restore ordering (use a fake timer that fires synchronously) and the
CLIPBOARD_ONLY paths.

## hotkey.py (agent E)

```python
@dataclass(frozen=True)
class HotkeySpec:
    keycode: int
    modifiers: frozenset[str]      # subset of {"cmd", "ctrl", "alt", "shift"}
    is_modifier_key: bool          # True for bare-modifier hotkeys like fn

    @classmethod
    def from_string(cls, spec: str) -> "HotkeySpec":
        """Parse "fn", "right_cmd", "right_alt", or combos like "ctrl+alt+space".
        Raise ValueError for unknown keys."""

class TapDecisionEngine:
    """Pure hold/tap/double-tap state machine — no Quartz imports, fully unit-testable.

    Feed it (event_type: "down"|"up", timestamp: float) transitions; it fires callbacks:
    on_hold_start when a press is held >= hold_threshold (fired by a timer check),
    on_hold_end on release after a hold, on_single_tap when a press+release shorter than
    hold_threshold is not followed by a second press within double_tap_window,
    on_double_tap for two short presses within double_tap_window.
    """
    def __init__(
        self,
        on_hold_start: Callable[[], None],
        on_hold_end: Callable[[], None],
        on_single_tap: Callable[[], None],
        on_double_tap: Callable[[], None],
        hold_threshold: float = 0.35,
        double_tap_window: float = 0.4,
        timer_factory: Callable[[float, Callable[[], None]], object] | None = None,
    ) -> None: ...
    def key_down(self, timestamp: float) -> None: ...
    def key_up(self, timestamp: float) -> None: ...

class HotkeyListener:
    """CGEventTap wiring: translates system key events for the configured hotkey into
    TapDecisionEngine transitions."""
    def __init__(self, spec: HotkeySpec, engine: TapDecisionEngine) -> None: ...
    def start(self) -> bool:
        """Create the event tap on the main run loop. False if creation failed
        (missing Accessibility/Input Monitoring permission)."""
    def stop(self) -> None: ...
```

Quartz details: tap at `kCGHIDEventTap` with `kCGEventTapOptionDefault`, mask for
keyDown | keyUp | flagsChanged. Fn is keycode 63 and arrives as flagsChanged — derive
down/up from the `kCGEventFlagMaskSecondaryFn` bit (0x800000). right_cmd is keycode 54,
right_alt 61 (also flagsChanged; use the corresponding device-independent mask bits).
For combos (keyDown/keyUp of a normal key with required modifier flags) return None from
the callback to consume the event; flagsChanged modifier events are passed through.
Handle `kCGEventTapDisabledByTimeout`/`ByUserInput` by re-enabling the tap. Add the run
loop source to `CFRunLoopGetMain()`. Timer scheduling for hold detection uses
`threading.Timer` via the injectable `timer_factory` (real default), and callback
exceptions must be caught and logged.

Tests: exhaustive TapDecisionEngine coverage with a fake timer_factory (hold, single
tap, double tap, tap-then-hold, hold shorter than threshold, rapid sequences);
HotkeySpec.from_string parsing and error cases. No CGEventTap in tests.

## widget.py + app.py + __main__.py (agent F)

`widget.py`: port the pill widget UI from `/Users/GregEzema/bin/whisper-widget.py` —
keep the drawing code, states (IDLE/RECORDING/TRANSCRIBING/SUCCESS/ERROR), gesture model
(click toggle, press-and-hold, drag, drag-to-corner close), position persistence (via
AppConfig), and non-activating panel setup. Class names: `FlowWidget(NSView)`,
`MenuHandler(NSObject)`, `create_panel(...)`. The widget receives injected callbacks and
never owns pipeline logic. Menu items: Microphone submenu, ASR Model submenu ("base",
"small", "large-v3-turbo"), Cleanup toggle, Cleanup Model submenu ("qwen2.5:7b",
"llama3.2:3b"), Auto-paste toggle, Open Dictionary (ensure the file exists, then
`open -e`), Open Log, Quit. Status text shows the ASR model name when idle.

`app.py`: `FlowController` — owns AppConfig, Dictionary, Transcriber, OllamaCleaner,
current RecordingSession, and UI state; wires HotkeyListener callbacks (hold =
push-to-talk, double-tap = hands-free toggle, single tap = stop when hands-free
recording) and widget callbacks to the same start/stop/toggle entry points, guarded by a
lock. On start: snapshot `get_frontmost_app()` BEFORE showing any state change, then
start the session. On stop: run the pipeline in a worker thread (transcribe → optional
clean → apply_replacements → insert_text), update widget state via
`performSelectorOnMainThread`, notify on CLIPBOARD_ONLY. At startup, warm the
transcriber and cleaner in background threads. If `HotkeyListener.start()` returns
False, log + notify that the global hotkey is disabled until permissions are granted.

`main()` (argparse): default → widget app (NSApplication accessory policy, like the old
widget). `--cli` → no GUI: record until Enter (or `--duration N` seconds), run the same
pipeline, print the result, insert unless `--no-paste`. `--verbose` → DEBUG logging.
`__main__.py` calls `main()`.

Keep every UI-thread rule from the old widget (all AppKit mutations via
`performSelectorOnMainThread`). Import AppKit lazily *inside* the GUI entry path is NOT
allowed — instead put GUI imports at top of `widget.py`, and keep `app.py` importable
headless by importing widget only inside the GUI branch of `main()` (documented
exception to the top-level import rule, mirroring the optional-import pattern).

Tests are not required for widget.py; `app.py` pipeline logic should be structured so
the text-processing step (`transcript → final text`) is a pure-ish function
`process_transcript(text, config, dictionary, cleaner, tone) -> str` covered by
`test_app.py` with mocks.
