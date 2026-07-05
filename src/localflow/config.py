"""Configuration and logging for localflow."""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "localflow"
CONFIG_PATH = CONFIG_DIR / "config.json"
LOG_PATH = CONFIG_DIR / "localflow.log"
DICTIONARY_PATH = CONFIG_DIR / "dictionary.json"

log = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """User-facing settings persisted to the config file.

    Attributes:
        language: ISO language code passed to the ASR engine.
        asr_backend: ASR engine to use, "mlx" or "whispercpp".
        asr_model: Whisper model name, e.g. "large-v3-turbo".
        cleanup_enabled: Whether to run the Ollama cleanup pass.
        cleanup_model: Ollama model tag used for cleanup.
        cleanup_min_chars: Minimum transcript length that triggers cleanup.
        cleanup_timeout_seconds: Ollama request timeout before falling back
            to the raw transcript. Generous by default: it is a safety net,
            not the typical latency, and long utterances need decode time on
            modest hardware.
        ollama_url: Base URL of the Ollama server.
        microphone_id: avfoundation audio device index.
        microphone_name: Display name of the selected microphone.
        auto_paste: Whether to paste into the focused app or only copy.
        hotkey: Global hotkey spec, e.g. "fn" or "ctrl+alt+space".
        max_recording_seconds: Hard cap on a single recording.
        window_x: Persisted widget window x origin.
        window_y: Persisted widget window y origin.
    """

    language: str = "en"
    asr_backend: str = "mlx"
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
    def load(cls) -> AppConfig:
        """Load the config from disk, falling back to defaults on any problem.

        Returns:
            The persisted configuration, or a default instance when the file
            is missing, unreadable, or contains unknown/invalid content.
        """
        try:
            raw = json.loads(CONFIG_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        """Persist the config, creating the config directory if needed."""
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(json.dumps(dataclasses.asdict(self), indent=2))
        except OSError:
            log.exception("Failed to save config")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure file logging under the config directory.

    Args:
        level: Root log level for the application log file.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_PATH),
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
