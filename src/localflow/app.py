"""Application controller and entry points for localflow.

FlowController owns the configuration, dictionary, transcriber, cleaner, the
current recording session, and UI state. It wires the global hotkey and the
widget gestures to the same start/stop/toggle entry points, guarded by a lock,
and runs the transcription pipeline on a worker thread.

This module is deliberately importable on a headless machine: it never imports
AppKit or the widget module at top level. GUI imports happen only inside the GUI
branch of main(), mirroring the optional-import pattern in the contract.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import threading
import time

from localflow.asr import AsrError, Transcriber, create_transcriber
from localflow.capture import (
    CaptureError,
    RecordingSession,
    prefetch_audio_devices,
)
from localflow.cleanup import CleanupRequest, OllamaCleaner, should_clean
from localflow.config import (
    DICTIONARY_PATH,
    LOG_PATH,
    AppConfig,
    setup_logging,
)
from localflow.dictionary import Dictionary
from localflow.hotkey import HotkeyListener, HotkeySpec, TapDecisionEngine
from localflow.insert import (
    get_frontmost_app,
    insert_text,
    notify,
    tone_for_app,
)

log = logging.getLogger(__name__)

APP_NAME = "localflow"


def process_transcript(
    text: str,
    config: AppConfig,
    dictionary: Dictionary,
    cleaner: OllamaCleaner,
    tone: str,
) -> str:
    """Turn a raw transcript into the final text to insert.

    Applies the cleanup length gate, runs the LLM cleaner when cleanup is
    enabled and the transcript is long enough, then applies dictionary
    replacements. This is a pure function of its inputs.

    Args:
        text: Raw transcript produced by the ASR engine.
        config: Active application configuration.
        dictionary: User dictionary providing bias words and replacements.
        cleaner: Ollama-backed cleaner used when the gate passes.
        tone: Cleanup tone, "standard" or "casual".

    Returns:
        The cleaned, replacement-applied text.
    """
    result = text
    if config.cleanup_enabled and should_clean(result, config.cleanup_min_chars):
        request = CleanupRequest(
            text=result,
            tone=tone,
            dictionary_words=tuple(dictionary.words),
        )
        result = cleaner.clean(request)
    return dictionary.apply_replacements(result)


class _NullUI:
    """No-op UI used in CLI mode so the controller can run without a widget."""

    def showRecording(self) -> None:
        pass

    def showTranscribing(self) -> None:
        pass

    def showSuccess(self) -> None:
        pass

    def showError(self) -> None:
        pass

    def applyLevel_(self, level: float) -> None:
        pass

    def setMicName_(self, name: str) -> None:
        pass

    def setModelName_(self, name: str) -> None:
        pass

    def applyModelName_(self, name: str) -> None:
        pass


class FlowController:
    """Coordinates capture, transcription, cleanup, and insertion.

    Owns all mutable application state and serializes the start/stop/toggle
    entry points with a lock. The heavy pipeline work runs on a worker thread so
    the UI stays responsive.
    """

    def __init__(
        self,
        config: AppConfig,
        dictionary: Dictionary,
        transcriber: Transcriber | None,
        cleaner: OllamaCleaner,
    ) -> None:
        """Initialize the controller.

        Args:
            config: Application configuration.
            dictionary: User dictionary for biasing and replacements.
            transcriber: ASR transcriber, or None to create it lazily on the
                warm-up thread so the UI can appear before the heavy backend
                import (the widget runs either way; dictation waits until the
                model is ready).
            cleaner: Ollama-backed text cleaner.
        """
        self.__config = config
        self.__dictionary = dictionary
        self.__transcriber = transcriber
        self.__transcriber_failed = False
        self.__cleaner = cleaner
        self.__ui: object = _NullUI()
        self.__lock = threading.Lock()
        self.__listener: HotkeyListener | None = None
        self.__session: RecordingSession | None = None
        self.__recording = False
        self.__handsfree = False
        self.__frontmost = None

    @property
    def config(self) -> AppConfig:
        """Return the active application configuration."""
        return self.__config

    def attach_ui(self, ui: object) -> None:
        """Attach the widget the controller should drive.

        Args:
            ui: An object exposing the widget state methods (showRecording,
                showTranscribing, showSuccess, showError, applyLevel_,
                setMicName_, setModelName_, applyModelName_).
        """
        self.__ui = ui

    def start(self) -> None:
        """Warm the models in the background and start the global hotkey.

        Also prefetches the microphone list so the first right-click menu
        opens instantly instead of waiting on ffmpeg device enumeration.
        """
        threading.Thread(target=self.__warm_transcriber, daemon=True).start()
        threading.Thread(target=self.__warm_cleaner, daemon=True).start()
        prefetch_audio_devices()
        self.__start_hotkey()

    def __start_hotkey(self) -> None:
        hotkey = self.__config.hotkey
        try:
            spec = HotkeySpec.from_string(hotkey)
        except ValueError:
            log.exception("Invalid hotkey %r; hotkey disabled", hotkey)
            notify(APP_NAME, f"Invalid hotkey '{hotkey}'; hotkey disabled")
            return
        engine = TapDecisionEngine(
            on_hold_start=self.push_to_talk_start,
            on_hold_end=self.push_to_talk_stop,
            on_single_tap=self.single_tap_stop,
            on_double_tap=self.handsfree_toggle,
        )
        listener = HotkeyListener(spec, engine)
        if not listener.start():
            log.warning("Hotkey listener failed to start; permissions missing")
            notify(
                APP_NAME,
                "Global hotkey disabled until Accessibility and Input Monitoring "
                "are granted in System Settings.",
            )
            return
        self.__listener = listener

    def __warm_transcriber(self) -> None:
        """Create the transcriber if needed and load it, off the UI thread."""
        transcriber = self.__transcriber
        if transcriber is None:
            self.__ui.applyModelName_("loading…")
            try:
                transcriber = create_transcriber(
                    self.__config.asr_backend, self.__config.asr_model
                )
            except AsrError:
                log.exception("Could not initialize ASR backend")
                self.__transcriber_failed = True
                self.__ui.applyModelName_(self.__config.asr_model)
                notify(
                    APP_NAME, "No ASR backend available. Install an engine to dictate."
                )
                return
            with self.__lock:
                self.__transcriber = transcriber
        try:
            transcriber.load()
        except Exception:
            log.exception("Transcriber warm-up failed")
        self.__ui.applyModelName_(self.__config.asr_model)

    def __warm_cleaner(self) -> None:
        try:
            self.__cleaner.warm_up()
        except Exception:
            log.exception("Cleaner warm-up failed")

    def push_to_talk_start(self) -> None:
        """Start a push-to-talk recording (hotkey hold / widget hold)."""
        with self.__lock:
            self.__begin(handsfree=False)

    def push_to_talk_stop(self) -> None:
        """Stop a push-to-talk recording and process it."""
        with self.__lock:
            self.__finish()

    def handsfree_toggle(self) -> None:
        """Toggle a hands-free recording (hotkey double-tap / widget click)."""
        with self.__lock:
            if self.__recording:
                self.__finish()
            else:
                self.__begin(handsfree=True)

    def single_tap_stop(self) -> None:
        """Stop only when a hands-free recording is currently active."""
        with self.__lock:
            if self.__recording and self.__handsfree:
                self.__finish()

    def __begin(self, handsfree: bool) -> None:
        if self.__recording:
            return
        self.__frontmost = get_frontmost_app()
        session = RecordingSession(
            microphone_id=self.__config.microphone_id,
            max_seconds=self.__config.max_recording_seconds,
            on_level=self.__on_level,
            on_max_duration=self.__on_max_duration,
        )
        try:
            session.start()
        except CaptureError:
            log.exception("Failed to start recording")
            notify(APP_NAME, "Could not start recording. Is ffmpeg installed?")
            self.__ui.showError()
            return
        self.__session = session
        self.__recording = True
        self.__handsfree = handsfree
        self.__ui.showRecording()

    def __finish(self) -> None:
        if not self.__recording:
            return
        session = self.__session
        frontmost = self.__frontmost
        self.__recording = False
        self.__handsfree = False
        self.__session = None
        self.__ui.showTranscribing()
        threading.Thread(
            target=self.__run_pipeline,
            args=(session, frontmost),
            daemon=True,
        ).start()

    def __on_level(self, level: float) -> None:
        self.__ui.applyLevel_(level)

    def __on_max_duration(self) -> None:
        with self.__lock:
            self.__finish()

    def __run_pipeline(self, session: RecordingSession, frontmost) -> None:
        try:
            pcm = session.stop()
            if pcm is None or pcm.size == 0:
                log.warning("Empty recording; nothing to transcribe")
                self.__ui.showError()
                return
            transcriber = self.__transcriber
            if transcriber is None:
                if self.__transcriber_failed:
                    notify(APP_NAME, "No ASR model loaded. Pick one from the menu.")
                else:
                    notify(
                        APP_NAME,
                        "The speech model is still loading - try again in a moment.",
                    )
                self.__ui.showError()
                return
            text = transcriber.transcribe(
                pcm, self.__config.language, self.__dictionary.initial_prompt()
            ).strip()
            if not text:
                log.info("No speech detected")
                self.__ui.showError()
                return
            tone = tone_for_app(frontmost)
            final = process_transcript(
                text, self.__config, self.__dictionary, self.__cleaner, tone
            ).strip()
            if not final:
                self.__ui.showError()
                return
            outcome = insert_text(final, self.__config.auto_paste)
            log.info("Inserted %d chars (%s)", len(final), outcome.name)
            self.__ui.showSuccess()
        except Exception:
            log.exception("Pipeline failed")
            self.__ui.showError()

    def set_microphone(self, device_id: int, name: str) -> None:
        """Persist the selected microphone and update the widget label."""
        with self.__lock:
            self.__config.microphone_id = device_id
            self.__config.microphone_name = name
            self.__config.save()
        self.__ui.setMicName_(name)
        log.info("Microphone set to %s (id=%d)", name, device_id)

    def set_asr_model(self, name: str) -> None:
        """Persist the ASR model, then swap the transcriber in the background."""
        with self.__lock:
            self.__config.asr_model = name
            self.__config.save()
            backend = self.__config.asr_backend
        self.__ui.setModelName_(name)
        try:
            transcriber = create_transcriber(backend, name)
        except AsrError:
            log.exception("Could not create transcriber for model %s", name)
            notify(APP_NAME, f"Could not load ASR model '{name}'.")
            return
        threading.Thread(
            target=self.__swap_transcriber,
            args=(transcriber,),
            daemon=True,
        ).start()
        log.info("ASR model set to %s", name)

    def __swap_transcriber(self, transcriber: Transcriber) -> None:
        try:
            transcriber.load()
        except Exception:
            log.exception("Failed to load replacement transcriber")
            return
        with self.__lock:
            self.__transcriber = transcriber

    def toggle_cleanup(self) -> None:
        """Toggle the LLM cleanup pass and persist the change."""
        with self.__lock:
            self.__config.cleanup_enabled = not self.__config.cleanup_enabled
            enabled = self.__config.cleanup_enabled
            self.__config.save()
        log.info("Cleanup %s", "enabled" if enabled else "disabled")

    def set_cleanup_model(self, name: str) -> None:
        """Persist the cleanup model and rebuild the cleaner."""
        with self.__lock:
            self.__config.cleanup_model = name
            self.__config.save()
            self.__cleaner = OllamaCleaner(
                self.__config.ollama_url,
                name,
                self.__config.cleanup_timeout_seconds,
            )
        threading.Thread(target=self.__warm_cleaner, daemon=True).start()
        log.info("Cleanup model set to %s", name)

    def toggle_auto_paste(self) -> None:
        """Toggle auto-paste and persist the change."""
        with self.__lock:
            self.__config.auto_paste = not self.__config.auto_paste
            enabled = self.__config.auto_paste
            self.__config.save()
        log.info("Auto-paste %s", "enabled" if enabled else "disabled")

    def open_dictionary(self) -> None:
        """Open the dictionary file, creating an empty one if it is missing."""
        if not DICTIONARY_PATH.exists():
            Dictionary().save()
        subprocess.run(["open", "-e", str(DICTIONARY_PATH)], check=False)

    def open_log(self) -> None:
        """Open the application log in the default viewer."""
        subprocess.run(["open", str(LOG_PATH)], check=False)


def run_cli(duration: float | None, no_paste: bool) -> int:
    """Run one dictation cycle without any GUI.

    Records until Enter is pressed, or for a fixed duration, runs the same
    pipeline as the widget, prints the final text, and inserts it unless
    disabled.

    Args:
        duration: Fixed recording length in seconds, or None to wait for Enter.
        no_paste: When True, do not insert the text into the focused app.

    Returns:
        Process exit code.
    """
    config = AppConfig.load()
    dictionary = Dictionary.load()
    try:
        transcriber = create_transcriber(config.asr_backend, config.asr_model)
    except AsrError:
        log.exception("Could not initialize ASR backend")
        print("Error: could not initialize the ASR backend.", file=sys.stderr)
        return 1
    print("Loading ASR model (first run downloads it)...", file=sys.stderr)
    transcriber.load()
    cleaner = OllamaCleaner(
        config.ollama_url, config.cleanup_model, config.cleanup_timeout_seconds
    )

    frontmost = get_frontmost_app()
    session = RecordingSession(
        microphone_id=config.microphone_id,
        max_seconds=config.max_recording_seconds,
    )
    try:
        session.start()
    except CaptureError:
        log.exception("Could not start recording")
        print("Error: could not start recording. Is ffmpeg installed?", file=sys.stderr)
        return 1

    if duration is not None:
        print(f"Recording for {duration:.0f}s...", file=sys.stderr)
        time.sleep(duration)
    else:
        print("Recording... press Enter to stop.", file=sys.stderr)
        try:
            input()
        except EOFError:
            pass

    pcm = session.stop()
    print("Transcribing...", file=sys.stderr)
    text = transcriber.transcribe(
        pcm, config.language, dictionary.initial_prompt()
    ).strip()
    tone = tone_for_app(frontmost)
    final = process_transcript(text, config, dictionary, cleaner, tone).strip()

    print(final)
    if final and not no_paste:
        insert_text(final, config.auto_paste)
    return 0


def run_gui() -> int:
    """Launch the widget application. Imports the GUI layer lazily.

    The transcriber is not created here: the heavy backend import and model
    load happen on the controller's warm-up thread so the widget appears
    immediately.

    Returns:
        Process exit code.
    """
    from localflow import widget

    config = AppConfig.load()
    dictionary = Dictionary.load()
    cleaner = OllamaCleaner(
        config.ollama_url, config.cleanup_model, config.cleanup_timeout_seconds
    )
    controller = FlowController(config, dictionary, None, cleaner)
    widget.run_app(controller)
    return 0


def main() -> int:
    """Parse arguments and dispatch to the widget or CLI entry point.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description="Local dictation: hold a key, speak, cleaned text is inserted.",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run one dictation cycle in the terminal with no GUI.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="CLI mode: record this many seconds instead of waiting for Enter.",
    )
    parser.add_argument(
        "--no-paste",
        action="store_true",
        help="CLI mode: print the text but do not insert it.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    if args.cli or args.duration is not None:
        return run_cli(duration=args.duration, no_paste=args.no_paste)
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
