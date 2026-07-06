"""In-memory microphone capture via ffmpeg/avfoundation.

Each `RecordingSession` spawns its own ffmpeg subprocess and reader thread and
owns a private PCM buffer; sessions are single-use (create a new one for each
recording) and hold no state shared with other sessions.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_BYTES = 3200
RMS_SCALE = 3000.0

_AUDIO_SECTION_RE = re.compile(r"audio devices", re.IGNORECASE)
_DEVICE_LINE_RE = re.compile(r"\[(\d+)\]\s+(.+)")

log = logging.getLogger(__name__)


class CaptureError(Exception):
    """Raised when recording cannot start (e.g. ffmpeg missing)."""


@dataclass
class AudioDevice:
    """An avfoundation audio input device.

    Attributes:
        id: avfoundation device index, as passed to ffmpeg's `-i :<id>`.
        name: Human-readable device name.
    """

    id: int
    name: str


def list_audio_devices() -> list[AudioDevice]:
    """List available avfoundation audio input devices.

    Runs `ffmpeg -f avfoundation -list_devices true -i ''` and parses the
    device table out of its stderr output.

    Returns:
        The parsed audio devices, or `[AudioDevice(0, "Default")]` if ffmpeg
        is missing, times out, or its output cannot be parsed into at least
        one device.
    """
    fallback = [AudioDevice(0, "Default")]
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        log.warning("Could not list audio devices; falling back to default")
        return fallback

    devices: list[AudioDevice] = []
    in_audio_section = False
    for line in result.stderr.splitlines():
        if _AUDIO_SECTION_RE.search(line):
            in_audio_section = True
            continue
        if not in_audio_section:
            continue
        match = _DEVICE_LINE_RE.search(line)
        if match:
            devices.append(AudioDevice(int(match.group(1)), match.group(2).strip()))
    return devices or fallback


_devices_cache: list[AudioDevice] = []
_devices_cache_lock = threading.Lock()
_devices_refresh_thread: threading.Thread | None = None


def refresh_audio_devices() -> list[AudioDevice]:
    """Query ffmpeg for the device list and update the cache. Blocking.

    Returns:
        The freshly enumerated devices.
    """
    devices = list_audio_devices()
    with _devices_cache_lock:
        _devices_cache[:] = devices
    return devices


def prefetch_audio_devices() -> None:
    """Refresh the device cache on a background thread.

    At most one refresh runs at a time; concurrent calls while a refresh is
    in flight are no-ops.
    """
    global _devices_refresh_thread
    with _devices_cache_lock:
        thread = _devices_refresh_thread
        if thread is not None and thread.is_alive():
            return
        thread = threading.Thread(target=refresh_audio_devices, daemon=True)
        _devices_refresh_thread = thread
    thread.start()


def cached_audio_devices() -> list[AudioDevice]:
    """Return the last known devices immediately, refreshing in the background.

    Enumerating avfoundation devices through ffmpeg takes a couple of
    seconds, far too slow for a context menu. This returns the cached list
    right away (falling back to the default device when nothing has been
    cached yet) and kicks off a background refresh so the next call sees any
    newly plugged-in device.

    Returns:
        A snapshot of the cached devices, or `[AudioDevice(0, "Default")]`
        when the cache is still empty.
    """
    with _devices_cache_lock:
        snapshot = list(_devices_cache)
    prefetch_audio_devices()
    return snapshot or [AudioDevice(0, "Default")]


def _ffmpeg_capture_command(microphone_id: int) -> list[str]:
    """Build the ffmpeg invocation that streams 16 kHz mono s16le to stdout.

    Args:
        microphone_id: avfoundation device index to capture from.

    Returns:
        The argv list for `subprocess.Popen`.
    """
    return [
        "ffmpeg",
        "-f",
        "avfoundation",
        "-i",
        f":{microphone_id}",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-loglevel",
        "error",
        "pipe:1",
    ]


def _truncate_to_even(data: bytes) -> bytes:
    """Drop a trailing odd byte so `data` holds only whole s16le samples.

    Args:
        data: Raw little-endian signed 16-bit PCM bytes.

    Returns:
        `data`, or `data` minus its last byte if its length is odd.
    """
    remainder = len(data) % BYTES_PER_SAMPLE
    return data[:-remainder] if remainder else data


def _decode_pcm(data: bytes) -> np.ndarray:
    """Convert raw s16le bytes into float32 mono PCM in [-1, 1].

    Args:
        data: Raw little-endian signed 16-bit PCM bytes.

    Returns:
        A float32 array of the decoded samples. An odd trailing byte (a
        partial sample) is dropped.
    """
    data = _truncate_to_even(data)
    if not data:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(data, dtype=np.int16)
    return (samples.astype(np.float32) / 32768.0).astype(np.float32)


class RecordingSession:
    """A single, single-use microphone recording backed by ffmpeg.

    Attributes are private to the instance; no state is shared across
    sessions. Create a new `RecordingSession` for each recording.
    """

    def __init__(
        self,
        microphone_id: int = 0,
        max_seconds: int = 1200,
        on_level: Callable[[float], None] | None = None,
        on_max_duration: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the session.

        Args:
            microphone_id: avfoundation device index to capture from.
            max_seconds: Hard cap on recording length; once reached the
                session stops itself and fires `on_max_duration`.
            on_level: Called from the reader thread with each chunk's RMS
                level, scaled to [0, 1]. Exceptions from this callback are
                caught and logged, never propagated.
            on_max_duration: Called once, from the reader thread, when
                `max_seconds` of audio has been captured.
        """
        self.__microphone_id = microphone_id
        self.__max_bytes = max_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE
        self.__on_level = on_level
        self.__on_max_duration = on_max_duration

        self.__process: subprocess.Popen[bytes] | None = None
        self.__reader_thread: threading.Thread | None = None
        self.__buffer = bytearray()
        self.__lock = threading.Lock()
        self.__stop_lock = threading.Lock()
        self.__done = threading.Event()
        self.__started = False
        self.__stopped = False
        self.__result: np.ndarray | None = None

    def start(self) -> None:
        """Spawn ffmpeg and the reader thread.

        Raises:
            CaptureError: If ffmpeg cannot be spawned (e.g. not installed) or
                if the session has already been started.
        """
        if self.__started:
            raise CaptureError("RecordingSession.start() called more than once")
        command = _ffmpeg_capture_command(self.__microphone_id)
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError as exc:
            raise CaptureError("ffmpeg executable not found") from exc

        self.__process = process
        self.__started = True
        self.__reader_thread = threading.Thread(target=self.__read_loop, daemon=True)
        self.__reader_thread.start()
        log.info("Recording started (microphone_id=%d)", self.__microphone_id)

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio.

        Terminates the ffmpeg process (if still running), joins the reader
        thread, and decodes the buffered bytes. Safe to call multiple times;
        subsequent calls return the same array without touching ffmpeg again.

        Returns:
            Float32 mono PCM in [-1, 1] at `SAMPLE_RATE`.
        """
        with self.__stop_lock:
            if not self.__stopped:
                self.__stopped = True
                self.__terminate_process()
                if self.__reader_thread is not None:
                    self.__reader_thread.join()
                with self.__lock:
                    pcm_bytes = bytes(self.__buffer)
                self.__result = _decode_pcm(pcm_bytes)
                log.info("Recording stopped (%.2fs captured)", self.duration_seconds)

            assert self.__result is not None
            return self.__result

    @property
    def duration_seconds(self) -> float:
        """Seconds of audio captured so far."""
        with self.__lock:
            length = len(self.__buffer)
        return length / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    @property
    def is_recording(self) -> bool:
        """Whether the session is currently capturing audio."""
        return self.__started and not self.__done.is_set()

    def __read_loop(self) -> None:
        """Reader thread body: pull chunks from ffmpeg stdout into the buffer."""
        assert self.__process is not None
        assert self.__process.stdout is not None
        stdout = self.__process.stdout
        max_duration_reached = False
        try:
            while True:
                chunk = stdout.read(CHUNK_BYTES)
                if not chunk:
                    break
                with self.__lock:
                    self.__buffer.extend(chunk)
                    overflow = len(self.__buffer) - self.__max_bytes
                    if overflow >= 0:
                        if overflow > 0:
                            del self.__buffer[-overflow:]
                        max_duration_reached = True
                self.__emit_level(chunk)
                if max_duration_reached:
                    break
        except (OSError, ValueError):
            log.exception("Error reading ffmpeg audio stream")
        finally:
            self.__done.set()
            if max_duration_reached:
                self.__terminate_process()
                self.__fire_max_duration()

    def __emit_level(self, chunk: bytes) -> None:
        """Compute the RMS level of a chunk and invoke `on_level` with it.

        Args:
            chunk: Raw s16le bytes read from ffmpeg stdout.
        """
        if self.__on_level is None:
            return
        try:
            usable = _truncate_to_even(chunk)
            if not usable:
                return
            samples = np.frombuffer(usable, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(np.square(samples))))
            level = min(rms / RMS_SCALE, 1.0)
            self.__on_level(level)
        except Exception:
            # on_level is caller-supplied; it must never crash the reader
            # thread, so any exception it raises is caught here.
            log.exception("on_level callback raised")

    def __fire_max_duration(self) -> None:
        """Invoke `on_max_duration`, guarding against exceptions."""
        if self.__on_max_duration is None:
            return
        try:
            self.__on_max_duration()
        except Exception:
            # Same rationale as __emit_level: never crash the reader thread.
            log.exception("on_max_duration callback raised")

    def __terminate_process(self) -> None:
        """Terminate the ffmpeg process if it is still running."""
        process = self.__process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
