"""Tests for localflow.capture.

No real ffmpeg process or audio hardware is used: `subprocess.Popen`/`.run`
are monkeypatched with fakes that hand back scripted stdout data.
"""

from __future__ import annotations

import struct
import subprocess
import time
from collections.abc import Callable

import numpy as np
import pytest

from localflow import capture
from localflow.capture import AudioDevice, CaptureError, RecordingSession

# A real ffmpeg avfoundation device listing, captured from `ffmpeg -f
# avfoundation -list_devices true -i ''` stderr on macOS.
SAMPLE_DEVICE_STDERR = """\
[AVFoundation indev @ 0x155705000] AVFoundation video devices:
[AVFoundation indev @ 0x155705000] [0] FaceTime HD Camera
[AVFoundation indev @ 0x155705000] [1] Capture screen 0
[AVFoundation indev @ 0x155705000] AVFoundation audio devices:
[AVFoundation indev @ 0x155705000] [0] MacBook Pro Microphone
[AVFoundation indev @ 0x155705000] [1] External Headset
"""


def _s16le(values: list[int]) -> bytes:
    """Pack signed 16-bit sample values into little-endian bytes."""
    return struct.pack(f"<{len(values)}h", *values)


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    """Poll `predicate` until it is true, failing the test if it times out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Condition not met within timeout")


class FakeStdout:
    """Fake ffmpeg stdout pipe that yields a scripted sequence of chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    def read(self, size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class FakeProcess:
    """Fake `subprocess.Popen` handle for ffmpeg."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.stdout = FakeStdout(chunks)
        self.terminated = False
        self.terminate_calls = 0
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self.terminate_calls += 1
        self._returncode = 0

    def kill(self) -> None:
        self._returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        assert self._returncode is not None
        return self._returncode


def _make_session(
    monkeypatch: pytest.MonkeyPatch, chunks: list[bytes], **kwargs: object
) -> tuple[RecordingSession, FakeProcess]:
    """Create and start a RecordingSession backed by a FakeProcess."""
    fake_process = FakeProcess(chunks)
    monkeypatch.setattr(capture.subprocess, "Popen", lambda *a, **kw: fake_process)
    session = RecordingSession(**kwargs)
    session.start()
    return session, fake_process


# -- PCM decoding -------------------------------------------------------


def test_decode_pcm_known_values() -> None:
    data = _s16le([0, 16384, -16384, 32767, -32768])
    result = capture._decode_pcm(data)
    expected = np.array(
        [0.0, 16384 / 32768.0, -16384 / 32768.0, 32767 / 32768.0, -1.0],
        dtype=np.float32,
    )
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_decode_pcm_drops_trailing_odd_byte() -> None:
    data = _s16le([100]) + b"\x01"
    result = capture._decode_pcm(data)
    assert result.shape == (1,)


def test_decode_pcm_empty() -> None:
    result = capture._decode_pcm(b"")
    assert result.shape == (0,)
    assert result.dtype == np.float32


# -- RecordingSession: capture + level callback --------------------------


def test_capture_converts_pcm_and_reports_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = [1500] * 100  # constant amplitude -> RMS 1500 -> level 0.5
    chunk = _s16le(values)
    levels: list[float] = []
    session, _ = _make_session(monkeypatch, [chunk], on_level=levels.append)

    _wait_until(lambda: not session.is_recording)
    pcm = session.stop()

    assert levels == [pytest.approx(0.5, abs=1e-3)]
    assert pcm.dtype == np.float32
    expected = np.full(100, 1500 / 32768.0, dtype=np.float32)
    np.testing.assert_allclose(pcm, expected, rtol=1e-6)


def test_level_is_clamped_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _s16le([32767] * 50)  # RMS far above RMS_SCALE
    levels: list[float] = []
    session, _ = _make_session(monkeypatch, [chunk], on_level=levels.append)

    _wait_until(lambda: not session.is_recording)
    session.stop()

    assert levels == [1.0]


def test_on_level_exception_does_not_crash_reader_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_on_level(level: float) -> None:
        raise RuntimeError("boom")

    chunk = _s16le([100] * 50)
    session, _ = _make_session(monkeypatch, [chunk], on_level=bad_on_level)

    _wait_until(lambda: not session.is_recording)
    pcm = session.stop()

    assert pcm.shape[0] == 50


# -- Max duration cutoff --------------------------------------------------


def test_max_duration_cutoff_stops_recording_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _s16le([100] * (capture.CHUNK_BYTES // 2))  # exactly one chunk
    chunks = [chunk] * 20  # far more than 1 second worth
    calls: list[int] = []
    session, fake_process = _make_session(
        monkeypatch,
        chunks,
        max_seconds=1,
        on_max_duration=lambda: calls.append(1),
    )

    _wait_until(lambda: not session.is_recording)

    assert calls == [1]
    assert fake_process.terminated is True
    pcm = session.stop()
    assert pcm.shape[0] == capture.SAMPLE_RATE


def test_on_max_duration_exception_does_not_crash_reader_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_on_max_duration() -> None:
        raise RuntimeError("boom")

    chunk = _s16le([100] * (capture.CHUNK_BYTES // 2))
    chunks = [chunk] * 20
    session, fake_process = _make_session(
        monkeypatch, chunks, max_seconds=1, on_max_duration=bad_on_max_duration
    )

    _wait_until(lambda: not session.is_recording)
    pcm = session.stop()

    assert fake_process.terminated is True
    assert pcm.shape[0] == capture.SAMPLE_RATE


# -- stop() idempotency and duration ---------------------------------------


def test_stop_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _s16le([10, 20, 30])
    session, fake_process = _make_session(monkeypatch, [chunk])

    _wait_until(lambda: not session.is_recording)
    first = session.stop()
    second = session.stop()

    assert second is first
    assert fake_process.terminate_calls == 1


def test_duration_seconds_reflects_captured_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _s16le([0] * capture.SAMPLE_RATE)  # exactly 1 second of samples
    session, _ = _make_session(monkeypatch, [chunk])

    _wait_until(lambda: not session.is_recording)
    assert session.duration_seconds == pytest.approx(1.0)
    session.stop()


# -- start() error handling -------------------------------------------------


def test_start_raises_capture_error_when_ffmpeg_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _popen(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("no such file: ffmpeg")

    monkeypatch.setattr(capture.subprocess, "Popen", _popen)
    session = RecordingSession()

    with pytest.raises(CaptureError):
        session.start()


def test_start_twice_raises_capture_error(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _ = _make_session(monkeypatch, [b""])

    with pytest.raises(CaptureError):
        session.start()


# -- list_audio_devices ------------------------------------------------------


def test_list_audio_devices_parses_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=list(args), returncode=1, stdout="", stderr=SAMPLE_DEVICE_STDERR
        )

    monkeypatch.setattr(capture.subprocess, "run", fake_run)

    assert capture.list_audio_devices() == [
        AudioDevice(0, "MacBook Pro Microphone"),
        AudioDevice(1, "External Headset"),
    ]


def test_list_audio_devices_fallback_on_missing_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("no such file: ffmpeg")

    monkeypatch.setattr(capture.subprocess, "run", fake_run)

    assert capture.list_audio_devices() == [AudioDevice(0, "Default")]


def test_list_audio_devices_fallback_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5)

    monkeypatch.setattr(capture.subprocess, "run", fake_run)

    assert capture.list_audio_devices() == [AudioDevice(0, "Default")]


def test_list_audio_devices_fallback_when_no_audio_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=list(args), returncode=1, stdout="", stderr="nothing useful here\n"
        )

    monkeypatch.setattr(capture.subprocess, "run", fake_run)

    assert capture.list_audio_devices() == [AudioDevice(0, "Default")]
