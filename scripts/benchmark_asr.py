#!/usr/bin/env python3
"""Benchmark available localflow ASR backends over WAV files or a recording.

Usage:
    benchmark_asr.py sample1.wav sample2.wav
    benchmark_asr.py --record 5
    benchmark_asr.py sample.wav --runs 5 --model small

For each installed backend (mlx, whispercpp) the script measures model load
time and per-file wall-clock transcription time, then prints a table with the
realtime factor and each transcript. Backends whose package is not installed
are skipped. The script exits nonzero if no backend is available.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from localflow import asr  # noqa: E402

SAMPLE_RATE = 16000
_BACKENDS = ("mlx", "whispercpp")


@dataclass
class AudioInput:
    """A named audio buffer to transcribe.

    Attributes:
        label: Display name (file name or "recording").
        pcm: Float32 mono 16 kHz PCM in [-1, 1].
    """

    label: str
    pcm: np.ndarray

    @property
    def duration_seconds(self) -> float:
        """Length of the buffer in seconds."""
        return len(self.pcm) / SAMPLE_RATE


def read_wav(path: Path) -> AudioInput:
    """Read a 16 kHz mono s16le WAV file into a float32 buffer.

    Resampling is out of scope; any other format is rejected with a clear
    error.

    Args:
        path: Path to the WAV file.

    Returns:
        The decoded audio input.

    Raises:
        ValueError: If the file is not 16 kHz mono 16-bit PCM.
    """
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if channels != 1:
        raise ValueError(f"{path}: expected mono, got {channels} channels.")
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {rate} Hz.")
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {width * 8}-bit.")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return AudioInput(label=path.name, pcm=pcm)


def record(seconds: int) -> AudioInput:
    """Capture microphone audio for a fixed duration via localflow.capture.

    Args:
        seconds: Number of seconds to record.

    Returns:
        The recorded audio input.
    """
    from localflow import capture

    session = capture.RecordingSession(max_seconds=max(seconds + 1, 2))
    session.start()
    time.sleep(seconds)
    pcm = session.stop()
    return AudioInput(label=f"recording[{seconds}s]", pcm=pcm)


def _time_transcribe(
    transcriber: asr.Transcriber,
    audio: AudioInput,
    language: str,
    runs: int,
) -> tuple[float, float, str]:
    """Transcribe an input several times and measure wall-clock time.

    Args:
        transcriber: A loaded transcriber.
        audio: The audio input to transcribe.
        language: ISO language code.
        runs: Number of repetitions.

    Returns:
        A tuple of (mean_seconds, min_seconds, transcript).
    """
    durations: list[float] = []
    transcript = ""
    for _ in range(runs):
        start = time.perf_counter()
        transcript = transcriber.transcribe(audio.pcm, language=language)
        durations.append(time.perf_counter() - start)
    return sum(durations) / len(durations), min(durations), transcript


@dataclass
class Row:
    """One measured (backend, file) result row."""

    backend: str
    model: str
    label: str
    audio_seconds: float
    mean_seconds: float
    min_seconds: float
    transcript: str


def _print_table(rows: list[Row], load_times: dict[str, float]) -> None:
    """Print the results table and each transcript.

    Args:
        rows: Measured rows.
        load_times: Backend load time in seconds keyed by backend name.
    """
    header = (
        f"{'backend':<11}{'model':<16}{'file':<22}"
        f"{'audio(s)':>9}{'mean(s)':>9}{'min(s)':>9}{'xRT':>8}"
    )
    print(
        "\nLoad times: "
        + ", ".join(f"{name}={secs:.2f}s" for name, secs in load_times.items())
    )
    print("(xRT = audio seconds / mean transcribe seconds; higher is faster)\n")
    print(header)
    print("-" * len(header))
    for row in rows:
        xrt = row.audio_seconds / row.mean_seconds if row.mean_seconds else 0.0
        print(
            f"{row.backend:<11}{row.model:<16}{row.label:<22}"
            f"{row.audio_seconds:>9.2f}{row.mean_seconds:>9.2f}"
            f"{row.min_seconds:>9.2f}{xrt:>8.1f}"
        )
    print("\nTranscripts:")
    for row in rows:
        print(f"  [{row.backend} · {row.label}] {row.transcript!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, defaulting to sys.argv.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "wav", nargs="*", type=Path, help="16 kHz mono s16le WAV file(s)."
    )
    parser.add_argument(
        "--record",
        type=int,
        metavar="N",
        help="Record N seconds from the microphone instead of reading files.",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Transcriptions per file (default 3)."
    )
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model name.")
    parser.add_argument(
        "--language", default="en", help="ISO language code (default en)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark.

    Args:
        argv: Argument list, defaulting to sys.argv.

    Returns:
        Process exit code (0 on success, nonzero on error).
    """
    args = parse_args(argv)
    if args.runs < 1:
        print("--runs must be at least 1.", file=sys.stderr)
        return 2

    inputs: list[AudioInput] = []
    try:
        if args.record:
            inputs.append(record(args.record))
        for path in args.wav:
            inputs.append(read_wav(path))
    except (ValueError, OSError, wave.Error) as exc:
        print(f"Failed to read audio: {exc}", file=sys.stderr)
        return 2

    if not inputs:
        print(
            "No audio to benchmark. Provide WAV path(s) or --record N.",
            file=sys.stderr,
        )
        return 2

    rows: list[Row] = []
    load_times: dict[str, float] = {}
    available = 0
    for backend in _BACKENDS:
        try:
            transcriber = asr.create_transcriber(backend, args.model)
        except asr.AsrError as exc:
            print(f"Skipping {backend}: {exc}", file=sys.stderr)
            continue
        available += 1
        start = time.perf_counter()
        transcriber.load()
        load_times[backend] = time.perf_counter() - start
        for audio in inputs:
            mean_s, min_s, text = _time_transcribe(
                transcriber, audio, args.language, args.runs
            )
            rows.append(
                Row(
                    backend=backend,
                    model=args.model,
                    label=audio.label,
                    audio_seconds=audio.duration_seconds,
                    mean_seconds=mean_s,
                    min_seconds=min_s,
                    transcript=text,
                )
            )

    if available == 0:
        print(
            "No ASR backend is available. Install one, e.g. pip install mlx-whisper",
            file=sys.stderr,
        )
        return 1

    _print_table(rows, load_times)
    return 0


if __name__ == "__main__":
    sys.exit(main())
