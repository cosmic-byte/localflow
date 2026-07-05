"""Tests for the ASR backends.

The underlying backend libraries (mlx_whisper, pywhispercpp) are never invoked
for real; they are replaced on the asr module via monkeypatch so no model is
ever downloaded and no inference runs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from localflow import asr
from localflow.asr import (
    AsrError,
    MlxWhisperTranscriber,
    WhisperCppTranscriber,
    create_transcriber,
)


@pytest.mark.parametrize(
    "name,repo",
    [
        ("large-v3-turbo", "mlx-community/whisper-large-v3-turbo"),
        ("turbo", "mlx-community/whisper-large-v3-turbo"),
        ("small", "mlx-community/whisper-small-mlx"),
        ("base", "mlx-community/whisper-base-mlx"),
    ],
)
def test_mlx_repo_mapping(name: str, repo: str) -> None:
    assert asr._mlx_repo_for_model(name) == repo


def test_mlx_repo_mapping_unknown() -> None:
    with pytest.raises(AsrError, match="Unknown MLX model"):
        asr._mlx_repo_for_model("medium")


@pytest.mark.parametrize(
    "name,model_id",
    [
        ("large-v3-turbo", "large-v3-turbo"),
        ("turbo", "large-v3-turbo"),
        ("small", "small"),
        ("base", "base"),
    ],
)
def test_whispercpp_model_mapping(name: str, model_id: str) -> None:
    assert asr._whispercpp_model_for_name(name) == model_id


def test_whispercpp_model_mapping_unknown() -> None:
    with pytest.raises(AsrError, match="Unknown whispercpp model"):
        asr._whispercpp_model_for_name("medium")


def test_create_transcriber_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asr, "mlx_whisper", MagicMock())
    transcriber = create_transcriber("mlx", "small")
    assert isinstance(transcriber, MlxWhisperTranscriber)
    assert transcriber.model_name == "small"


def test_create_transcriber_whispercpp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asr, "WhisperCppModel", MagicMock())
    transcriber = create_transcriber("whispercpp", "base")
    assert isinstance(transcriber, WhisperCppTranscriber)
    assert transcriber.model_name == "base"


def test_create_transcriber_mlx_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asr, "mlx_whisper", None)
    with pytest.raises(AsrError, match="mlx-whisper") as exc:
        create_transcriber("mlx", "small")
    assert "pip install" in str(exc.value)


def test_create_transcriber_whispercpp_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asr, "WhisperCppModel", None)
    with pytest.raises(AsrError, match="pywhispercpp") as exc:
        create_transcriber("whispercpp", "base")
    assert "pip install" in str(exc.value)


def test_create_transcriber_unknown_backend() -> None:
    with pytest.raises(AsrError, match="Unknown ASR backend"):
        create_transcriber("bogus", "small")


def test_validate_audio_rejects_non_ndarray() -> None:
    with pytest.raises(AsrError, match="numpy.ndarray"):
        asr._validate_audio([0.0, 1.0])


def test_validate_audio_rejects_wrong_dtype() -> None:
    with pytest.raises(AsrError, match="float32"):
        asr._validate_audio(np.zeros(10, dtype=np.float64))


def test_validate_audio_rejects_multichannel() -> None:
    with pytest.raises(AsrError, match="mono"):
        asr._validate_audio(np.zeros((10, 2), dtype=np.float32))


def test_validate_audio_accepts_valid() -> None:
    asr._validate_audio(np.zeros(10, dtype=np.float32))


def test_mlx_load_warms_up(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.transcribe.return_value = {"text": " warm "}
    monkeypatch.setattr(asr, "mlx_whisper", fake)

    transcriber = MlxWhisperTranscriber("small")
    transcriber.load()

    fake.transcribe.assert_called_once()
    args, kwargs = fake.transcribe.call_args
    audio = args[0]
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) == int(asr.SAMPLE_RATE * asr._WARMUP_SECONDS)
    assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-small-mlx"


def test_mlx_transcribe_returns_stripped_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    fake.transcribe.return_value = {"text": "  hello world  "}
    monkeypatch.setattr(asr, "mlx_whisper", fake)

    transcriber = MlxWhisperTranscriber("base")
    out = transcriber.transcribe(
        np.zeros(16000, dtype=np.float32),
        language="en",
        initial_prompt="Glossary: Kubernetes",
    )

    assert out == "hello world"
    _, kwargs = fake.transcribe.call_args
    assert kwargs["language"] == "en"
    assert kwargs["initial_prompt"] == "Glossary: Kubernetes"
    assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-base-mlx"


def test_mlx_transcribe_rejects_bad_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    monkeypatch.setattr(asr, "mlx_whisper", fake)

    transcriber = MlxWhisperTranscriber("small")
    with pytest.raises(AsrError, match="float32"):
        transcriber.transcribe(np.zeros(10, dtype=np.int16))
    fake.transcribe.assert_not_called()


def test_mlx_transcribe_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asr, "mlx_whisper", None)
    transcriber = MlxWhisperTranscriber("small")
    with pytest.raises(AsrError, match="mlx-whisper"):
        transcriber.transcribe(np.zeros(10, dtype=np.float32))


def test_whispercpp_load_and_transcribe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = MagicMock()
    first.text = " hello"
    second = MagicMock()
    second.text = " world"
    model_instance = MagicMock()
    model_instance.transcribe.return_value = [first, second]
    model_cls = MagicMock(return_value=model_instance)
    monkeypatch.setattr(asr, "WhisperCppModel", model_cls)

    transcriber = WhisperCppTranscriber("large-v3-turbo")
    transcriber.load()

    model_cls.assert_called_once_with("large-v3-turbo")
    assert model_instance.transcribe.call_count == 1  # warmup
    warmup_audio = model_instance.transcribe.call_args.args[0]
    assert warmup_audio.dtype == np.float32
    assert len(warmup_audio) == int(asr.SAMPLE_RATE * asr._WARMUP_SECONDS)

    out = transcriber.transcribe(np.zeros(16000, dtype=np.float32), language="en")
    assert out == "hello world"


def test_whispercpp_transcribe_before_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asr, "WhisperCppModel", MagicMock())
    transcriber = WhisperCppTranscriber("small")
    with pytest.raises(AsrError, match="not loaded"):
        transcriber.transcribe(np.zeros(10, dtype=np.float32))


def test_whispercpp_load_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asr, "WhisperCppModel", None)
    transcriber = WhisperCppTranscriber("small")
    with pytest.raises(AsrError, match="pywhispercpp"):
        transcriber.load()


def test_whispercpp_forwards_language_and_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = MagicMock()
    segment.text = "hi"
    model_instance = MagicMock()
    model_instance.transcribe.return_value = [segment]
    monkeypatch.setattr(asr, "WhisperCppModel", MagicMock(return_value=model_instance))

    transcriber = WhisperCppTranscriber("base")
    transcriber.load()
    model_instance.transcribe.reset_mock()

    transcriber.transcribe(
        np.zeros(10, dtype=np.float32),
        language="fr",
        initial_prompt="Glossary: Baseten",
    )

    _, kwargs = model_instance.transcribe.call_args
    assert kwargs["language"] == "fr"
    assert kwargs["initial_prompt"] == "Glossary: Baseten"


def test_whispercpp_omits_prompt_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = MagicMock()
    segment.text = "hi"
    model_instance = MagicMock()
    model_instance.transcribe.return_value = [segment]
    monkeypatch.setattr(asr, "WhisperCppModel", MagicMock(return_value=model_instance))

    transcriber = WhisperCppTranscriber("base")
    transcriber.load()
    model_instance.transcribe.reset_mock()

    transcriber.transcribe(np.zeros(10, dtype=np.float32))

    _, kwargs = model_instance.transcribe.call_args
    assert "initial_prompt" not in kwargs
