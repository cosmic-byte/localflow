"""Speech-to-text backends for localflow.

Provides a common Transcriber interface over two local Whisper backends:
MLX (mlx-whisper, Apple Silicon GPU) and whisper.cpp (pywhispercpp). Both
optional backend packages are imported defensively so the module stays
importable when neither is installed; create_transcriber raises AsrError with
an actionable install hint when the requested backend is missing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None

try:
    from pywhispercpp.model import Model as WhisperCppModel
except ImportError:
    WhisperCppModel = None

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_WARMUP_SECONDS = 0.5

_MLX_REPOS: dict[str, str] = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
}

_WHISPERCPP_MODELS: dict[str, str] = {
    "large-v3-turbo": "large-v3-turbo",
    "turbo": "large-v3-turbo",
    "small": "small",
    "base": "base",
}


class AsrError(Exception):
    """Raised when a backend is unavailable or transcription fails."""


def _mlx_repo_for_model(model_name: str) -> str:
    """Map a Whisper model name to its mlx-community Hugging Face repo.

    Args:
        model_name: Short model name such as "large-v3-turbo", "small", or
            "base".

    Returns:
        The Hugging Face repo id understood by mlx_whisper.

    Raises:
        AsrError: If the model name has no known MLX repo mapping.
    """
    try:
        return _MLX_REPOS[model_name]
    except KeyError:
        supported = ", ".join(sorted(_MLX_REPOS))
        raise AsrError(
            f"Unknown MLX model {model_name!r}. Supported models: {supported}."
        ) from None


def _whispercpp_model_for_name(model_name: str) -> str:
    """Map a Whisper model name to the identifier pywhispercpp downloads.

    Args:
        model_name: Short model name such as "large-v3-turbo", "small", or
            "base".

    Returns:
        The model identifier understood by pywhispercpp.

    Raises:
        AsrError: If the model name has no known whisper.cpp mapping.
    """
    try:
        return _WHISPERCPP_MODELS[model_name]
    except KeyError:
        supported = ", ".join(sorted(_WHISPERCPP_MODELS))
        raise AsrError(
            f"Unknown whispercpp model {model_name!r}. Supported models: {supported}."
        ) from None


def _warmup_audio() -> np.ndarray:
    """Return a short buffer of silence used to warm up a backend."""
    return np.zeros(int(SAMPLE_RATE * _WARMUP_SECONDS), dtype=np.float32)


def _validate_audio(audio: np.ndarray) -> None:
    """Validate that audio is float32 mono PCM.

    Args:
        audio: Candidate audio buffer.

    Raises:
        AsrError: If audio is not a 1-D float32 numpy array.
    """
    if not isinstance(audio, np.ndarray):
        raise AsrError(f"audio must be a numpy.ndarray, got {type(audio).__name__}.")
    if audio.dtype != np.float32:
        raise AsrError(f"audio must be float32, got {audio.dtype}.")
    if audio.ndim != 1:
        raise AsrError(f"audio must be mono (1-D), got a {audio.ndim}-D array.")


class Transcriber(ABC):
    """Abstract speech-to-text backend."""

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
    """Transcriber backed by mlx-whisper (Apple Silicon GPU)."""

    def __init__(self, model_name: str = "large-v3-turbo") -> None:
        """Initialize the transcriber and resolve the backing HF repo.

        Args:
            model_name: Short Whisper model name to load.

        Raises:
            AsrError: If the model name has no known MLX repo mapping.
        """
        self.__model_name = model_name
        self.__repo = _mlx_repo_for_model(model_name)

    @property
    def model_name(self) -> str:
        """Short Whisper model name this transcriber loads."""
        return self.__model_name

    def load(self) -> None:
        """Warm the model cache with a short buffer of silence.

        mlx_whisper caches weights per repo internally, so there is no separate
        model handle to keep; the warmup call triggers the (possibly cached)
        download and compilation so the first real transcription is fast.

        Raises:
            AsrError: If the mlx-whisper package is not installed.
        """
        if mlx_whisper is None:
            raise AsrError(_missing_package_message("mlx"))
        self.transcribe(_warmup_audio())

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en",
        initial_prompt: str | None = None,
    ) -> str:
        """Transcribe audio with mlx-whisper and return stripped text.

        Args:
            audio: Float32 mono 16 kHz PCM in [-1, 1].
            language: ISO language code passed to the decoder.
            initial_prompt: Optional biasing prompt (e.g. a glossary).

        Returns:
            The stripped transcript text.

        Raises:
            AsrError: If the package is missing, the audio is invalid, or
                transcription fails.
        """
        if mlx_whisper is None:
            raise AsrError(_missing_package_message("mlx"))
        _validate_audio(audio)
        try:
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.__repo,
                language=language,
                initial_prompt=initial_prompt,
            )
        except Exception as exc:
            log.exception("mlx-whisper transcription failed")
            raise AsrError(f"mlx-whisper transcription failed: {exc}") from exc
        return str(result["text"]).strip()


class WhisperCppTranscriber(Transcriber):
    """Transcriber backed by whisper.cpp via pywhispercpp."""

    def __init__(self, model_name: str = "large-v3-turbo") -> None:
        """Initialize the transcriber and resolve the whisper.cpp model id.

        Args:
            model_name: Short Whisper model name to load.

        Raises:
            AsrError: If the model name has no known whisper.cpp mapping.
        """
        self.__model_name = model_name
        self.__model_id = _whispercpp_model_for_name(model_name)
        self.__model: object | None = None

    @property
    def model_name(self) -> str:
        """Short Whisper model name this transcriber loads."""
        return self.__model_name

    def load(self) -> None:
        """Construct the whisper.cpp model and warm it with silence.

        Raises:
            AsrError: If the pywhispercpp package is not installed.
        """
        if WhisperCppModel is None:
            raise AsrError(_missing_package_message("whispercpp"))
        self.__model = WhisperCppModel(self.__model_id)
        self.transcribe(_warmup_audio())

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en",
        initial_prompt: str | None = None,
    ) -> str:
        """Transcribe audio with whisper.cpp and return stripped text.

        pywhispercpp's Model.transcribe accepts a numpy float32 array and
        returns a list of segments; their texts are concatenated. The language
        and initial_prompt are forwarded as whisper.cpp full parameters.

        Args:
            audio: Float32 mono 16 kHz PCM in [-1, 1].
            language: ISO language code passed to whisper.cpp.
            initial_prompt: Optional biasing prompt (e.g. a glossary).

        Returns:
            The stripped transcript text.

        Raises:
            AsrError: If the model is not loaded, the audio is invalid, or
                transcription fails.
        """
        if WhisperCppModel is None:
            raise AsrError(_missing_package_message("whispercpp"))
        if self.__model is None:
            raise AsrError("whispercpp model is not loaded; call load() first.")
        _validate_audio(audio)
        params: dict[str, object] = {"language": language}
        if initial_prompt:
            params["initial_prompt"] = initial_prompt
        try:
            segments = self.__model.transcribe(audio, **params)
        except Exception as exc:
            log.exception("whisper.cpp transcription failed")
            raise AsrError(f"whisper.cpp transcription failed: {exc}") from exc
        return "".join(segment.text for segment in segments).strip()


def _missing_package_message(backend: str) -> str:
    """Build an actionable error message for a missing backend package.

    Args:
        backend: Backend key, "mlx" or "whispercpp".

    Returns:
        A human-readable message naming the missing package and how to install
        it.
    """
    if backend == "mlx":
        return (
            "ASR backend 'mlx' is unavailable: the 'mlx-whisper' package is not "
            "installed. Install it with: pip install mlx-whisper"
        )
    return (
        "ASR backend 'whispercpp' is unavailable: the 'pywhispercpp' package is "
        "not installed. Install it with: pip install 'localflow[whispercpp]' "
        "(or pip install pywhispercpp)"
    )


def create_transcriber(backend: str, model_name: str) -> Transcriber:
    """Create a transcriber for the requested backend.

    Args:
        backend: Backend key, "mlx" or "whispercpp".
        model_name: Short Whisper model name to load.

    Returns:
        A Transcriber for the backend.

    Raises:
        AsrError: If the backend is unknown or its package is not installed.
    """
    if backend == "mlx":
        if mlx_whisper is None:
            raise AsrError(_missing_package_message("mlx"))
        return MlxWhisperTranscriber(model_name)
    if backend == "whispercpp":
        if WhisperCppModel is None:
            raise AsrError(_missing_package_message("whispercpp"))
        return WhisperCppTranscriber(model_name)
    raise AsrError(f"Unknown ASR backend {backend!r}. Expected 'mlx' or 'whispercpp'.")
