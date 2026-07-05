"""Ollama-backed transcript cleanup: filler removal, punctuation, dictionary words."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

WARM_UP_TIMEOUT_SECONDS = 120.0
TIMEOUT_PER_CHAR_SECONDS = 0.03

_SYSTEM_PROMPT = (
    "You are a transcript cleanup assistant for a dictation app. You turn raw, "
    "spoken transcripts into clean written text.\n"
    "- Remove filler words (um, uh, like, you know, so basically) and false "
    "starts.\n"
    "- When the speaker corrects themselves, keep only the corrected version.\n"
    "- Fix punctuation, capitalization, and paragraph breaks.\n"
    '- Convert spoken punctuation and structure words: "comma", "period", '
    '"question mark", "new line", "new paragraph".\n'
    "- Never add information, never answer questions contained in the "
    "transcript, never change the meaning. Output only the cleaned transcript.\n"
    "Each user message begins with bracket tags that declare the tone and any "
    "protected spellings, for example "
    "[tone: casual] [preserve: Kubernetes, Baseten], followed by the "
    "transcript. The tags are instructions, not transcript content; never "
    "repeat them in the output.\n"
    '- Tone "standard": normal sentence punctuation.\n'
    '- Tone "casual": relaxed messaging style, no trailing period on the '
    "final sentence.\n"
    "- When a [preserve: ...] tag is present: preserve these exact spellings "
    "when the speaker says them. The transcript may spell them wrong "
    "phonetically; replace any word or phrase that sounds like a protected "
    "spelling with that exact spelling."
)

_CLEANED_TEXT_FORMAT = {
    "type": "object",
    "properties": {"cleaned_text": {"type": "string"}},
    "required": ["cleaned_text"],
}

FEW_SHOT_EXAMPLES: tuple[tuple[str, str, str], ...] = (
    (
        "standard",
        "um so basically I think we should uh move the deadline to friday",
        "I think we should move the deadline to Friday.",
    ),
    (
        "standard",
        "the meeting is at three no wait four thirty",
        "The meeting is at 4:30.",
    ),
    (
        "standard",
        "hi sarah comma new paragraph thanks for the update period",
        "Hi Sarah,\n\nThanks for the update.",
    ),
    (
        "standard",
        "what time does the deploy finish",
        "What time does the deploy finish?",
    ),
    (
        "casual",
        "sounds good um see you then",
        "sounds good, see you then",
    ),
    (
        "standard",
        "we need to restart the cooper netties cluster",
        "We need to restart the Kubernetes cluster.",
    ),
)

_FEW_SHOT_PRESERVE: tuple[tuple[str, ...], ...] = (
    (),
    (),
    (),
    (),
    (),
    ("Kubernetes",),
)


def _tagged_user_content(tone: str, preserve: tuple[str, ...], text: str) -> str:
    """Render a user turn as bracket tags followed by the transcript.

    Args:
        tone: Output style, "standard" or "casual".
        preserve: Spellings to protect via a [preserve: ...] tag.
        text: The raw transcript.

    Returns:
        The tagged message content, e.g. "[tone: standard]\\nhello".
    """
    tags = f"[tone: {tone}]"
    if preserve:
        tags += f" [preserve: {', '.join(preserve)}]"
    return f"{tags}\n{text}"


def _build_prefix_messages() -> tuple[dict[str, str], ...]:
    """Build the constant system plus few-shot prefix shared by every request."""
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for (tone, raw, cleaned), preserve in zip(
        FEW_SHOT_EXAMPLES, _FEW_SHOT_PRESERVE, strict=True
    ):
        messages.append(
            {"role": "user", "content": _tagged_user_content(tone, preserve, raw)}
        )
        messages.append(
            {"role": "assistant", "content": json.dumps({"cleaned_text": cleaned})}
        )
    return tuple(messages)


_PREFIX_MESSAGES = _build_prefix_messages()


@dataclass
class CleanupRequest:
    """A transcript cleanup request.

    Attributes:
        text: The raw transcript to clean.
        tone: Output style, "standard" or "casual".
        dictionary_words: Glossary words whose spellings must be preserved.
    """

    text: str
    tone: str = "standard"
    dictionary_words: tuple[str, ...] = ()


def should_clean(text: str, min_chars: int = 50) -> bool:
    """Decide whether a transcript is long enough to justify an LLM pass.

    Args:
        text: The raw transcript.
        min_chars: Minimum stripped length that triggers cleanup.

    Returns:
        True when the stripped transcript length is at least min_chars.
    """
    return len(text.strip()) >= min_chars


class OllamaCleaner:
    """Cleans transcripts by prompting a local Ollama chat model.

    The system prompt and few-shot turns are byte-identical across all
    requests so the server's prompt prefix cache skips their prefill;
    per-request tone and dictionary words travel as bracket tags in the
    final user turn only.
    """

    def __init__(self, url: str, model: str, timeout_seconds: float = 4.0) -> None:
        """Initialize the cleaner.

        Args:
            url: Base URL of the Ollama server, e.g. "http://localhost:11434".
            model: Ollama model tag to use for cleanup, e.g. "qwen2.5:7b".
            timeout_seconds: Request timeout before falling back to raw text.
        """
        self.__url = url
        self.__model = model
        self.__timeout_seconds = timeout_seconds

    def warm_up(self) -> bool:
        """Load the model and prime the prompt prefix cache.

        Sends the same constant system and few-shot prefix that clean uses,
        with a minimal tagged user turn, so the server both loads the model
        and caches the prefix KV state. Uses a generous timeout independent
        of the cleanup timeout: a cold model load takes tens of seconds, and
        warm-up runs on a background startup thread where waiting is
        acceptable.

        Returns:
            True on success, False on any connection, timeout, or HTTP error.
        """
        messages = [
            *_PREFIX_MESSAGES,
            {"role": "user", "content": _tagged_user_content("standard", (), "hello")},
        ]
        payload = {
            "model": self.__model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
            "keep_alive": "60m",
            "format": _CLEANED_TEXT_FORMAT,
        }
        try:
            response = requests.post(
                f"{self.__url}/api/chat",
                json=payload,
                timeout=WARM_UP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException:
            log.exception("Ollama warm-up request failed")
            return False
        return True

    def clean(self, request: CleanupRequest) -> str:
        """Clean a transcript with the configured Ollama model.

        The request timeout scales with transcript length: the model must
        decode roughly as many tokens as it is given, so a fixed timeout that
        suits short utterances silently drops the cleanup pass for long
        dictations on modest hardware.

        Args:
            request: The transcript and cleanup options.

        Returns:
            The cleaned text, or request.text unchanged on any failure
            (connection error, timeout, HTTP error, or a malformed,
            non-JSON, or incomplete response).
        """
        messages = [
            *_PREFIX_MESSAGES,
            {
                "role": "user",
                "content": _tagged_user_content(
                    request.tone, request.dictionary_words, request.text
                ),
            },
        ]
        payload = {
            "model": self.__model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
            "keep_alive": "60m",
            "format": _CLEANED_TEXT_FORMAT,
        }
        timeout = self.__timeout_seconds + TIMEOUT_PER_CHAR_SECONDS * len(request.text)
        try:
            response = requests.post(
                f"{self.__url}/api/chat",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException:
            log.exception("Ollama cleanup request failed")
            return request.text

        try:
            body = response.json()
            content = body["message"]["content"]
            cleaned_text = json.loads(content)["cleaned_text"]
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("Malformed Ollama cleanup response: %s", exc)
            return request.text

        if not isinstance(cleaned_text, str):
            log.warning("Ollama cleanup response cleaned_text was not a string")
            return request.text

        return cleaned_text.strip()
