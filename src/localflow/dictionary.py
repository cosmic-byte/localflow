"""User-maintained glossary for ASR biasing and transcript text replacement."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from localflow import config

log = logging.getLogger(__name__)


@dataclass
class Dictionary:
    """A user's custom vocabulary and text replacement rules.

    Attributes:
        words: Vocabulary words used to bias transcription toward the
            speaker's own jargon and proper nouns (e.g. product names).
        replacements: Case-insensitive, whole-word replacements applied to
            transcripts after cleanup, mapping a misheard form to its
            intended correction.
    """

    words: list[str] = field(default_factory=list)
    replacements: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls) -> Dictionary:
        """Load the dictionary from config.DICTIONARY_PATH.

        Returns:
            The persisted dictionary, or an empty Dictionary when the file is
            missing, unreadable, or holds unexpected content.
        """
        try:
            raw = json.loads(config.DICTIONARY_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        words = raw.get("words", [])
        replacements = raw.get("replacements", {})
        if not isinstance(words, list) or not isinstance(replacements, dict):
            return cls()
        return cls(words=list(words), replacements=dict(replacements))

    def save(self) -> None:
        """Persist the dictionary, creating the destination directory if needed."""
        try:
            config.DICTIONARY_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {"words": self.words, "replacements": self.replacements}
            config.DICTIONARY_PATH.write_text(json.dumps(payload, indent=2))
        except OSError:
            log.exception("Failed to save dictionary")

    def initial_prompt(self) -> str | None:
        """Build a Whisper biasing prompt from the glossary words.

        Returns:
            A prompt like "Glossary: Kubernetes, Baseten." or None when there
            are no words to bias toward.
        """
        if not self.words:
            return None
        return f"Glossary: {', '.join(self.words)}."

    def apply_replacements(self, text: str) -> str:
        """Apply case-insensitive whole-word replacements to text.

        Args:
            text: The text to transform.

        Returns:
            The text with each configured key replaced by its value.
            Matching is case-insensitive and restricted to whole words, so a
            key like "cat" does not match inside "catalog".
        """
        for key, value in self.replacements.items():
            pattern = re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE)
            text = pattern.sub(value, text)
        return text
