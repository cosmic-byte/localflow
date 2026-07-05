"""Tests for the pure text-processing step of the application controller.

These tests exercise ``process_transcript`` with mock cleaner and dictionary
objects that satisfy the module contracts, so no network, audio, GUI, or model
weights are involved. The real ``should_clean`` gate and ``CleanupRequest`` from
the cleanup module are used, which lets the tests assert exactly what the cleaner
receives.
"""

from __future__ import annotations

from localflow import app
from localflow.config import AppConfig


class FakeCleaner:
    """Records the requests it receives and returns a canned result."""

    def __init__(self, result: str) -> None:
        self.result = result
        self.requests: list[object] = []

    def clean(self, request: object) -> str:
        self.requests.append(request)
        return self.result


class FakeDictionary:
    """Minimal dictionary satisfying the words + apply_replacements contract."""

    def __init__(
        self,
        words: tuple[str, ...] = (),
        replacements: dict[str, str] | None = None,
    ) -> None:
        self.words = list(words)
        self.__replacements = replacements or {}
        self.apply_calls: list[str] = []

    def apply_replacements(self, text: str) -> str:
        self.apply_calls.append(text)
        for key, value in self.__replacements.items():
            text = text.replace(key, value)
        return text


def test_cleanup_enabled_runs_cleaner_then_replacements() -> None:
    config = AppConfig(cleanup_enabled=True, cleanup_min_chars=10)
    cleaner = FakeCleaner("cleaned KUBE text goes here")
    dictionary = FakeDictionary(
        words=("Kubernetes",), replacements={"KUBE": "Kubernetes"}
    )
    text = "this is a fairly long raw transcript that should be cleaned"

    result = app.process_transcript(text, config, dictionary, cleaner, "standard")

    assert len(cleaner.requests) == 1
    assert dictionary.apply_calls == ["cleaned KUBE text goes here"]
    assert result == "cleaned Kubernetes text goes here"


def test_cleanup_disabled_skips_cleaner_but_applies_replacements() -> None:
    config = AppConfig(cleanup_enabled=False, cleanup_min_chars=10)
    cleaner = FakeCleaner("SHOULD NOT BE USED")
    dictionary = FakeDictionary(replacements={"raw": "cooked"})
    text = "this raw transcript stays uncleaned but is still replaced"

    result = app.process_transcript(text, config, dictionary, cleaner, "standard")

    assert cleaner.requests == []
    assert result == "this cooked transcript stays uncleaned but is still replaced"


def test_short_text_skips_cleaner_even_when_enabled() -> None:
    config = AppConfig(cleanup_enabled=True, cleanup_min_chars=50)
    cleaner = FakeCleaner("SHOULD NOT BE USED")
    dictionary = FakeDictionary()
    text = "hi there"

    result = app.process_transcript(text, config, dictionary, cleaner, "standard")

    assert cleaner.requests == []
    assert result == "hi there"


def test_dictionary_words_and_tone_forwarded_to_request() -> None:
    config = AppConfig(cleanup_enabled=True, cleanup_min_chars=5)
    cleaner = FakeCleaner("ok")
    dictionary = FakeDictionary(words=("Kubernetes", "Baseten"))
    text = "please clean up this reasonably long transcript now"

    app.process_transcript(text, config, dictionary, cleaner, "casual")

    request = cleaner.requests[0]
    assert request.text == text
    assert request.tone == "casual"
    assert request.dictionary_words == ("Kubernetes", "Baseten")


def test_replacements_applied_after_cleanup() -> None:
    config = AppConfig(cleanup_enabled=True, cleanup_min_chars=5)
    cleaner = FakeCleaner("we use kubernetes every single day")
    dictionary = FakeDictionary(replacements={"kubernetes": "Kubernetes"})
    text = "we use cooper netties every single day at work"

    result = app.process_transcript(text, config, dictionary, cleaner, "standard")

    assert result == "we use Kubernetes every single day"
