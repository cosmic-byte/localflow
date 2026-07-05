"""Tests for localflow.dictionary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from localflow import config
from localflow.dictionary import Dictionary


@pytest.fixture
def dictionary_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point config.DICTIONARY_PATH at a throwaway file for the duration of a test."""
    path = tmp_path / "dictionary.json"
    monkeypatch.setattr(config, "DICTIONARY_PATH", path)
    return path


def test_load_missing_file_returns_empty(dictionary_path: Path) -> None:
    assert Dictionary.load() == Dictionary()


def test_round_trip_load_save(dictionary_path: Path) -> None:
    original = Dictionary(
        words=["Kubernetes", "Baseten"],
        replacements={"cooper netties": "Kubernetes"},
    )

    original.save()
    loaded = Dictionary.load()

    assert loaded == original


def test_save_creates_missing_parent_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "nested" / "dir" / "dictionary.json"
    monkeypatch.setattr(config, "DICTIONARY_PATH", path)

    Dictionary(words=["foo"]).save()

    assert path.exists()
    assert json.loads(path.read_text())["words"] == ["foo"]


def test_load_corrupt_json_returns_empty(dictionary_path: Path) -> None:
    dictionary_path.write_text("not valid json{")

    assert Dictionary.load() == Dictionary()


def test_load_non_dict_json_returns_empty(dictionary_path: Path) -> None:
    dictionary_path.write_text(json.dumps(["a", "list", "not", "a", "dict"]))

    assert Dictionary.load() == Dictionary()


def test_load_wrong_field_types_returns_empty(dictionary_path: Path) -> None:
    dictionary_path.write_text(json.dumps({"words": "not-a-list", "replacements": {}}))

    assert Dictionary.load() == Dictionary()


def test_initial_prompt_with_words() -> None:
    dictionary = Dictionary(words=["Kubernetes", "Baseten"])

    assert dictionary.initial_prompt() == "Glossary: Kubernetes, Baseten."


def test_initial_prompt_empty_is_none() -> None:
    assert Dictionary().initial_prompt() is None


def test_apply_replacements_case_insensitive() -> None:
    dictionary = Dictionary(replacements={"kubernetes": "Kubernetes"})

    result = dictionary.apply_replacements("we use kubernetes every day")

    assert result == "we use Kubernetes every day"


def test_apply_replacements_whole_word_only() -> None:
    dictionary = Dictionary(replacements={"cat": "CAT"})

    result = dictionary.apply_replacements("the cat sat near the catalog")

    assert result == "the CAT sat near the catalog"


def test_apply_replacements_no_match_leaves_text_unchanged() -> None:
    dictionary = Dictionary(replacements={"foo": "bar"})
    text = "nothing to replace here"

    assert dictionary.apply_replacements(text) == text


def test_apply_replacements_multiple_keys() -> None:
    dictionary = Dictionary(
        replacements={"cooper netties": "Kubernetes", "base ten": "Baseten"}
    )

    result = dictionary.apply_replacements(
        "restart the cooper netties cluster and check base ten"
    )

    assert result == "restart the Kubernetes cluster and check Baseten"


def test_apply_replacements_mixed_case_input() -> None:
    dictionary = Dictionary(replacements={"kubernetes": "Kubernetes"})

    result = dictionary.apply_replacements("KUBERNETES and Kubernetes and kubernetes")

    assert result == "Kubernetes and Kubernetes and Kubernetes"
