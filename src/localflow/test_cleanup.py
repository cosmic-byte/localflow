"""Tests for localflow.cleanup."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from localflow import cleanup
from localflow.cleanup import (
    FEW_SHOT_EXAMPLES,
    WARM_UP_TIMEOUT_SECONDS,
    CleanupRequest,
    OllamaCleaner,
    should_clean,
)

EXPECTED_FEW_SHOT_USER_CONTENTS = (
    "[tone: standard]\n"
    "um so basically I think we should uh move the deadline to friday",
    "[tone: standard]\nthe meeting is at three no wait four thirty",
    "[tone: standard]\nhi sarah comma new paragraph thanks for the update period",
    "[tone: standard]\nwhat time does the deploy finish",
    "[tone: casual]\nsounds good um see you then",
    "[tone: standard] [preserve: Kubernetes]\n"
    "we need to restart the cooper netties cluster",
)

EXPECTED_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {"cleaned_text": {"type": "string"}},
    "required": ["cleaned_text"],
}


def _fake_response(
    content: str | None = None,
    http_error: Exception | None = None,
    json_body: Any = None,
) -> Mock:
    """Build a Mock standing in for a requests.Response.

    Args:
        content: The message.content string of a well-formed chat response.
        http_error: If given, raise_for_status() raises this.
        json_body: If given, overrides the whole body returned by .json().
    """
    response = Mock()
    if http_error is not None:
        response.raise_for_status.side_effect = http_error
    else:
        response.raise_for_status.return_value = None
    if json_body is not None:
        response.json.return_value = json_body
    elif content is not None:
        response.json.return_value = {"message": {"content": content}}
    return response


def _ok_post(monkeypatch: pytest.MonkeyPatch, cleaned_text: str = "ok") -> Mock:
    """Patch requests.post with a well-formed cleanup response; return the mock."""
    response = _fake_response(content=json.dumps({"cleaned_text": cleaned_text}))
    post = Mock(return_value=response)
    monkeypatch.setattr(cleanup.requests, "post", post)
    return post


@pytest.fixture
def cleaner() -> OllamaCleaner:
    return OllamaCleaner(
        url="http://localhost:11434", model="qwen2.5:7b", timeout_seconds=4.0
    )


class TestShouldClean:
    def test_below_min_chars_is_false(self) -> None:
        assert should_clean("short text", min_chars=50) is False

    def test_at_min_chars_is_true(self) -> None:
        text = "a" * 50
        assert should_clean(text, min_chars=50) is True

    def test_above_min_chars_is_true(self) -> None:
        text = "a" * 51
        assert should_clean(text, min_chars=50) is True

    def test_strips_whitespace_before_measuring(self) -> None:
        text = "   " + ("a" * 50) + "   "
        assert should_clean(text, min_chars=50) is True

    def test_whitespace_only_is_false(self) -> None:
        assert should_clean("   ", min_chars=1) is False


class TestFewShotExamples:
    def test_has_exactly_six_examples(self) -> None:
        assert len(FEW_SHOT_EXAMPLES) == 6

    def test_five_standard_and_one_casual(self) -> None:
        tones = [tone for tone, _raw, _cleaned in FEW_SHOT_EXAMPLES]
        assert tones.count("standard") == 5
        assert tones.count("casual") == 1

    def test_kubernetes_example_present(self) -> None:
        cleaned_texts = [cleaned for _tone, _raw, cleaned in FEW_SHOT_EXAMPLES]
        assert "We need to restart the Kubernetes cluster." in cleaned_texts


class TestClean:
    def test_happy_path_returns_cleaned_text(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ok_post(monkeypatch, cleaned_text="Hello world.")

        result = cleaner.clean(CleanupRequest(text="um hello world"))

        assert result == "Hello world."

    def test_payload_shape(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        post = _ok_post(monkeypatch)

        cleaner.clean(CleanupRequest(text="the real transcript"))

        assert post.call_count == 1
        args, kwargs = post.call_args
        assert args[0] == "http://localhost:11434/api/chat"
        payload = kwargs["json"]
        assert payload["model"] == "qwen2.5:7b"
        assert payload["stream"] is False
        assert payload["options"] == {"temperature": 0}
        assert payload["keep_alive"] == "60m"
        assert payload["format"] == EXPECTED_FORMAT_SCHEMA
        assert kwargs["timeout"] == 4.0 + cleanup.TIMEOUT_PER_CHAR_SECONDS * len(
            "the real transcript"
        )

        messages = payload["messages"]
        assert messages[0]["role"] == "system"
        assert messages[-1] == {
            "role": "user",
            "content": "[tone: standard]\nthe real transcript",
        }

        few_shot_messages = messages[1:-1]
        assert len(few_shot_messages) == len(FEW_SHOT_EXAMPLES) * 2
        expected = zip(FEW_SHOT_EXAMPLES, EXPECTED_FEW_SHOT_USER_CONTENTS, strict=True)
        for i, ((_tone, _raw, cleaned_text), user_content) in enumerate(expected):
            user_msg = few_shot_messages[i * 2]
            assistant_msg = few_shot_messages[i * 2 + 1]
            assert user_msg == {"role": "user", "content": user_content}
            assert assistant_msg["role"] == "assistant"
            assert json.loads(assistant_msg["content"]) == {
                "cleaned_text": cleaned_text
            }

    def test_prefix_identical_across_tone_and_dictionary(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        post = _ok_post(monkeypatch)

        cleaner.clean(CleanupRequest(text="first transcript"))
        cleaner.clean(
            CleanupRequest(
                text="second transcript",
                tone="casual",
                dictionary_words=("Kubernetes", "Baseten"),
            )
        )

        first = post.call_args_list[0].kwargs["json"]["messages"]
        second = post.call_args_list[1].kwargs["json"]["messages"]
        assert len(first) == 1 + len(FEW_SHOT_EXAMPLES) * 2 + 1
        assert first[:-1] == second[:-1]
        assert all(
            "second transcript" not in message["content"] for message in second[:-1]
        )

    def test_dictionary_words_tagged_in_final_user_turn(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        post = _ok_post(monkeypatch)

        cleaner.clean(
            CleanupRequest(
                text="restart the cluster",
                dictionary_words=("Kubernetes", "Baseten"),
            )
        )

        messages = post.call_args.kwargs["json"]["messages"]
        assert messages[-1]["content"] == (
            "[tone: standard] [preserve: Kubernetes, Baseten]\nrestart the cluster"
        )
        assert (
            "Preserve these exact spellings when the speaker says them: "
            "Kubernetes, Baseten." not in messages[0]["content"]
        )

    def test_no_dictionary_words_omits_preserve_tag(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        post = _ok_post(monkeypatch)

        cleaner.clean(CleanupRequest(text="restart the cluster"))

        final_content = post.call_args.kwargs["json"]["messages"][-1]["content"]
        assert final_content == "[tone: standard]\nrestart the cluster"
        assert "[preserve:" not in final_content

    def test_casual_tone_tagged_in_final_user_turn(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        post = _ok_post(monkeypatch)

        cleaner.clean(CleanupRequest(text="sounds good", tone="casual"))

        final_content = post.call_args.kwargs["json"]["messages"][-1]["content"]
        assert final_content == "[tone: casual]\nsounds good"

    def test_system_prompt_is_constant_and_defines_both_tones(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        post = _ok_post(monkeypatch)

        cleaner.clean(CleanupRequest(text="anything", tone="casual"))

        system_content = post.call_args.kwargs["json"]["messages"][0]["content"]
        assert 'Tone "standard": normal sentence punctuation.' in system_content
        assert "no trailing period on the final sentence" in system_content
        assert "[preserve:" in system_content
        assert "preserve these exact spellings when the speaker says them" in (
            system_content
        )

    def test_strips_whitespace_from_cleaned_text(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ok_post(monkeypatch, cleaned_text="  Hello.  ")

        result = cleaner.clean(CleanupRequest(text="hello"))

        assert result == "Hello."

    def test_timeout_returns_raw_text(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cleanup.requests, "post", Mock(side_effect=requests.exceptions.Timeout)
        )

        result = cleaner.clean(CleanupRequest(text="raw transcript text"))

        assert result == "raw transcript text"

    def test_connection_error_returns_raw_text(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cleanup.requests,
            "post",
            Mock(side_effect=requests.exceptions.ConnectionError),
        )

        result = cleaner.clean(CleanupRequest(text="raw transcript text"))

        assert result == "raw transcript text"

    def test_http_error_returns_raw_text(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _fake_response(http_error=requests.exceptions.HTTPError("500"))
        monkeypatch.setattr(cleanup.requests, "post", Mock(return_value=response))

        result = cleaner.clean(CleanupRequest(text="raw transcript text"))

        assert result == "raw transcript text"

    def test_non_json_content_returns_raw_text(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _fake_response(content="this is not json")
        monkeypatch.setattr(cleanup.requests, "post", Mock(return_value=response))

        result = cleaner.clean(CleanupRequest(text="raw transcript text"))

        assert result == "raw transcript text"

    def test_missing_cleaned_text_key_returns_raw_text(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _fake_response(content=json.dumps({"unexpected": "value"}))
        monkeypatch.setattr(cleanup.requests, "post", Mock(return_value=response))

        result = cleaner.clean(CleanupRequest(text="raw transcript text"))

        assert result == "raw transcript text"

    def test_missing_message_key_returns_raw_text(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _fake_response(json_body={"unrelated": "shape"})
        monkeypatch.setattr(cleanup.requests, "post", Mock(return_value=response))

        result = cleaner.clean(CleanupRequest(text="raw transcript text"))

        assert result == "raw transcript text"

    def test_non_dict_body_returns_raw_text(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _fake_response(json_body=["not", "a", "dict"])
        monkeypatch.setattr(cleanup.requests, "post", Mock(return_value=response))

        result = cleaner.clean(CleanupRequest(text="raw transcript text"))

        assert result == "raw transcript text"

    def test_non_string_cleaned_text_returns_raw_text(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _fake_response(content=json.dumps({"cleaned_text": 123}))
        monkeypatch.setattr(cleanup.requests, "post", Mock(return_value=response))

        result = cleaner.clean(CleanupRequest(text="raw transcript text"))

        assert result == "raw transcript text"


class TestWarmUp:
    def test_success_returns_true(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ok_post(monkeypatch)

        assert cleaner.warm_up() is True

    def test_sends_same_prefix_as_clean(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        post = _ok_post(monkeypatch)

        cleaner.warm_up()
        cleaner.clean(CleanupRequest(text="a real transcript"))

        warm_up_messages = post.call_args_list[0].kwargs["json"]["messages"]
        clean_messages = post.call_args_list[1].kwargs["json"]["messages"]
        assert warm_up_messages[:-1] == clean_messages[:-1]
        assert warm_up_messages[-1] == {
            "role": "user",
            "content": "[tone: standard]\nhello",
        }

    def test_payload_matches_clean_settings_with_long_timeout(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        post = _ok_post(monkeypatch)

        cleaner.warm_up()

        args, kwargs = post.call_args
        assert args[0] == "http://localhost:11434/api/chat"
        assert kwargs["timeout"] == WARM_UP_TIMEOUT_SECONDS
        payload = kwargs["json"]
        assert payload["model"] == "qwen2.5:7b"
        assert payload["stream"] is False
        assert payload["options"] == {"temperature": 0}
        assert payload["keep_alive"] == "60m"
        assert payload["format"] == EXPECTED_FORMAT_SCHEMA

    def test_connection_error_returns_false(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cleanup.requests,
            "post",
            Mock(side_effect=requests.exceptions.ConnectionError),
        )

        assert cleaner.warm_up() is False

    def test_http_error_returns_false(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _fake_response(http_error=requests.exceptions.HTTPError("500"))
        monkeypatch.setattr(cleanup.requests, "post", Mock(return_value=response))

        assert cleaner.warm_up() is False

    def test_timeout_returns_false(
        self, cleaner: OllamaCleaner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cleanup.requests, "post", Mock(side_effect=requests.exceptions.Timeout)
        )

        assert cleaner.warm_up() is False
