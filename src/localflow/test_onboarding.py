"""Tests for the onboarding page content and the onboarding-done config flag.

The page content is plain data, so these tests need no running GUI. The
controller's panel behavior (Next/Back/Skip wiring) is AppKit glue verified
manually; what matters here is that the tour text reflects the configured
hotkey and that the done flag survives a save/load round trip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localflow import config as config_module
from localflow.config import AppConfig
from localflow.onboarding import build_pages


def test_pages_are_non_empty_and_ordered() -> None:
    pages = build_pages(AppConfig())

    assert len(pages) >= 4
    assert all(page.title and page.body for page in pages)
    assert pages[0].title == "Dictate anywhere"
    assert pages[-1].title == "Permissions"


def test_pages_mention_default_hotkey() -> None:
    pages = build_pages(AppConfig())

    assert any("fn" in page.body for page in pages)


def test_pages_use_configured_hotkey() -> None:
    pages = build_pages(AppConfig(hotkey="ctrl+alt+space"))

    bodies = " ".join(page.body for page in pages)
    assert "ctrl+alt+space" in bodies


def test_pages_cover_help_replay_and_permissions() -> None:
    bodies = " ".join(page.body for page in build_pages(AppConfig()))

    assert "Help" in bodies
    assert "Microphone" in bodies
    assert "Accessibility" in bodies
    assert "Input Monitoring" in bodies


def test_onboarding_done_defaults_false() -> None:
    assert AppConfig().onboarding_done is False


def test_onboarding_done_round_trips_through_save_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.json")

    config = AppConfig()
    config.onboarding_done = True
    config.save()

    assert AppConfig.load().onboarding_done is True
