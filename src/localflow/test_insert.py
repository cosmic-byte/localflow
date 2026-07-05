"""Tests for the text insertion layer.

Every test that would otherwise touch the real system (clipboard, synthetic key
events, notifications, or Carbon) monkeypatches the relevant module-level
primitive, so the suite never mutates the pasteboard, posts events, or shows
notifications.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from localflow import insert
from localflow.insert import FrontmostApp, InsertOutcome


class _CallRecorder:
    """Installs fake insertion primitives and records the order they run in."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, saved: str | None) -> None:
        self.events: list[tuple] = []
        self.__monkeypatch = monkeypatch
        self.__saved = saved
        monkeypatch.setattr(insert, "_read_clipboard", self.__read)
        monkeypatch.setattr(insert, "_write_clipboard", self.__write)
        monkeypatch.setattr(insert, "_post_paste_event", self.__paste)
        monkeypatch.setattr(insert, "_start_timer", self.__start_timer)
        monkeypatch.setattr(insert, "notify", self.__notify)

    def set_environment(self, trusted: bool, secure_input: bool) -> None:
        self.__monkeypatch.setattr(insert, "is_accessibility_trusted", lambda: trusted)
        self.__monkeypatch.setattr(
            insert, "is_secure_input_active", lambda: secure_input
        )

    def __read(self) -> str | None:
        self.events.append(("read",))
        return self.__saved

    def __write(self, text: str) -> None:
        self.events.append(("write", text))

    def __paste(self) -> None:
        self.events.append(("paste",))

    def __start_timer(self, delay: float, func: Callable[[], None]) -> None:
        self.events.append(("schedule", delay))
        func()

    def __notify(self, title: str, message: str) -> None:
        self.events.append(("notify", title, message))


# --- tone_for_app -----------------------------------------------------------


@pytest.mark.parametrize("bundle_id", sorted(insert.CASUAL_BUNDLE_IDS))
def test_tone_for_app_casual(bundle_id: str) -> None:
    app = FrontmostApp(bundle_id=bundle_id, name="Messaging")
    assert insert.tone_for_app(app) == "casual"


def test_tone_for_app_default() -> None:
    app = FrontmostApp(bundle_id="com.apple.dt.Xcode", name="Xcode")
    assert insert.tone_for_app(app) == "standard"


def test_tone_for_app_empty_bundle_id() -> None:
    app = FrontmostApp(bundle_id="", name="")
    assert insert.tone_for_app(app) == "standard"


def test_tone_for_app_none() -> None:
    assert insert.tone_for_app(None) == "standard"


# --- resolve_strategy -------------------------------------------------------


@pytest.mark.parametrize(
    ("auto_paste", "trusted", "secure_input", "expected"),
    [
        (True, True, False, InsertOutcome.PASTED),
        (False, True, False, InsertOutcome.CLIPBOARD_ONLY),
        (True, False, False, InsertOutcome.CLIPBOARD_ONLY),
        (True, True, True, InsertOutcome.CLIPBOARD_ONLY),
        (False, False, False, InsertOutcome.CLIPBOARD_ONLY),
        (False, True, True, InsertOutcome.CLIPBOARD_ONLY),
        (True, False, True, InsertOutcome.CLIPBOARD_ONLY),
        (False, False, True, InsertOutcome.CLIPBOARD_ONLY),
    ],
)
def test_resolve_strategy(
    auto_paste: bool, trusted: bool, secure_input: bool, expected: InsertOutcome
) -> None:
    assert insert.resolve_strategy(auto_paste, trusted, secure_input) == expected


# --- insert_text: paste path ------------------------------------------------


def test_insert_text_paste_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _CallRecorder(monkeypatch, saved="OLD CLIPBOARD")
    recorder.set_environment(trusted=True, secure_input=False)

    outcome = insert.insert_text("hello", auto_paste=True)

    assert outcome is InsertOutcome.PASTED
    assert recorder.events == [
        ("read",),
        ("write", "hello"),
        ("paste",),
        ("schedule", insert.RESTORE_DELAY_SECONDS),
        ("write", "OLD CLIPBOARD"),
    ]


def test_insert_text_paste_no_saved_clipboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _CallRecorder(monkeypatch, saved=None)
    recorder.set_environment(trusted=True, secure_input=False)

    outcome = insert.insert_text("hello", auto_paste=True)

    assert outcome is InsertOutcome.PASTED
    assert recorder.events == [
        ("read",),
        ("write", "hello"),
        ("paste",),
    ]
    assert not any(event[0] == "schedule" for event in recorder.events)


# --- insert_text: clipboard-only paths --------------------------------------


@pytest.mark.parametrize(
    ("auto_paste", "trusted", "secure_input"),
    [
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_insert_text_clipboard_only(
    monkeypatch: pytest.MonkeyPatch,
    auto_paste: bool,
    trusted: bool,
    secure_input: bool,
) -> None:
    recorder = _CallRecorder(monkeypatch, saved="OLD CLIPBOARD")
    recorder.set_environment(trusted=trusted, secure_input=secure_input)

    outcome = insert.insert_text("hello", auto_paste=auto_paste)

    assert outcome is InsertOutcome.CLIPBOARD_ONLY
    assert ("write", "hello") in recorder.events
    assert any(event[0] == "notify" for event in recorder.events)
    assert not any(event[0] == "read" for event in recorder.events)
    assert not any(event[0] == "paste" for event in recorder.events)
    assert not any(event[0] == "schedule" for event in recorder.events)


# --- notify -----------------------------------------------------------------


def test_notify_runs_osascript(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return None

    monkeypatch.setattr(insert.subprocess, "run", fake_run)
    insert.notify("localflow", "done")

    assert captured["cmd"][0] == "osascript"
    script = captured["cmd"][-1]
    assert "done" in script
    assert "localflow" in script


def test_notify_escapes_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(
        insert.subprocess, "run", lambda cmd, **kwargs: captured.setdefault("cmd", cmd)
    )
    insert.notify("localflow", 'say "hi"')

    script = captured["cmd"][-1]
    assert '\\"hi\\"' in script


def test_notify_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(cmd, **kwargs):
        raise OSError("osascript missing")

    monkeypatch.setattr(insert.subprocess, "run", boom)
    insert.notify("localflow", "done")  # must not raise


# --- is_secure_input_active -------------------------------------------------


class _FakeCFunc:
    def __init__(self, value: int) -> None:
        self.restype = None
        self.__value = value

    def __call__(self) -> int:
        return self.__value


class _FakeCarbon:
    def __init__(self, value: int) -> None:
        self.IsSecureEventInputEnabled = _FakeCFunc(value)


def test_is_secure_input_active_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(insert.ctypes, "CDLL", lambda path: _FakeCarbon(1))
    assert insert.is_secure_input_active() is True


def test_is_secure_input_active_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(insert.ctypes, "CDLL", lambda path: _FakeCarbon(0))
    assert insert.is_secure_input_active() is False


def test_is_secure_input_active_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(path: str):
        raise OSError("cannot load Carbon")

    monkeypatch.setattr(insert.ctypes, "CDLL", boom)
    assert insert.is_secure_input_active() is False


# --- get_frontmost_app ------------------------------------------------------


class _FakeNSApp:
    def __init__(self, bundle_id: str | None, name: str | None) -> None:
        self.__bundle_id = bundle_id
        self.__name = name

    def bundleIdentifier(self) -> str | None:
        return self.__bundle_id

    def localizedName(self) -> str | None:
        return self.__name


class _FakeWorkspace:
    def __init__(self, app: object) -> None:
        self.__app = app

    def frontmostApplication(self) -> object:
        return self.__app


def _install_workspace(monkeypatch: pytest.MonkeyPatch, app: object) -> None:
    class _NSWorkspace:
        @staticmethod
        def sharedWorkspace() -> _FakeWorkspace:
            return _FakeWorkspace(app)

    monkeypatch.setattr(insert, "NSWorkspace", _NSWorkspace)


def test_get_frontmost_app(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_workspace(monkeypatch, _FakeNSApp("com.apple.Safari", "Safari"))
    app = insert.get_frontmost_app()
    assert app == FrontmostApp(bundle_id="com.apple.Safari", name="Safari")


def test_get_frontmost_app_none_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_workspace(monkeypatch, _FakeNSApp(None, None))
    app = insert.get_frontmost_app()
    assert app == FrontmostApp(bundle_id="", name="")


def test_get_frontmost_app_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_workspace(monkeypatch, None)
    assert insert.get_frontmost_app() is None
