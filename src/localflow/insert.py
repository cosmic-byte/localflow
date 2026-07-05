"""Text insertion into the focused macOS app.

The public surface is a small ladder: decide whether the transcript can be
pasted into the frontmost application or only placed on the clipboard, then act
on that decision. The side-effectful primitives (reading and writing the
pasteboard, posting the Cmd-V key event, scheduling the clipboard restore, and
sending a user notification) are module-level functions so that tests can
monkeypatch them without touching the real system.
"""

from __future__ import annotations

import ctypes
import logging
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

from AppKit import NSPasteboard, NSPasteboardTypeString, NSWorkspace
from ApplicationServices import AXIsProcessTrusted
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPost,
    CGEventSetFlags,
    kCGEventFlagMaskCommand,
    kCGHIDEventTap,
)

log = logging.getLogger(__name__)

CASUAL_BUNDLE_IDS: frozenset[str] = frozenset(
    {
        "com.tinyspeck.slackmacgap",
        "com.hnc.Discord",
        "com.apple.MobileSMS",
        "net.whatsapp.WhatsApp",
        "ru.keepcoder.Telegram",
    }
)

RESTORE_DELAY_SECONDS = 0.6

_V_KEYCODE = 9
_PASTE_KEY_DELAY_SECONDS = 0.02
_CARBON_FRAMEWORK = "/System/Library/Frameworks/Carbon.framework/Carbon"
_NOTIFY_TITLE = "localflow"
_CLIPBOARD_ONLY_MESSAGE = "Text copied to the clipboard. Press Cmd-V to paste."


class InsertOutcome(Enum):
    """Result of an insertion attempt."""

    PASTED = auto()
    CLIPBOARD_ONLY = auto()


@dataclass
class FrontmostApp:
    """Identity of the application that currently has focus.

    Attributes:
        bundle_id: The application's bundle identifier, or an empty string when
            the system does not report one.
        name: The application's localized display name, or an empty string when
            the system does not report one.
    """

    bundle_id: str
    name: str


def get_frontmost_app() -> FrontmostApp | None:
    """Return the frontmost application, or None if it cannot be determined.

    Returns:
        A FrontmostApp describing the focused application, or None when
        NSWorkspace reports no frontmost application.
    """
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        return None
    return FrontmostApp(
        bundle_id=app.bundleIdentifier() or "",
        name=app.localizedName() or "",
    )


def tone_for_app(app: FrontmostApp | None) -> str:
    """Return the cleanup tone appropriate for the focused application.

    This is a pure function with no side effects.

    Args:
        app: The focused application, or None when it is unknown.

    Returns:
        "casual" when the application is a known messaging app, otherwise
        "standard".
    """
    if app is not None and app.bundle_id in CASUAL_BUNDLE_IDS:
        return "casual"
    return "standard"


def is_accessibility_trusted() -> bool:
    """Return whether this process is a trusted accessibility client.

    Returns:
        The value of AXIsProcessTrusted(), which is required before synthetic
        keyboard events will be delivered.
    """
    return bool(AXIsProcessTrusted())


def is_secure_input_active() -> bool:
    """Return whether macOS secure event input is currently enabled.

    Secure input (used by password fields) suppresses synthetic key events, so
    pasting via Cmd-V will not work while it is active. Any failure to query the
    Carbon framework is treated as "not active".

    Returns:
        True when IsSecureEventInputEnabled reports an active secure input
        session, False otherwise or on any failure.
    """
    try:
        carbon = ctypes.CDLL(_CARBON_FRAMEWORK)
        func = carbon.IsSecureEventInputEnabled
        func.restype = ctypes.c_bool
        return bool(func())
    except (OSError, AttributeError):
        return False


def resolve_strategy(
    auto_paste: bool, trusted: bool, secure_input: bool
) -> InsertOutcome:
    """Decide how to deliver text given the current environment.

    This is a pure function so the ladder can be unit-tested in isolation.

    Args:
        auto_paste: Whether the user has enabled pasting into the focused app.
        trusted: Whether this process is a trusted accessibility client.
        secure_input: Whether macOS secure event input is active.

    Returns:
        InsertOutcome.PASTED when a Cmd-V paste should be attempted, otherwise
        InsertOutcome.CLIPBOARD_ONLY.
    """
    if not auto_paste or not trusted or secure_input:
        return InsertOutcome.CLIPBOARD_ONLY
    return InsertOutcome.PASTED


def insert_text(text: str, auto_paste: bool = True) -> InsertOutcome:
    """Insert text into the focused application.

    When pasting is not possible (auto_paste disabled, accessibility not
    trusted, or secure input active) the text is placed on the clipboard and the
    user is notified. Otherwise the current clipboard string is saved, the text
    is written to the clipboard, a Cmd-V key event is posted, and the saved
    string is restored on a timer thread roughly RESTORE_DELAY_SECONDS later.
    The clipboard is only restored when there was a saved string to restore.

    Args:
        text: The text to insert.
        auto_paste: Whether to attempt a paste into the focused app.

    Returns:
        InsertOutcome.PASTED when a paste was attempted, otherwise
        InsertOutcome.CLIPBOARD_ONLY.
    """
    outcome = resolve_strategy(
        auto_paste=auto_paste,
        trusted=is_accessibility_trusted(),
        secure_input=is_secure_input_active(),
    )
    if outcome is InsertOutcome.CLIPBOARD_ONLY:
        _write_clipboard(text)
        notify(_NOTIFY_TITLE, _CLIPBOARD_ONLY_MESSAGE)
        return outcome

    saved = _read_clipboard()
    _write_clipboard(text)
    _post_paste_event()
    if saved is not None:
        _start_timer(RESTORE_DELAY_SECONDS, lambda: _write_clipboard(saved))
    return InsertOutcome.PASTED


def notify(title: str, message: str) -> None:
    """Show a user notification via osascript.

    This never raises; any failure to run osascript is logged and swallowed.

    Args:
        title: The notification title.
        message: The notification body.
    """
    script = (
        f'display notification "{_escape_applescript(message)}" '
        f'with title "{_escape_applescript(title)}"'
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        log.exception("Failed to display notification")


def _escape_applescript(value: str) -> str:
    """Escape backslashes and double quotes for embedding in an AppleScript string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _read_clipboard() -> str | None:
    """Return the general pasteboard's string contents, or None when absent."""
    value = NSPasteboard.generalPasteboard().stringForType_(NSPasteboardTypeString)
    return None if value is None else str(value)


def _write_clipboard(text: str) -> None:
    """Replace the general pasteboard's contents with the given string."""
    pasteboard = NSPasteboard.generalPasteboard()
    pasteboard.clearContents()
    pasteboard.setString_forType_(text, NSPasteboardTypeString)


def _post_paste_event() -> None:
    """Post a synthetic Cmd-V keystroke via CGEvent to the HID event tap."""
    key_down = CGEventCreateKeyboardEvent(None, _V_KEYCODE, True)
    key_up = CGEventCreateKeyboardEvent(None, _V_KEYCODE, False)
    if key_down is None or key_up is None:
        log.error("CGEventCreateKeyboardEvent returned None; paste not sent")
        return
    CGEventSetFlags(key_down, kCGEventFlagMaskCommand)
    CGEventSetFlags(key_up, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, key_down)
    time.sleep(_PASTE_KEY_DELAY_SECONDS)
    CGEventPost(kCGHIDEventTap, key_up)


def _start_timer(delay: float, func: Callable[[], None]) -> None:
    """Run func after delay seconds on a daemon timer thread."""
    timer = threading.Timer(delay, func)
    timer.daemon = True
    timer.start()
