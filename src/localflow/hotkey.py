"""Global hotkey layer for localflow.

This module provides three pieces:

HotkeySpec parses a textual hotkey ("fn", "right_cmd", "ctrl+alt+space") into a
keycode plus required modifier flags.

TapDecisionEngine is a pure, Quartz-free state machine that turns raw key
down/up transitions into hold, single-tap and double-tap gestures. It is fully
unit-testable via an injectable timer factory.

HotkeyListener wires a Quartz CGEventTap to a TapDecisionEngine, translating
system key events for the configured hotkey into engine transitions.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from Quartz import (
    CFMachPortCreateRunLoopSource,
    CFRunLoopAddSource,
    CFRunLoopGetMain,
    CFRunLoopRemoveSource,
    CGEventGetFlags,
    CGEventGetIntegerValueField,
    CGEventMaskBit,
    CGEventTapCreate,
    CGEventTapEnable,
    kCFRunLoopCommonModes,
    kCGEventFlagMaskAlternate,
    kCGEventFlagMaskCommand,
    kCGEventFlagMaskControl,
    kCGEventFlagMaskSecondaryFn,
    kCGEventFlagMaskShift,
    kCGEventFlagsChanged,
    kCGEventKeyDown,
    kCGEventKeyUp,
    kCGEventTapDisabledByTimeout,
    kCGEventTapDisabledByUserInput,
    kCGEventTapOptionDefault,
    kCGHeadInsertEventTap,
    kCGHIDEventTap,
    kCGKeyboardEventKeycode,
)

log = logging.getLogger(__name__)

FN_KEYCODE = 63
RIGHT_CMD_KEYCODE = 54
RIGHT_ALT_KEYCODE = 61

_MODIFIER_KEY_KEYCODES: dict[str, int] = {
    "fn": FN_KEYCODE,
    "right_cmd": RIGHT_CMD_KEYCODE,
    "right_alt": RIGHT_ALT_KEYCODE,
}

_MODIFIER_ALIASES: dict[str, str] = {
    "cmd": "cmd",
    "command": "cmd",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "opt": "alt",
    "option": "alt",
    "shift": "shift",
}

_MODIFIER_FLAG_BITS: dict[str, int] = {
    "cmd": kCGEventFlagMaskCommand,
    "ctrl": kCGEventFlagMaskControl,
    "alt": kCGEventFlagMaskAlternate,
    "shift": kCGEventFlagMaskShift,
}

_MODIFIER_KEY_FLAG_BITS: dict[int, int] = {
    FN_KEYCODE: kCGEventFlagMaskSecondaryFn,
    RIGHT_CMD_KEYCODE: kCGEventFlagMaskCommand,
    RIGHT_ALT_KEYCODE: kCGEventFlagMaskAlternate,
}

_KEY_CODES: dict[str, int] = {
    "a": 0,
    "b": 11,
    "c": 8,
    "d": 2,
    "e": 14,
    "f": 3,
    "g": 5,
    "h": 4,
    "i": 34,
    "j": 38,
    "k": 40,
    "l": 37,
    "m": 46,
    "n": 45,
    "o": 31,
    "p": 35,
    "q": 12,
    "r": 15,
    "s": 1,
    "t": 17,
    "u": 32,
    "v": 9,
    "w": 13,
    "x": 7,
    "y": 16,
    "z": 6,
    "0": 29,
    "1": 18,
    "2": 19,
    "3": 20,
    "4": 21,
    "5": 23,
    "6": 22,
    "7": 26,
    "8": 28,
    "9": 25,
    "space": 49,
    "return": 36,
    "enter": 36,
    "tab": 48,
    "escape": 53,
    "esc": 53,
    "delete": 51,
}


@dataclass(frozen=True)
class HotkeySpec:
    """Parsed description of a global hotkey.

    Attributes:
        keycode: The macOS virtual keycode of the primary key.
        modifiers: Required modifier names, a subset of {"cmd", "ctrl", "alt",
            "shift"}. Always empty for bare-modifier hotkeys.
        is_modifier_key: True when the hotkey is a bare modifier such as fn,
            right_cmd or right_alt, which arrive as flagsChanged events rather
            than key down/up events.
    """

    keycode: int
    modifiers: frozenset[str]
    is_modifier_key: bool

    @classmethod
    def from_string(cls, spec: str) -> HotkeySpec:
        """Parse a textual hotkey specification.

        Accepts bare-modifier names ("fn", "right_cmd", "right_alt") and
        modifier-plus-key combos ("ctrl+alt+space", "cmd+shift+v"). Modifier
        aliases such as "opt"/"option" for alt and "command" for cmd are
        accepted.

        Args:
            spec: The hotkey string to parse.

        Returns:
            The parsed HotkeySpec.

        Raises:
            ValueError: If the specification is empty, names an unknown
                modifier, or names an unknown primary key.
        """
        if not spec or not spec.strip():
            raise ValueError("Empty hotkey specification")
        normalized = spec.strip().lower()
        if normalized in _MODIFIER_KEY_KEYCODES:
            return cls(
                keycode=_MODIFIER_KEY_KEYCODES[normalized],
                modifiers=frozenset(),
                is_modifier_key=True,
            )
        parts = [part.strip() for part in normalized.split("+") if part.strip()]
        if not parts:
            raise ValueError(f"Invalid hotkey specification: {spec!r}")
        *modifier_tokens, key_token = parts
        modifiers: set[str] = set()
        for token in modifier_tokens:
            canonical = _MODIFIER_ALIASES.get(token)
            if canonical is None:
                raise ValueError(f"Unknown modifier {token!r} in hotkey {spec!r}")
            modifiers.add(canonical)
        keycode = _KEY_CODES.get(key_token)
        if keycode is None:
            raise ValueError(f"Unknown key {key_token!r} in hotkey {spec!r}")
        return cls(
            keycode=keycode,
            modifiers=frozenset(modifiers),
            is_modifier_key=False,
        )


def _default_timer_factory(
    delay: float, callback: Callable[[], None]
) -> threading.Timer:
    """Create and start a daemon threading.Timer.

    Args:
        delay: Delay in seconds before the callback fires.
        callback: The zero-argument callable to invoke when the timer fires.

    Returns:
        The started timer, which exposes a cancel() method.
    """
    timer = threading.Timer(delay, callback)
    timer.daemon = True
    timer.start()
    return timer


class TapDecisionEngine:
    """Pure hold/tap/double-tap state machine.

    The engine consumes raw key_down/key_up transitions and fires one of four
    callbacks. It performs no Quartz work and schedules all delayed decisions
    through an injectable timer factory so it is fully deterministic in tests.

    Firing rules:
        on_hold_start fires once a press has been held for hold_threshold
        seconds, driven by a scheduled timer.
        on_hold_end fires on release, but only after on_hold_start has fired for
        that press.
        on_single_tap fires when a press shorter than hold_threshold is not
        followed by a second press within double_tap_window seconds. The window
        is measured from the release of the first tap and is enforced by a
        scheduled timer.
        on_double_tap fires when a second short press begins within
        double_tap_window of the first tap's release and is itself released
        before hold_threshold.

    Edge-case semantics:
        A release before hold_threshold cancels the pending hold and is treated
        as a tap.
        Duplicate key_down events while already pressed (keyboard auto-repeat)
        are ignored.
        Tap-then-hold: if the second press within the double-tap window is held
        past hold_threshold instead of released quickly, the pending single tap
        is discarded and the press is treated as a fresh hold (on_hold_start
        then on_hold_end). The first tap produces no callback in this case.
        The double-tap decision is therefore only committed on release of the
        second press, never on its key_down.

    Every callback invocation is wrapped so that an exception raised by a
    callback is logged and never propagated, leaving the engine consistent.
    """

    def __init__(
        self,
        on_hold_start: Callable[[], None],
        on_hold_end: Callable[[], None],
        on_single_tap: Callable[[], None],
        on_double_tap: Callable[[], None],
        hold_threshold: float = 0.35,
        double_tap_window: float = 0.4,
        timer_factory: Callable[[float, Callable[[], None]], object] | None = None,
    ) -> None:
        """Initialize the engine.

        Args:
            on_hold_start: Called when a press is held past hold_threshold.
            on_hold_end: Called on release after a hold has started.
            on_single_tap: Called for a lone short tap.
            on_double_tap: Called for two short taps within double_tap_window.
            hold_threshold: Seconds a press must be held to count as a hold.
            double_tap_window: Seconds allowed between a tap's release and the
                next press for the pair to count as a double tap.
            timer_factory: Factory that schedules a callback after a delay and
                returns a timer object exposing cancel(). Defaults to a daemon
                threading.Timer.
        """
        self.__on_hold_start = on_hold_start
        self.__on_hold_end = on_hold_end
        self.__on_single_tap = on_single_tap
        self.__on_double_tap = on_double_tap
        self.__hold_threshold = hold_threshold
        self.__double_tap_window = double_tap_window
        self.__timer_factory = timer_factory or _default_timer_factory

        self.__lock = threading.Lock()
        self.__pressed = False
        self.__hold_started = False
        self.__is_second_press = False
        self.__tap_pending = False
        self.__hold_timer: object | None = None
        self.__single_tap_timer: object | None = None
        self.__hold_gen = 0
        self.__tap_gen = 0

    def key_down(self, timestamp: float) -> None:
        """Record a key-press transition.

        Duplicate presses while already held are ignored so keyboard
        auto-repeat cannot start spurious gestures. A press that arrives while a
        prior tap is pending is marked as the second half of a potential double
        tap; the pending single-tap timer is cancelled and the double-tap versus
        fresh-hold decision is deferred to the matching release.

        Args:
            timestamp: Monotonic time of the event. Accepted for interface
                compatibility; gesture decisions are timer-driven.
        """
        del timestamp
        with self.__lock:
            if self.__pressed:
                return
            self.__pressed = True
            self.__hold_started = False
            if self.__tap_pending:
                self.__cancel_single_tap_timer()
                self.__tap_pending = False
                self.__is_second_press = True
            else:
                self.__is_second_press = False
            self.__start_hold_timer()

    def key_up(self, timestamp: float) -> None:
        """Record a key-release transition.

        Args:
            timestamp: Monotonic time of the event. Accepted for interface
                compatibility; gesture decisions are timer-driven.
        """
        del timestamp
        fire: Callable[[], None] | None = None
        with self.__lock:
            if not self.__pressed:
                return
            self.__pressed = False
            was_hold = self.__hold_started
            was_second = self.__is_second_press
            self.__hold_started = False
            self.__cancel_hold_timer()
            if was_hold:
                self.__is_second_press = False
                fire = self.__on_hold_end
            elif was_second:
                self.__is_second_press = False
                fire = self.__on_double_tap
            else:
                self.__tap_pending = True
                self.__start_single_tap_timer()
        if fire is not None:
            self.__safe_call(fire)

    def __hold_timer_fired(self, generation: int) -> None:
        """Fire on_hold_start if the press is still held and current.

        Args:
            generation: The hold generation captured when the timer was
                scheduled, used to ignore a timer that fired after being
                cancelled.
        """
        with self.__lock:
            if (
                generation != self.__hold_gen
                or not self.__pressed
                or self.__hold_started
            ):
                return
            self.__hold_started = True
            self.__hold_timer = None
            self.__is_second_press = False
        self.__safe_call(self.__on_hold_start)

    def __single_tap_timer_fired(self, generation: int) -> None:
        """Fire on_single_tap if no second press arrived within the window.

        Args:
            generation: The tap generation captured when the timer was
                scheduled, used to ignore a timer that fired after being
                cancelled.
        """
        with self.__lock:
            if generation != self.__tap_gen or not self.__tap_pending:
                return
            self.__tap_pending = False
            self.__single_tap_timer = None
        self.__safe_call(self.__on_single_tap)

    def __start_hold_timer(self) -> None:
        """Schedule the hold-detection timer for the current press."""
        self.__hold_gen += 1
        generation = self.__hold_gen
        self.__hold_timer = self.__timer_factory(
            self.__hold_threshold,
            lambda: self.__hold_timer_fired(generation),
        )

    def __cancel_hold_timer(self) -> None:
        """Cancel and invalidate any pending hold-detection timer."""
        timer = self.__hold_timer
        self.__hold_timer = None
        self.__hold_gen += 1
        if timer is not None:
            self.__cancel_timer(timer)

    def __start_single_tap_timer(self) -> None:
        """Schedule the single-tap timer that closes the double-tap window."""
        self.__tap_gen += 1
        generation = self.__tap_gen
        self.__single_tap_timer = self.__timer_factory(
            self.__double_tap_window,
            lambda: self.__single_tap_timer_fired(generation),
        )

    def __cancel_single_tap_timer(self) -> None:
        """Cancel and invalidate any pending single-tap timer."""
        timer = self.__single_tap_timer
        self.__single_tap_timer = None
        self.__tap_gen += 1
        if timer is not None:
            self.__cancel_timer(timer)

    @staticmethod
    def __cancel_timer(timer: object) -> None:
        """Cancel a timer object, logging any failure.

        Args:
            timer: A timer previously returned by the timer factory.
        """
        cancel = getattr(timer, "cancel", None)
        if cancel is None:
            return
        try:
            cancel()
        except Exception:
            log.exception("Failed to cancel hotkey timer")

    @staticmethod
    def __safe_call(callback: Callable[[], None]) -> None:
        """Invoke a callback, logging and swallowing any exception.

        Args:
            callback: The gesture callback to invoke.
        """
        try:
            callback()
        except Exception:
            log.exception("Hotkey gesture callback raised")


class HotkeyListener:
    """CGEventTap wiring that drives a TapDecisionEngine.

    The listener installs a Quartz event tap for the configured hotkey and
    feeds key down/up transitions into the engine. Bare-modifier hotkeys such
    as fn arrive as flagsChanged events and are passed through; normal-key
    combos are consumed so the keystroke does not reach the focused app.
    """

    def __init__(self, spec: HotkeySpec, engine: TapDecisionEngine) -> None:
        """Initialize the listener.

        Args:
            spec: The hotkey to listen for.
            engine: The decision engine that receives key transitions.
        """
        self.__spec = spec
        self.__engine = engine
        self.__tap: object | None = None
        self.__run_loop_source: object | None = None
        self.__combo_active = False
        self.__modifier_bit = _MODIFIER_KEY_FLAG_BITS.get(spec.keycode, 0)
        self.__required_mask = 0
        for modifier in spec.modifiers:
            self.__required_mask |= _MODIFIER_FLAG_BITS.get(modifier, 0)
        self.__callback = self.__tap_callback

    def start(self) -> bool:
        """Create the event tap and add it to the main run loop.

        Returns:
            True if the tap was created and installed, False when
            CGEventTapCreate returns None, which indicates the process lacks
            Accessibility or Input Monitoring permission.
        """
        mask = (
            CGEventMaskBit(kCGEventKeyDown)
            | CGEventMaskBit(kCGEventKeyUp)
            | CGEventMaskBit(kCGEventFlagsChanged)
        )
        tap = CGEventTapCreate(
            kCGHIDEventTap,
            kCGHeadInsertEventTap,
            kCGEventTapOptionDefault,
            mask,
            self.__callback,
            None,
        )
        if tap is None:
            log.warning(
                "CGEventTapCreate returned None; global hotkey requires "
                "Accessibility or Input Monitoring permission"
            )
            return False
        self.__tap = tap
        source = CFMachPortCreateRunLoopSource(None, tap, 0)
        self.__run_loop_source = source
        CFRunLoopAddSource(CFRunLoopGetMain(), source, kCFRunLoopCommonModes)
        CGEventTapEnable(tap, True)
        return True

    def stop(self) -> None:
        """Disable the tap and remove its run-loop source."""
        if self.__tap is not None:
            try:
                CGEventTapEnable(self.__tap, False)
            except Exception:
                log.exception("Failed to disable event tap")
        if self.__run_loop_source is not None:
            try:
                CFRunLoopRemoveSource(
                    CFRunLoopGetMain(),
                    self.__run_loop_source,
                    kCFRunLoopCommonModes,
                )
            except Exception:
                log.exception("Failed to remove run loop source")
        self.__tap = None
        self.__run_loop_source = None
        self.__combo_active = False

    def __tap_callback(
        self, proxy: object, event_type: int, event: object, refcon: object
    ) -> object | None:
        """Handle one CGEventTap event.

        Args:
            proxy: The event tap proxy (unused).
            event_type: The Quartz event type.
            event: The CGEvent.
            refcon: User data pointer (unused).

        Returns:
            The event to pass it through unchanged, or None to consume it.
        """
        del proxy, refcon
        if event_type in (
            kCGEventTapDisabledByTimeout,
            kCGEventTapDisabledByUserInput,
        ):
            if self.__tap is not None:
                CGEventTapEnable(self.__tap, True)
            return event
        try:
            return self.__dispatch(event_type, event)
        except Exception:
            log.exception("Hotkey event tap callback failed")
            return event

    def __dispatch(self, event_type: int, event: object) -> object | None:
        """Route an event to the engine and decide whether to consume it.

        Args:
            event_type: The Quartz event type.
            event: The CGEvent.

        Returns:
            The event to pass through, or None to consume it.
        """
        keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
        flags = CGEventGetFlags(event)
        timestamp = time.monotonic()
        if self.__spec.is_modifier_key:
            if event_type == kCGEventFlagsChanged and keycode == self.__spec.keycode:
                if flags & self.__modifier_bit:
                    self.__engine.key_down(timestamp)
                else:
                    self.__engine.key_up(timestamp)
            return event
        if keycode != self.__spec.keycode:
            return event
        if event_type == kCGEventKeyDown and self.__modifiers_satisfied(flags):
            self.__combo_active = True
            self.__engine.key_down(timestamp)
            return None
        if event_type == kCGEventKeyUp and self.__combo_active:
            self.__combo_active = False
            self.__engine.key_up(timestamp)
            return None
        return event

    def __modifiers_satisfied(self, flags: int) -> bool:
        """Check that all required modifier flags are present.

        Args:
            flags: The event's device-independent modifier flags.

        Returns:
            True if every required modifier bit is set.
        """
        return (flags & self.__required_mask) == self.__required_mask
