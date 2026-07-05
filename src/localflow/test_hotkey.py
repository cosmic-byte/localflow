"""Tests for the localflow global hotkey layer.

These tests exercise HotkeySpec parsing and the pure TapDecisionEngine state
machine. No CGEventTap is created; the engine is driven with a fake timer
factory so all delayed decisions are deterministic.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from localflow.hotkey import (
    FN_KEYCODE,
    RIGHT_ALT_KEYCODE,
    RIGHT_CMD_KEYCODE,
    HotkeySpec,
    TapDecisionEngine,
)


class FakeTimer:
    """A timer stub that records its callback instead of scheduling it."""

    def __init__(self, delay: float, callback: Callable[[], None]) -> None:
        """Store the requested delay and callback.

        Args:
            delay: The requested delay in seconds.
            callback: The callback that would fire after the delay.
        """
        self.delay = delay
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        """Mark this timer cancelled."""
        self.cancelled = True

    def fire(self) -> None:
        """Invoke the stored callback as if the delay elapsed."""
        self.callback()


class FakeClock:
    """A timer factory that records every scheduled FakeTimer."""

    def __init__(self) -> None:
        """Initialize with an empty timer list."""
        self.timers: list[FakeTimer] = []

    def __call__(self, delay: float, callback: Callable[[], None]) -> FakeTimer:
        """Create and record a FakeTimer.

        Args:
            delay: The requested delay in seconds.
            callback: The callback to store.

        Returns:
            The created FakeTimer.
        """
        timer = FakeTimer(delay, callback)
        self.timers.append(timer)
        return timer

    @property
    def live_timers(self) -> list[FakeTimer]:
        """Return timers that have not been cancelled."""
        return [timer for timer in self.timers if not timer.cancelled]


class Recorder:
    """Collects the sequence of gesture callbacks fired by the engine."""

    def __init__(self) -> None:
        """Initialize with an empty event list."""
        self.events: list[str] = []

    def make(self, name: str) -> Callable[[], None]:
        """Return a callback that appends its name when invoked.

        Args:
            name: The label recorded when the callback fires.

        Returns:
            The recording callback.
        """

        def _callback() -> None:
            self.events.append(name)

        return _callback


def make_engine(
    clock: FakeClock,
    recorder: Recorder,
    hold_threshold: float = 0.35,
    double_tap_window: float = 0.4,
) -> TapDecisionEngine:
    """Build an engine wired to a recorder and fake clock.

    Args:
        clock: The fake timer factory.
        recorder: The gesture recorder.
        hold_threshold: Hold detection threshold in seconds.
        double_tap_window: Double-tap window in seconds.

    Returns:
        A configured TapDecisionEngine.
    """
    return TapDecisionEngine(
        on_hold_start=recorder.make("hold_start"),
        on_hold_end=recorder.make("hold_end"),
        on_single_tap=recorder.make("single_tap"),
        on_double_tap=recorder.make("double_tap"),
        hold_threshold=hold_threshold,
        double_tap_window=double_tap_window,
        timer_factory=clock,
    )


def test_from_string_fn() -> None:
    spec = HotkeySpec.from_string("fn")
    assert spec.keycode == FN_KEYCODE
    assert spec.modifiers == frozenset()
    assert spec.is_modifier_key is True


def test_from_string_right_cmd() -> None:
    spec = HotkeySpec.from_string("right_cmd")
    assert spec.keycode == RIGHT_CMD_KEYCODE
    assert spec.modifiers == frozenset()
    assert spec.is_modifier_key is True


def test_from_string_right_alt() -> None:
    spec = HotkeySpec.from_string("right_alt")
    assert spec.keycode == RIGHT_ALT_KEYCODE
    assert spec.modifiers == frozenset()
    assert spec.is_modifier_key is True


def test_from_string_combo_ctrl_alt_space() -> None:
    spec = HotkeySpec.from_string("ctrl+alt+space")
    assert spec.keycode == 49
    assert spec.modifiers == frozenset({"ctrl", "alt"})
    assert spec.is_modifier_key is False


def test_from_string_combo_cmd_shift_v() -> None:
    spec = HotkeySpec.from_string("cmd+shift+v")
    assert spec.keycode == 9
    assert spec.modifiers == frozenset({"cmd", "shift"})
    assert spec.is_modifier_key is False


def test_from_string_modifier_aliases() -> None:
    spec = HotkeySpec.from_string("Option+Command+Control+space")
    assert spec.modifiers == frozenset({"alt", "cmd", "ctrl"})
    assert spec.keycode == 49


def test_from_string_unknown_key_raises() -> None:
    with pytest.raises(ValueError):
        HotkeySpec.from_string("ctrl+widget")


def test_from_string_unknown_modifier_raises() -> None:
    with pytest.raises(ValueError):
        HotkeySpec.from_string("hyper+v")


def test_from_string_empty_raises() -> None:
    with pytest.raises(ValueError):
        HotkeySpec.from_string("")


def test_from_string_whitespace_raises() -> None:
    with pytest.raises(ValueError):
        HotkeySpec.from_string("   ")


def test_hold_then_release() -> None:
    clock = FakeClock()
    recorder = Recorder()
    engine = make_engine(clock, recorder)

    engine.key_down(0.0)
    assert recorder.events == []
    # The hold timer fires, signalling the press is held.
    clock.timers[0].fire()
    assert recorder.events == ["hold_start"]

    engine.key_up(0.5)
    assert recorder.events == ["hold_start", "hold_end"]


def test_single_tap() -> None:
    clock = FakeClock()
    recorder = Recorder()
    engine = make_engine(clock, recorder)

    engine.key_down(0.0)
    engine.key_up(0.1)
    # The hold timer was cancelled; a single-tap timer is now pending.
    assert recorder.events == []
    single_tap_timer = clock.live_timers[-1]
    single_tap_timer.fire()
    assert recorder.events == ["single_tap"]


def test_hold_shorter_than_threshold_is_single_tap() -> None:
    clock = FakeClock()
    recorder = Recorder()
    engine = make_engine(clock, recorder)

    engine.key_down(0.0)
    hold_timer = clock.timers[0]
    engine.key_up(0.2)
    # Releasing before the hold timer fires must cancel it.
    assert hold_timer.cancelled is True
    assert recorder.events == []
    clock.live_timers[-1].fire()
    assert recorder.events == ["single_tap"]


def test_double_tap() -> None:
    clock = FakeClock()
    recorder = Recorder()
    engine = make_engine(clock, recorder)

    engine.key_down(0.0)
    engine.key_up(0.1)
    single_tap_timer = clock.live_timers[-1]

    engine.key_down(0.2)
    # The second press within the window cancels the pending single tap.
    assert single_tap_timer.cancelled is True
    engine.key_up(0.25)
    assert recorder.events == ["double_tap"]

    # A stale single-tap timer firing after cancellation must be ignored.
    single_tap_timer.fire()
    assert recorder.events == ["double_tap"]


def test_tap_then_hold_behaves_as_fresh_hold() -> None:
    clock = FakeClock()
    recorder = Recorder()
    engine = make_engine(clock, recorder)

    engine.key_down(0.0)
    engine.key_up(0.1)
    single_tap_timer = clock.live_timers[-1]

    # Second press within the window, but this time held past the threshold.
    engine.key_down(0.2)
    assert single_tap_timer.cancelled is True
    hold_timer = clock.live_timers[-1]
    hold_timer.fire()
    assert recorder.events == ["hold_start"]

    engine.key_up(0.7)
    assert recorder.events == ["hold_start", "hold_end"]

    # The original single tap must never fire, even if its stale timer runs.
    single_tap_timer.fire()
    assert recorder.events == ["hold_start", "hold_end"]


def test_key_repeat_duplicate_downs_ignored() -> None:
    clock = FakeClock()
    recorder = Recorder()
    engine = make_engine(clock, recorder)

    engine.key_down(0.0)
    engine.key_down(0.05)
    engine.key_down(0.10)
    # Only one hold timer should have been scheduled by the first press.
    assert len(clock.timers) == 1
    clock.timers[0].fire()
    assert recorder.events == ["hold_start"]

    engine.key_up(0.5)
    assert recorder.events == ["hold_start", "hold_end"]


def test_spurious_key_up_ignored() -> None:
    clock = FakeClock()
    recorder = Recorder()
    engine = make_engine(clock, recorder)

    engine.key_up(0.0)
    assert recorder.events == []
    assert clock.timers == []


def test_key_up_before_hold_fires_no_hold_end() -> None:
    clock = FakeClock()
    recorder = Recorder()
    engine = make_engine(clock, recorder)

    engine.key_down(0.0)
    engine.key_up(0.1)
    # A tap must not produce hold_end even though a hold timer existed.
    assert "hold_end" not in recorder.events


def test_stale_hold_timer_after_release_is_ignored() -> None:
    clock = FakeClock()
    recorder = Recorder()
    engine = make_engine(clock, recorder)

    engine.key_down(0.0)
    hold_timer = clock.timers[0]
    engine.key_up(0.1)
    # Fire the cancelled hold timer; it must not start a hold.
    hold_timer.fire()
    assert "hold_start" not in recorder.events


def test_sequential_gestures() -> None:
    clock = FakeClock()
    recorder = Recorder()
    engine = make_engine(clock, recorder)

    # A hold.
    engine.key_down(0.0)
    clock.timers[0].fire()
    engine.key_up(0.5)

    # Then a single tap.
    engine.key_down(1.0)
    engine.key_up(1.1)
    clock.live_timers[-1].fire()

    assert recorder.events == ["hold_start", "hold_end", "single_tap"]


def test_callback_exception_does_not_break_engine() -> None:
    clock = FakeClock()
    recorder = Recorder()

    def boom() -> None:
        recorder.events.append("single_tap_raised")
        raise RuntimeError("callback failure")

    engine = TapDecisionEngine(
        on_hold_start=recorder.make("hold_start"),
        on_hold_end=recorder.make("hold_end"),
        on_single_tap=boom,
        on_double_tap=recorder.make("double_tap"),
        timer_factory=clock,
    )

    engine.key_down(0.0)
    engine.key_up(0.1)
    # The raising single-tap callback must be swallowed.
    clock.live_timers[-1].fire()
    assert recorder.events == ["single_tap_raised"]

    # The engine must still work after the exception.
    engine.key_down(1.0)
    clock.timers[-1].fire()
    engine.key_up(1.5)
    assert recorder.events == ["single_tap_raised", "hold_start", "hold_end"]


def test_double_tap_then_new_single_tap() -> None:
    clock = FakeClock()
    recorder = Recorder()
    engine = make_engine(clock, recorder)

    # Double tap.
    engine.key_down(0.0)
    engine.key_up(0.1)
    engine.key_down(0.2)
    engine.key_up(0.25)
    assert recorder.events == ["double_tap"]

    # A following lone tap is a fresh single tap.
    engine.key_down(1.0)
    engine.key_up(1.1)
    clock.live_timers[-1].fire()
    assert recorder.events == ["double_tap", "single_tap"]
