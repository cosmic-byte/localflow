"""Floating capsule HUD widget for localflow.

A small frosted-glass capsule inspired by modern dictation HUDs: a status dot on
the left, an audio-reactive bar visualizer in the middle, and a compact status
label on the right. The view only draws, handles gestures, and calls injected
callbacks; it owns no pipeline logic. All AppKit imports live at module top so
that ``app.py`` can stay importable on a headless machine by importing this
module only inside the GUI entry path.
"""

from __future__ import annotations

import logging
import math
import time

import objc
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSEvent,
    NSFont,
    NSFontWeightMedium,
    NSMakeRect,
    NSMakeSize,
    NSMenu,
    NSMenuItem,
    NSNumber,
    NSObject,
    NSOnState,
    NSPanel,
    NSPoint,
    NSScreen,
    NSString,
    NSTimer,
    NSTrackingActiveAlways,
    NSTrackingArea,
    NSTrackingInVisibleRect,
    NSTrackingMouseEnteredAndExited,
    NSView,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectMaterialHUDWindow,
    NSVisualEffectStateActive,
    NSVisualEffectView,
    NSWindowCloseButton,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowMiniaturizeButton,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskFullSizeContentView,
    NSWindowStyleMaskNonactivatingPanel,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
    NSWindowTitleHidden,
    NSWindowZoomButton,
)

from localflow.capture import cached_audio_devices
from localflow.config import AppConfig
from localflow.onboarding import OnboardingController

log = logging.getLogger(__name__)

PANEL_WIDTH = 190
PANEL_HEIGHT = 44
TITLE_BAR_BASE = 22
TRAFFIC_LIGHT_MARGIN = 8
TITLE_BAR_INSET = TITLE_BAR_BASE + TRAFFIC_LIGHT_MARGIN
WINDOW_HEIGHT = PANEL_HEIGHT + TITLE_BAR_INSET
CORNER_RADIUS = PANEL_HEIGHT / 2

HOLD_THRESHOLD = 0.3
DRAG_THRESHOLD = 5
CLOSE_ZONE = 80
WAVE_SPEED_IDLE = 0.04
WAVE_SPEED_RECORD = 0.10
WAVE_SPEED_TRANSCRIBE = 0.14

DOT_CX = 19.0
DOT_RADIUS = 4.25

BAR_COUNT = 28
BAR_WIDTH = 2.4
BARS_LEFT = 32.0
BARS_RIGHT = 136.0
BAR_MIN_HEIGHT = 3.0
BAR_MAX_HEIGHT = 24.0
IDLE_BAR_HEIGHT = 5.5
SWEEP_HALF_WIDTH = 5.0
SWEEP_SPEED = 4.0

STATUS_CX = 161.0
TIMER_FONT_SIZE = 10.0
IDLE_FONT_SIZE = 9.0

FALLBACK_BACKGROUND = (0.11, 0.11, 0.11, 0.95)
BORDER_COLOR = (1.0, 1.0, 1.0, 0.08)
CLOSE_OVERLAY_COLOR = (0.85, 0.15, 0.15, 0.80)

DOT_IDLE = (0.25, 0.75, 0.40, 0.55)
DOT_RECORDING = (1.0, 0.27, 0.23, 1.0)
DOT_TRANSCRIBING = (1.0, 0.72, 0.10, 0.95)
DOT_SUCCESS = (0.20, 0.85, 0.45, 1.0)
DOT_ERROR = (1.0, 0.50, 0.0, 1.0)

BARS_IDLE = (0.62, 0.62, 0.65)
BARS_RECORDING = (0.98, 0.90, 0.90)
BARS_TRANSCRIBING = (0.85, 0.85, 0.90)
BARS_SUCCESS = (0.20, 0.85, 0.45)
BARS_ERROR = (1.0, 0.50, 0.0)

TEXT_RECORDING = (1.0, 0.38, 0.32, 0.95)
TEXT_IDLE = (0.60, 0.60, 0.62, 0.55)

STATE_IDLE = 0
STATE_RECORDING = 1
STATE_TRANSCRIBING = 2
STATE_SUCCESS = 3
STATE_ERROR = 4

ASR_MODELS = ("base", "small", "large-v3-turbo")
CLEANUP_MODELS = ("qwen2.5:7b", "llama3.2:3b")

# NonactivatingPanel keeps the widget from stealing focus from the app being
# dictated into; Miniaturizable is omitted because an agent app (LSUIElement)
# has no Dock icon to restore a minimized window from.
PANEL_STYLE_MASK = (
    NSWindowStyleMaskTitled
    | NSWindowStyleMaskClosable
    | NSWindowStyleMaskResizable
    | NSWindowStyleMaskFullSizeContentView
    | NSWindowStyleMaskNonactivatingPanel
)


def get_screen_frame():
    """Return the main screen frame."""
    return NSScreen.mainScreen().frame()


def is_near_close_zone(window_origin: NSPoint, width: float, height: float) -> bool:
    """Return True when the panel center is inside the bottom-right close zone.

    Args:
        window_origin: Bottom-left origin of the panel in screen coordinates.
        width: Panel width in points.
        height: Panel height in points.

    Returns:
        True when releasing here should close the widget.
    """
    screen = get_screen_frame()
    sw = screen.size.width
    sb = screen.origin.y
    cx = window_origin.x + width / 2
    cy = window_origin.y + height / 2
    return (sw - cx) < CLOSE_ZONE and (cy - sb) < CLOSE_ZONE


def clamp_to_screen(origin: NSPoint, width: float, height: float) -> NSPoint:
    """Clamp a panel origin so the whole window stays on the visible screen.

    Args:
        origin: Desired bottom-left origin of the panel.
        width: Window width in points.
        height: Window height in points.

    Returns:
        The origin clamped to the visible frame.
    """
    screen = NSScreen.mainScreen().visibleFrame()
    x = max(
        screen.origin.x,
        min(origin.x, screen.origin.x + screen.size.width - width),
    )
    y = max(
        screen.origin.y,
        min(origin.y, screen.origin.y + screen.size.height - height),
    )
    return NSPoint(x, y)


def _rgba(r: float, g: float, b: float, a: float):
    """Return a calibrated NSColor for the given channel values."""
    return NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a)


def _schedule_timer(interval, target, selector, repeats):
    """Schedule an NSTimer on the current run loop.

    Args:
        interval: Fire interval in seconds.
        target: Object receiving the selector.
        selector: Selector string invoked on each fire.
        repeats: Whether the timer repeats.

    Returns:
        The scheduled NSTimer.
    """
    return NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        interval, target, selector, None, repeats
    )


def _capsule_path(w: float, h: float) -> NSBezierPath:
    """Return the capsule bezier path filling a w x h rectangle."""
    return NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
        NSMakeRect(0, 0, w, h), CORNER_RADIUS, CORNER_RADIUS
    )


def _install_pill_backdrop(parent: NSView) -> bool:
    """Install a frosted-glass capsule backdrop for the pill body only.

    The title-bar region above the pill stays transparent so the traffic-light
    buttons appear to float over the desktop.

    Args:
        parent: Container view that holds the pill backdrop and widget.

    Returns:
        True when the backdrop was installed, False when the caller should
        fall back to a solid painted background.
    """
    try:
        effect = NSVisualEffectView.alloc().initWithFrame_(
            NSMakeRect(0, 0, PANEL_WIDTH, PANEL_HEIGHT)
        )
        effect.setMaterial_(NSVisualEffectMaterialHUDWindow)
        effect.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        effect.setState_(NSVisualEffectStateActive)
        effect.setWantsLayer_(True)
        layer = effect.layer()
        if layer is None:
            return False
        layer.setCornerRadius_(CORNER_RADIUS)
        layer.setMasksToBounds_(True)
        parent.addSubview_(effect)
    except (AttributeError, ValueError, objc.error):
        log.exception("Visual effect backdrop unavailable; using solid capsule")
        return False
    return True


def _has_backdrop(panel: NSPanel) -> bool:
    """Return True when the panel container includes a visual effect backdrop."""
    container = panel.contentView()
    if container is None:
        return False
    for subview in container.subviews():
        if subview.isKindOfClass_(NSVisualEffectView):
            return True
    return False


def create_panel(config: AppConfig) -> NSPanel:
    """Create a floating panel with transparent title bar and pill body.

    Args:
        config: Application configuration providing the persisted window origin.

    Returns:
        A configured NSPanel with traffic-light buttons floating above the
        frosted capsule content view when the visual effect material is available.
    """
    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(config.window_x, config.window_y, PANEL_WIDTH, WINDOW_HEIGHT),
        PANEL_STYLE_MASK,
        NSBackingStoreBuffered,
        False,
    )
    panel.setTitle_("localflow")
    panel.setTitleVisibility_(NSWindowTitleHidden)
    panel.setTitlebarAppearsTransparent_(True)
    try:
        panel.setTitlebarSeparatorStyle_(1)
    except (AttributeError, ValueError, objc.error):
        pass
    panel.setMinSize_(NSMakeSize(PANEL_WIDTH, WINDOW_HEIGHT))
    panel.setMaxSize_(NSMakeSize(PANEL_WIDTH, WINDOW_HEIGHT))
    panel.setFloatingPanel_(True)
    panel.setBecomesKeyOnlyIfNeeded_(True)
    panel.setHidesOnDeactivate_(False)
    panel.setHasShadow_(True)
    panel.setOpaque_(False)
    panel.setBackgroundColor_(NSColor.clearColor())
    panel.setLevel_(3)
    panel.setAcceptsMouseMovedEvents_(True)
    panel.setWorksWhenModal_(True)
    panel.setMovableByWindowBackground_(True)
    panel.setCollectionBehavior_(
        NSWindowCollectionBehaviorCanJoinAllSpaces
        | NSWindowCollectionBehaviorIgnoresCycle
    )
    container = NSView.alloc().initWithFrame_(
        NSMakeRect(0, 0, PANEL_WIDTH, WINDOW_HEIGHT)
    )
    panel.setContentView_(container)
    _install_pill_backdrop(container)
    return panel


class PanelDelegate(NSObject):
    """Quit the application when the user closes the widget window."""

    def windowShouldClose_(self, sender):
        NSApplication.sharedApplication().terminate_(None)
        return True


_TRAFFIC_LIGHT_KINDS = (
    NSWindowCloseButton,
    NSWindowMiniaturizeButton,
    NSWindowZoomButton,
)


def _set_traffic_lights_visible(panel: NSPanel, visible: bool) -> None:
    """Show or hide the standard window close, minimize, and zoom buttons."""
    for kind in _TRAFFIC_LIGHT_KINDS:
        button = panel.standardWindowButton_(kind)
        if button is not None:
            button.setHidden_(not visible)


class TrafficLightHoverController(NSObject):
    """Reveal window traffic-light buttons while the pointer is over the panel."""

    panel = objc.ivar()
    tracking_area = objc.ivar()

    def installWithPanel_(self, panel):
        """Hide the traffic-light buttons and track pointer enter/exit on the panel.

        Args:
            panel: The widget window whose standard buttons should auto-hide.
        """
        self.panel = panel
        _set_traffic_lights_visible(panel, False)
        view = panel.contentView()
        if view is None:
            return
        options = (
            NSTrackingMouseEnteredAndExited
            | NSTrackingActiveAlways
            | NSTrackingInVisibleRect
        )
        area = NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
            view.bounds(), options, self, None
        )
        view.addTrackingArea_(area)
        self.tracking_area = area

    def mouseEntered_(self, event):
        _set_traffic_lights_visible(self.panel, True)

    def mouseExited_(self, event):
        if not self._pointer_in_panel():
            _set_traffic_lights_visible(self.panel, False)

    def _pointer_in_panel(self) -> bool:
        """Return True when the cursor is still inside the panel frame."""
        panel = self.panel
        if panel is None:
            return False
        loc = NSEvent.mouseLocation()
        frame = panel.frame()
        return (
            frame.origin.x <= loc.x <= frame.origin.x + frame.size.width
            and frame.origin.y <= loc.y <= frame.origin.y + frame.size.height
        )


class FlowWidget(NSView):
    """Capsule-shaped dictation HUD.

    Draws the five UI states as a status dot, an audio-reactive bar visualizer,
    and a compact status label; handles the click/hold/drag gesture model;
    persists its position through the injected AppConfig; and forwards user
    intent to the injected callbacks. It never runs pipeline work itself.
    """

    state = objc.ivar()
    audio_level = objc.ivar()
    levels = objc.ivar()
    wave_phase = objc.ivar()
    recording_time = objc.ivar()
    is_dragging = objc.ivar()
    mouse_down_screen = objc.ivar()
    panel_down_origin = objc.ivar()
    mouse_down_time = objc.ivar()
    hold_timer = objc.ivar()
    pulse_timer = objc.ivar()
    recording_timer = objc.ivar()
    flash_timer = objc.ivar()
    panel = objc.ivar()
    config = objc.ivar()
    near_close_zone = objc.ivar()
    draws_background = objc.ivar()
    on_click = objc.ivar()
    on_hold_start = objc.ivar()
    on_hold_end = objc.ivar()
    on_right_click = objc.ivar()
    mic_name = objc.ivar()
    model_name = objc.ivar()

    def initWithFrame_panel_config_callbacks_(self, frame, panel, config, callbacks):
        """Initialize the widget.

        Args:
            frame: View frame rectangle.
            panel: The hosting NSPanel used for drag/position operations.
            config: Application configuration used to persist the position.
            callbacks: Tuple of (on_click, on_hold_start, on_hold_end) callables.

        Returns:
            The initialized widget or None.
        """
        self = objc.super(FlowWidget, self).initWithFrame_(frame)
        if self is None:
            return None
        self.panel = panel
        self.config = config
        self.state = STATE_IDLE
        self.audio_level = 0.0
        self.levels = [0.0] * BAR_COUNT
        self.wave_phase = 0.0
        self.recording_time = 0
        self.is_dragging = False
        self.mouse_down_screen = NSPoint(0, 0)
        self.panel_down_origin = NSPoint(0, 0)
        self.mouse_down_time = 0.0
        self.hold_timer = None
        self.pulse_timer = None
        self.recording_timer = None
        self.flash_timer = None
        self.near_close_zone = False
        self.draws_background = True
        self.mic_name = "Default"
        self.model_name = "large-v3-turbo"
        self.on_click = callbacks[0]
        self.on_hold_start = callbacks[1]
        self.on_hold_end = callbacks[2]
        self.on_right_click = None
        self._start_pulse_timer()
        return self

    def isFlipped(self):
        return False

    def acceptsFirstMouse_(self, event):
        return True

    def mouseDownCanMoveWindow(self):
        return False

    def drawRect_(self, rect):
        bounds = self.bounds()
        w = bounds.size.width
        h = bounds.size.height

        if self.draws_background:
            _rgba(*FALLBACK_BACKGROUND).setFill()
            _capsule_path(w, h).fill()

        if self.near_close_zone:
            _rgba(*CLOSE_OVERLAY_COLOR).setFill()
            _capsule_path(w, h).fill()
            self._draw_x_icon(NSPoint(w / 2, h / 2))
            return

        border = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(0.5, 0.5, w - 1.0, h - 1.0),
            CORNER_RADIUS - 0.5,
            CORNER_RADIUS - 0.5,
        )
        _rgba(*BORDER_COLOR).setStroke()
        border.setLineWidth_(1.0)
        border.stroke()

        self._draw_status_dot(NSPoint(DOT_CX, h / 2))
        self._draw_bars(h / 2)
        self._draw_status_text(STATUS_CX, h / 2)

    def _draw_status_dot(self, center):
        if self.state == STATE_RECORDING:
            pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(self.wave_phase * 2.0))
            r = DOT_RADIUS + 1.2 * self.audio_level
            color = _rgba(DOT_RECORDING[0], DOT_RECORDING[1], DOT_RECORDING[2], pulse)
        elif self.state == STATE_TRANSCRIBING:
            r = DOT_RADIUS
            color = _rgba(*DOT_TRANSCRIBING)
        elif self.state == STATE_SUCCESS:
            r = DOT_RADIUS
            color = _rgba(*DOT_SUCCESS)
        elif self.state == STATE_ERROR:
            r = DOT_RADIUS
            color = _rgba(*DOT_ERROR)
        else:
            r = DOT_RADIUS
            color = _rgba(*DOT_IDLE)
        dot = NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(center.x - r, center.y - r, r * 2, r * 2)
        )
        color.setFill()
        dot.fill()

    def _bar_height_and_alpha(self, index: int) -> tuple[float, float]:
        """Return the height and alpha of one visualizer bar for this frame.

        Args:
            index: Bar index from 0 (leftmost, oldest sample) to BAR_COUNT - 1
                (rightmost, newest sample).

        Returns:
            Tuple of (height in points, alpha in [0, 1]).
        """
        if self.state == STATE_RECORDING:
            level = self.levels[index] if index < len(self.levels) else 0.0
            height = BAR_MIN_HEIGHT + level * (BAR_MAX_HEIGHT - BAR_MIN_HEIGHT)
            return height, 0.35 + 0.60 * level
        if self.state == STATE_TRANSCRIBING:
            span = BAR_COUNT + 2 * SWEEP_HALF_WIDTH
            pos = (self.wave_phase * SWEEP_SPEED) % span - SWEEP_HALF_WIDTH
            boost = max(0.0, 1.0 - abs(index - pos) / SWEEP_HALF_WIDTH)
            return BAR_MIN_HEIGHT + 12.0 * boost, 0.30 + 0.65 * boost
        if self.state in (STATE_SUCCESS, STATE_ERROR):
            height = BAR_MIN_HEIGHT + 7.0 * (0.5 + 0.5 * math.sin(index * 0.7))
            return height, 0.85
        breath = 0.5 + 0.5 * math.sin(self.wave_phase * 0.6 + index * 0.45)
        return IDLE_BAR_HEIGHT * (0.6 + 0.4 * breath), 0.45

    def _draw_bars(self, center_y):
        if self.state == STATE_RECORDING:
            rgb = BARS_RECORDING
        elif self.state == STATE_TRANSCRIBING:
            rgb = BARS_TRANSCRIBING
        elif self.state == STATE_SUCCESS:
            rgb = BARS_SUCCESS
        elif self.state == STATE_ERROR:
            rgb = BARS_ERROR
        else:
            rgb = BARS_IDLE

        slot = (BARS_RIGHT - BARS_LEFT) / BAR_COUNT
        for i in range(BAR_COUNT):
            height, alpha = self._bar_height_and_alpha(i)
            x = BARS_LEFT + i * slot + (slot - BAR_WIDTH) / 2
            bar = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, center_y - height / 2, BAR_WIDTH, height),
                BAR_WIDTH / 2,
                BAR_WIDTH / 2,
            )
            _rgba(rgb[0], rgb[1], rgb[2], alpha).setFill()
            bar.fill()

    def _model_tag(self) -> str:
        """Return a short display tag for the current ASR model name."""
        name = self.model_name or ""
        if len(name) <= 8:
            return name
        return name.split("-")[-1]

    def _draw_status_text(self, cx, cy):
        if self.state == STATE_RECORDING:
            m = self.recording_time // 60
            s = self.recording_time % 60
            text = f"{m}:{s:02d}"
            font = NSFont.monospacedDigitSystemFontOfSize_weight_(
                TIMER_FONT_SIZE, NSFontWeightMedium
            )
            color = _rgba(*TEXT_RECORDING)
        elif self.state == STATE_IDLE:
            text = self._model_tag()
            if not text:
                return
            font = NSFont.systemFontOfSize_(IDLE_FONT_SIZE)
            color = _rgba(*TEXT_IDLE)
        else:
            return

        ns = NSString.stringWithString_(text)
        attrs = {"NSFont": font, "NSColor": color}
        sz = ns.sizeWithAttributes_(attrs)
        ns.drawAtPoint_withAttributes_(
            NSPoint(cx - sz.width / 2, cy - sz.height / 2), attrs
        )

    def _draw_x_icon(self, center):
        path = NSBezierPath.bezierPath()
        d = 7
        path.moveToPoint_(NSPoint(center.x - d, center.y - d))
        path.lineToPoint_(NSPoint(center.x + d, center.y + d))
        path.moveToPoint_(NSPoint(center.x + d, center.y - d))
        path.lineToPoint_(NSPoint(center.x - d, center.y + d))
        _rgba(1.0, 1.0, 1.0, 0.9).setStroke()
        path.setLineWidth_(2.5)
        path.setLineCapStyle_(1)
        path.stroke()

    def mouseDown_(self, event):
        self.mouse_down_screen = NSEvent.mouseLocation()
        self.panel_down_origin = self.panel.frame().origin
        self.mouse_down_time = time.time()
        self.is_dragging = False
        self.hold_timer = _schedule_timer(HOLD_THRESHOLD, self, "onHoldTimer:", False)

    def mouseDragged_(self, event):
        current = NSEvent.mouseLocation()
        dx = current.x - self.mouse_down_screen.x
        dy = current.y - self.mouse_down_screen.y

        if abs(dx) > DRAG_THRESHOLD or abs(dy) > DRAG_THRESHOLD:
            self.is_dragging = True
            if self.hold_timer:
                self.hold_timer.invalidate()
                self.hold_timer = None

            new_origin = NSPoint(
                self.panel_down_origin.x + dx,
                self.panel_down_origin.y + dy,
            )
            self.panel.setFrameOrigin_(new_origin)

            frame = self.panel.frame()
            size = frame.size
            was_near = self.near_close_zone
            self.near_close_zone = is_near_close_zone(
                new_origin, size.width, size.height
            )
            if was_near != self.near_close_zone:
                self.setNeedsDisplay_(True)

    def mouseUp_(self, event):
        if self.hold_timer:
            self.hold_timer.invalidate()
            self.hold_timer = None

        if self.is_dragging:
            self.is_dragging = False
            self.near_close_zone = False
            self.setNeedsDisplay_(True)

            frame = self.panel.frame()
            size = frame.size
            if is_near_close_zone(frame.origin, size.width, size.height):
                NSApplication.sharedApplication().terminate_(None)
                return

            origin = clamp_to_screen(frame.origin, size.width, size.height)
            self.panel.setFrameOrigin_(origin)
            self._save_position(origin)
            return

        elapsed = time.time() - self.mouse_down_time
        if elapsed < HOLD_THRESHOLD:
            if self.on_click:
                self.on_click()
        elif self.on_hold_end:
            self.on_hold_end()

    def rightMouseDown_(self, event):
        if self.on_right_click:
            self.on_right_click(event)

    def _save_position(self, origin):
        self.config.window_x = int(origin.x)
        self.config.window_y = int(origin.y)
        self.config.save()

    def updateState_(self, new_state):
        if self.flash_timer:
            self.flash_timer.invalidate()
            self.flash_timer = None
        if self.recording_timer:
            self.recording_timer.invalidate()
            self.recording_timer = None

        self.state = int(new_state)

        if self.state == STATE_RECORDING:
            self.recording_time = 0
            self.levels = [0.0] * BAR_COUNT
            self.recording_timer = _schedule_timer(1.0, self, "onRecordingTimer:", True)

        if self.state in (STATE_SUCCESS, STATE_ERROR):
            self.flash_timer = _schedule_timer(0.8, self, "onFlashTimer:", False)

        self.setNeedsDisplay_(True)

    def setAudioLevel_(self, level):
        value = float(level)
        self.audio_level = value
        levels = self.levels
        levels.append(value)
        if len(levels) > BAR_COUNT:
            del levels[: len(levels) - BAR_COUNT]

    def setMicName_(self, name):
        self.mic_name = str(name)

    def setModelName_(self, name):
        self.model_name = str(name)
        self.setNeedsDisplay_(True)

    def showRecording(self):
        """Switch to the recording state on the main thread. Thread-safe."""
        self._dispatch_state(STATE_RECORDING)

    def showTranscribing(self):
        """Switch to the transcribing state on the main thread. Thread-safe."""
        self._dispatch_state(STATE_TRANSCRIBING)

    def showSuccess(self):
        """Flash the success state on the main thread. Thread-safe."""
        self._dispatch_state(STATE_SUCCESS)

    def showError(self):
        """Flash the error state on the main thread. Thread-safe."""
        self._dispatch_state(STATE_ERROR)

    def applyLevel_(self, level):
        """Push a new audio level to the view on the main thread. Thread-safe."""
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "setAudioLevel:", NSNumber.numberWithFloat_(float(level)), False
        )

    def applyModelName_(self, name):
        """Set the model label on the main thread. Thread-safe."""
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "setModelName:", NSString.stringWithString_(str(name)), False
        )

    def _dispatch_state(self, state):
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "updateState:", NSNumber.numberWithInt_(state), False
        )

    def _start_pulse_timer(self):
        if self.pulse_timer:
            self.pulse_timer.invalidate()
        self.pulse_timer = _schedule_timer(0.04, self, "onPulseTimer:", True)

    def onHoldTimer_(self, timer):
        self.hold_timer = None
        if not self.is_dragging and self.on_hold_start:
            self.on_hold_start()

    def onPulseTimer_(self, timer):
        if self.state == STATE_RECORDING:
            self.wave_phase += WAVE_SPEED_RECORD
        elif self.state == STATE_TRANSCRIBING:
            self.wave_phase += WAVE_SPEED_TRANSCRIBE
        else:
            self.wave_phase += WAVE_SPEED_IDLE
        self.setNeedsDisplay_(True)

    def onRecordingTimer_(self, timer):
        self.recording_time += 1
        self.setNeedsDisplay_(True)

    def onFlashTimer_(self, timer):
        self.flash_timer = None
        self.state = STATE_IDLE
        self.setNeedsDisplay_(True)


def _build_microphone_submenu(config: AppConfig, handler) -> NSMenu:
    """Build the microphone picker submenu with the active device highlighted."""
    mic_sub = NSMenu.alloc().initWithTitle_("Microphone")
    for device in cached_audio_devices():
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            device.name, "selectMicrophone:", ""
        )
        item.setTarget_(handler)
        item.setRepresentedObject_(NSNumber.numberWithInt_(device.id))
        if device.id == config.microphone_id:
            item.setState_(NSOnState)
        mic_sub.addItem_(item)
    return mic_sub


def _append_settings_items(menu: NSMenu, config: AppConfig, handler) -> None:
    """Append ASR, cleanup, and utility items to a menu.

    Args:
        menu: Destination menu.
        config: Application configuration.
        handler: The MenuHandler receiving setting actions.
    """
    asr_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "ASR Model", None, ""
    )
    asr_sub = NSMenu.alloc().initWithTitle_("ASR Model")
    for name in ASR_MODELS:
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            name, "selectAsrModel:", ""
        )
        item.setTarget_(handler)
        item.setRepresentedObject_(name)
        if name == config.asr_model:
            item.setState_(NSOnState)
        asr_sub.addItem_(item)
    asr_item.setSubmenu_(asr_sub)
    menu.addItem_(asr_item)

    menu.addItem_(NSMenuItem.separatorItem())

    cleanup_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Cleanup", "toggleCleanup:", ""
    )
    cleanup_item.setTarget_(handler)
    if config.cleanup_enabled:
        cleanup_item.setState_(NSOnState)
    menu.addItem_(cleanup_item)

    cleanup_model_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Cleanup Model", None, ""
    )
    cleanup_model_sub = NSMenu.alloc().initWithTitle_("Cleanup Model")
    for name in CLEANUP_MODELS:
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            name, "selectCleanupModel:", ""
        )
        item.setTarget_(handler)
        item.setRepresentedObject_(name)
        if name == config.cleanup_model:
            item.setState_(NSOnState)
        cleanup_model_sub.addItem_(item)
    cleanup_model_item.setSubmenu_(cleanup_model_sub)
    menu.addItem_(cleanup_model_item)

    paste_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Auto-paste", "toggleAutoPaste:", ""
    )
    paste_item.setTarget_(handler)
    if config.auto_paste:
        paste_item.setState_(NSOnState)
    menu.addItem_(paste_item)

    menu.addItem_(NSMenuItem.separatorItem())

    dict_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Open Dictionary", "openDictionary:", ""
    )
    dict_item.setTarget_(handler)
    menu.addItem_(dict_item)

    log_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Open Log", "openLog:", ""
    )
    log_item.setTarget_(handler)
    menu.addItem_(log_item)

    help_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Help", "showHelp:", ""
    )
    help_item.setTarget_(handler)
    menu.addItem_(help_item)


def _append_quit_item(menu: NSMenu, handler, title: str = "Quit") -> None:
    """Append a separator and quit item to a menu.

    Args:
        menu: Destination menu.
        handler: The MenuHandler receiving ``quit:`` actions.
        title: Label for the quit menu item.
    """
    menu.addItem_(NSMenuItem.separatorItem())
    quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        title, "quit:", ""
    )
    quit_item.setTarget_(handler)
    menu.addItem_(quit_item)


def _build_context_menu(controller, handler) -> NSMenu:
    """Build the right-click settings menu.

    Args:
        controller: The FlowController exposing configuration and actions.
        handler: The MenuHandler object that receives the menu actions.

    Returns:
        The populated context menu.
    """
    config = controller.config
    menu = NSMenu.alloc().initWithTitle_("localflow")

    mic_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Microphone", None, ""
    )
    mic_item.setSubmenu_(_build_microphone_submenu(config, handler))
    menu.addItem_(mic_item)

    _append_settings_items(menu, config, handler)
    _append_quit_item(menu, handler)

    return menu


class MenuHandler(NSObject):
    """Target object for the widget's right-click settings menu.

    Builds the menu on demand and forwards each selection to the controller.
    """

    controller = objc.ivar()
    widget = objc.ivar()
    onboarding = objc.ivar()

    def initWithController_widget_(self, controller, widget):
        """Initialize the handler.

        Args:
            controller: The FlowController receiving setting changes.
            widget: The FlowWidget the menu pops up over.

        Returns:
            The initialized handler or None.
        """
        self = objc.super(MenuHandler, self).init()
        if self is None:
            return None
        self.controller = controller
        self.widget = widget
        return self

    def showMenu_(self, event):
        menu = _build_context_menu(self.controller, self)
        NSMenu.popUpContextMenu_withEvent_forView_(menu, event, self.widget)

    def selectMicrophone_(self, sender):
        self.controller.set_microphone(
            sender.representedObject().intValue(), str(sender.title())
        )

    def selectAsrModel_(self, sender):
        self.controller.set_asr_model(str(sender.representedObject()))

    def toggleCleanup_(self, sender):
        self.controller.toggle_cleanup()

    def selectCleanupModel_(self, sender):
        self.controller.set_cleanup_model(str(sender.representedObject()))

    def toggleAutoPaste_(self, sender):
        self.controller.toggle_auto_paste()

    def openDictionary_(self, sender):
        self.controller.open_dictionary()

    def openLog_(self, sender):
        self.controller.open_log()

    def showHelp_(self, sender):
        if self.onboarding is not None:
            self.onboarding.show()

    def quit_(self, sender):
        NSApplication.sharedApplication().terminate_(None)


def run_app(controller) -> None:
    """Build the widget UI, wire it to the controller, and run the main loop.

    Args:
        controller: The FlowController that owns the pipeline and settings.
    """
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    config = controller.config
    panel = create_panel(config)
    panel_delegate = PanelDelegate.alloc().init()
    panel.setDelegate_(panel_delegate)
    widget = FlowWidget.alloc().initWithFrame_panel_config_callbacks_(
        NSMakeRect(0, 0, PANEL_WIDTH, PANEL_HEIGHT),
        panel,
        config,
        (
            controller.handsfree_toggle,
            controller.push_to_talk_start,
            controller.push_to_talk_stop,
        ),
    )
    handler = MenuHandler.alloc().initWithController_widget_(controller, widget)
    widget.on_right_click = handler.showMenu_
    widget.draws_background = not _has_backdrop(panel)
    widget.setMicName_(config.microphone_name)
    widget.setModelName_(config.asr_model)
    container = panel.contentView()
    container.addSubview_(widget)
    panel.makeKeyAndOrderFront_(None)
    traffic_lights = TrafficLightHoverController.alloc().init()
    traffic_lights.installWithPanel_(panel)

    onboarding = OnboardingController.alloc().initWithConfig_(config)
    handler.onboarding = onboarding
    if not config.onboarding_done:
        onboarding.show()

    controller.attach_ui(widget)
    controller.start()
    app.run()
