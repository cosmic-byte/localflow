"""First-launch onboarding tour for the localflow widget.

A small floating panel that walks new users through the dictation basics in a
few pages with Back, Next, and Skip controls. It opens automatically on the
first launch, gated by the ``onboarding_done`` config flag, and can be
replayed at any time from the widget's right-click menu via Help. Page content
lives in plain data (``build_pages``) so it stays testable without AppKit.
Like ``widget``, all AppKit imports live at module top; ``app.py`` imports
this layer only inside the GUI entry path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import objc
from AppKit import (
    NSApplication,
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSFont,
    NSMakeRect,
    NSObject,
    NSPanel,
    NSTextField,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskTitled,
)

from localflow.config import AppConfig

log = logging.getLogger(__name__)

WINDOW_WIDTH = 470
WINDOW_HEIGHT = 300
MARGIN = 24
TITLE_HEIGHT = 24
TITLE_FONT_SIZE = 15.0
BODY_FONT_SIZE = 13.0
PROGRESS_FONT_SIZE = 11.0
BUTTON_WIDTH = 88
BUTTON_HEIGHT = 32
BUTTON_Y = 14
PANEL_LEVEL_FLOATING = 3


@dataclass(frozen=True)
class Page:
    """One onboarding page.

    Attributes:
        title: Short heading shown in bold at the top of the panel.
        body: A few sentences of explanation shown below the title.
    """

    title: str
    body: str


def build_pages(config: AppConfig) -> tuple[Page, ...]:
    """Return the onboarding pages with the active hotkey woven into the text.

    Args:
        config: Application configuration providing the hotkey spec.

    Returns:
        The ordered pages of the tour.
    """
    hotkey = config.hotkey
    return (
        Page(
            "Dictate anywhere",
            f"Hold {hotkey}, speak, and release: your words are transcribed, "
            "cleaned up, and typed into whichever app has focus. Everything "
            "runs on this Mac — no audio or text ever leaves the machine.",
        ),
        Page(
            "Two ways to record",
            f"Push-to-talk: hold {hotkey} while speaking and release to "
            f"finish. Hands-free: double-tap {hotkey} to start, then tap it "
            "once to stop. Clicking the pill also starts and stops a "
            "hands-free recording.",
        ),
        Page(
            "Reading the pill",
            "The dot on the left shows what is happening: green is idle, red "
            "is recording (a timer appears on the right), and amber means "
            "transcribing. A green flash means the text was inserted; an "
            "orange flash means something went wrong — see Open Log in the "
            "right-click menu.",
        ),
        Page(
            "Make it yours",
            "Right-click the pill to switch the microphone, the ASR and "
            "cleanup models, or toggle the cleanup pass and auto-paste. Open "
            "Dictionary holds words that bias recognition (names, jargon) "
            "and exact replacements applied to every transcript.",
        ),
        Page(
            "Moving and quitting",
            "Drag the pill anywhere; its position is remembered. To quit, "
            "hover over the pill and click the red close button, choose Quit "
            "in the right-click menu, or drag the pill into the bottom-right "
            "corner of the screen.",
        ),
        Page(
            "Permissions",
            "macOS asks for three permissions on first use: Microphone "
            "(recording), Accessibility (typing into other apps), and Input "
            "Monitoring (the global hotkey). Grant them in System Settings → "
            "Privacy & Security. Replay this tour any time from the "
            "right-click menu under Help.",
        ),
    )


class OnboardingController(NSObject):
    """Owns the tour panel, page navigation, and the onboarding-done flag.

    The panel is created lazily on first ``show`` and kept alive across
    closes so Help can reopen it. Dismissing the tour by any path (Done,
    Skip, or the window close button) marks onboarding as done in the config.
    """

    config = objc.ivar()
    pages = objc.ivar()
    index = objc.ivar()
    panel = objc.ivar()
    title_label = objc.ivar()
    body_label = objc.ivar()
    progress_label = objc.ivar()
    back_button = objc.ivar()
    next_button = objc.ivar()
    skip_button = objc.ivar()

    def initWithConfig_(self, config):
        """Initialize the controller.

        Args:
            config: Application configuration holding the onboarding flag.

        Returns:
            The initialized controller or None.
        """
        self = objc.super(OnboardingController, self).init()
        if self is None:
            return None
        self.config = config
        self.pages = build_pages(config)
        self.index = 0
        self.panel = None
        return self

    def show(self):
        """Open the tour at the first page, creating the panel on first use."""
        self.index = 0
        if self.panel is None:
            self._build_panel()
        self._render()
        self.panel.center()
        self.panel.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def next_(self, sender):
        if self.index >= len(self.pages) - 1:
            self.panel.close()
            return
        self.index += 1
        self._render()

    def back_(self, sender):
        if self.index > 0:
            self.index -= 1
            self._render()

    def skip_(self, sender):
        self.panel.close()

    def windowWillClose_(self, notification):
        self._mark_done()

    def _build_panel(self):
        """Create the panel and its static subviews once."""
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable,
            NSBackingStoreBuffered,
            False,
        )
        panel.setTitle_("Welcome to localflow")
        panel.setLevel_(PANEL_LEVEL_FLOATING)
        panel.setReleasedWhenClosed_(False)
        panel.setDelegate_(self)
        content = panel.contentView()

        title = NSTextField.labelWithString_("")
        title.setFont_(NSFont.boldSystemFontOfSize_(TITLE_FONT_SIZE))
        title.setFrame_(
            NSMakeRect(
                MARGIN,
                WINDOW_HEIGHT - MARGIN - TITLE_HEIGHT,
                WINDOW_WIDTH - 2 * MARGIN,
                TITLE_HEIGHT,
            )
        )
        content.addSubview_(title)

        body_top = WINDOW_HEIGHT - MARGIN - TITLE_HEIGHT - 12
        body_bottom = BUTTON_Y + BUTTON_HEIGHT + 12
        body = NSTextField.wrappingLabelWithString_("")
        body.setFont_(NSFont.systemFontOfSize_(BODY_FONT_SIZE))
        body.setSelectable_(False)
        body.setFrame_(
            NSMakeRect(
                MARGIN,
                body_bottom,
                WINDOW_WIDTH - 2 * MARGIN,
                body_top - body_bottom,
            )
        )
        content.addSubview_(body)

        progress = NSTextField.labelWithString_("")
        progress.setFont_(NSFont.systemFontOfSize_(PROGRESS_FONT_SIZE))
        progress.setTextColor_(NSColor.secondaryLabelColor())
        progress.setFrame_(NSMakeRect(MARGIN, BUTTON_Y + 8, 60, 16))
        content.addSubview_(progress)

        skip = NSButton.buttonWithTitle_target_action_("Skip", self, "skip:")
        skip.setFrame_(NSMakeRect(MARGIN + 60, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT))
        content.addSubview_(skip)

        next_button = NSButton.buttonWithTitle_target_action_("Next", self, "next:")
        next_button.setFrame_(
            NSMakeRect(
                WINDOW_WIDTH - MARGIN - BUTTON_WIDTH,
                BUTTON_Y,
                BUTTON_WIDTH,
                BUTTON_HEIGHT,
            )
        )
        next_button.setKeyEquivalent_("\r")
        content.addSubview_(next_button)

        back = NSButton.buttonWithTitle_target_action_("Back", self, "back:")
        back.setFrame_(
            NSMakeRect(
                WINDOW_WIDTH - MARGIN - 2 * BUTTON_WIDTH - 8,
                BUTTON_Y,
                BUTTON_WIDTH,
                BUTTON_HEIGHT,
            )
        )
        content.addSubview_(back)

        self.panel = panel
        self.title_label = title
        self.body_label = body
        self.progress_label = progress
        self.back_button = back
        self.next_button = next_button
        self.skip_button = skip

    def _render(self):
        """Sync the labels and buttons with the current page."""
        page = self.pages[self.index]
        last = self.index == len(self.pages) - 1
        self.title_label.setStringValue_(page.title)
        self.body_label.setStringValue_(page.body)
        self.progress_label.setStringValue_(f"{self.index + 1} of {len(self.pages)}")
        self.back_button.setHidden_(self.index == 0)
        self.next_button.setTitle_("Done" if last else "Next")
        self.skip_button.setHidden_(last)

    def _mark_done(self):
        """Persist the onboarding flag the first time the tour is dismissed."""
        if not self.config.onboarding_done:
            self.config.onboarding_done = True
            self.config.save()
            log.info("Onboarding completed")
