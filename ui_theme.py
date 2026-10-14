"""Turns :mod:`ui_tokens` into live ttk styles, and switches theme at runtime.

Structure: this module owns *every* color the interface renders. Tabs ask for a
named style (``"Muted.TLabel"``, ``"Accent.TButton"``) or read
``theme().palette``; no widget anywhere should carry a literal color again. The
old code had `foreground="blue"` / `"gray"` / `"orange"` / `"red"` scattered
across five files, which is why a second theme was impossible.

Two Tk realities shape the design:

* **ttk widgets restyle live, classic widgets do not.** Changing a ttk style
  updates every widget using it immediately. ``tk.Listbox`` and ``tk.Text``
  take their colors as instance options at construction, so a theme switch has
  to find them and reconfigure them. :func:`register_classic` records them for
  exactly that, and theme switching is why it exists.
* **ttk has no shadows, no radii, no transitions.** Depth is therefore carried
  by three surface lightness levels plus real 1px borders, and the only motion
  is a periodic callback. Fighting that produces a worse-looking app, not a
  more modern one.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from typing import Callable, List, Optional, Tuple

from ui_tokens import (
    MONO_FONT_CANDIDATES,
    SPACE,
    TYPE_SCALE,
    UI_FONT_CANDIDATES,
    DEFAULT_THEME,
    Palette,
    palette_for,
)


def _first_available(candidates: Tuple[str, ...], installed: set,
                     fallback: str) -> str:
    """First candidate family the system actually has.

    Tk silently accepts an unknown family name and substitutes something, so a
    hardcoded font is not a guarantee — it is an unverifiable guess. Matching
    against ``font.families()`` makes the fallback chain real.
    """
    for family in candidates:
        if family in installed:
            return family
    return fallback


class Fonts:
    """Named font objects, shared by every widget.

    Named ttk fonts (rather than ``("Family", 10)`` tuples) so a future
    "interface scale" setting can resize the whole UI by reconfiguring these
    objects instead of walking the widget tree.
    """

    def __init__(self, root: tk.Misc):
        installed = set(tkfont.families(root))
        ui = _first_available(UI_FONT_CANDIDATES, installed,
                              tkfont.nametofont("TkDefaultFont").cget("family"))
        mono = _first_available(MONO_FONT_CANDIDATES, installed,
                                tkfont.nametofont("TkFixedFont").cget("family"))
        self.ui_family = ui
        self.mono_family = mono

        self.display = tkfont.Font(root, family=ui, size=TYPE_SCALE["display"],
                                   weight="bold")
        self.title = tkfont.Font(root, family=ui, size=TYPE_SCALE["title"],
                                 weight="bold")
        self.heading = tkfont.Font(root, family=ui, size=TYPE_SCALE["heading"],
                                   weight="bold")
        self.body = tkfont.Font(root, family=ui, size=TYPE_SCALE["body"])
        self.body_strong = tkfont.Font(root, family=ui, size=TYPE_SCALE["body"],
                                       weight="bold")
        self.label = tkfont.Font(root, family=ui, size=TYPE_SCALE["label"])
        self.label_strong = tkfont.Font(root, family=ui,
                                        size=TYPE_SCALE["label"], weight="bold")
        self.micro = tkfont.Font(root, family=ui, size=TYPE_SCALE["micro"])
        # Data faces. Digits must not reflow as a counter ticks, which is the
        # whole reason rates and ETAs are mono rather than the UI face.
        self.mono = tkfont.Font(root, family=mono, size=TYPE_SCALE["label"])
        self.mono_body = tkfont.Font(root, family=mono, size=TYPE_SCALE["body"])
        self.mono_micro = tkfont.Font(root, family=mono, size=TYPE_SCALE["micro"])


class Theme:
    """The app's single source of color and type, switchable at runtime."""

    def __init__(self, root: tk.Misc, name: str = DEFAULT_THEME):
        self.root = root
        self.fonts = Fonts(root)
        self.style = ttk.Style(root)
        self.style.theme_use("clam")  # the only built-in theme that honors colors
        self.palette: Palette = palette_for(name)
        # (widget, role) pairs for classic Tk widgets that cannot restyle
        # themselves. Entries are pruned lazily as widgets are destroyed.
        self._classic: List[Tuple[tk.Misc, str]] = []
        self._listeners: List[Callable[[Palette], None]] = []
        self._apply()

    # -------------------------------------------------------------- public
    @property
    def name(self) -> str:
        return self.palette.name

    def switch(self, name: str) -> None:
        """Repaint the whole interface in another palette."""
        self.palette = palette_for(name)
        self._apply()
        self._restyle_classic()
        for listener in list(self._listeners):
            listener(self.palette)

    def toggle(self) -> str:
        """Flip dark/light and return the new theme name."""
        self.switch("light" if self.palette.dark else "dark")
        return self.name

    def on_change(self, callback: Callable[[Palette], None]) -> Callable[[], None]:
        """Subscribe to theme changes; returns an unsubscribe callable.

        For the places a style name cannot reach: canvas-drawn content, log tag
        colors, and anything that computed a color once at construction.
        """
        self._listeners.append(callback)

        def _unsubscribe() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return _unsubscribe

    def register_classic(self, widget: tk.Misc, role: str) -> tk.Misc:
        """Track a classic Tk widget so theme switches reach it.

        ``role`` is one of ``listbox``, ``text``, ``toplevel``. Returns the
        widget so calls can wrap construction inline.
        """
        self._classic.append((widget, role))
        self._style_classic(widget, role)
        return widget

    # ------------------------------------------------------------- classic
    def _restyle_classic(self) -> None:
        alive: List[Tuple[tk.Misc, str]] = []
        for widget, role in self._classic:
            try:
                if not widget.winfo_exists():
                    continue
                self._style_classic(widget, role)
                alive.append((widget, role))
            except tk.TclError:
                # Destroyed between the existence check and the configure.
                continue
        self._classic = alive

    def _style_classic(self, widget: tk.Misc, role: str) -> None:
        p = self.palette
        if role == "listbox":
            widget.configure(
                background=p.surface, foreground=p.ink,
                selectbackground=p.selection_bg, selectforeground=p.selection_ink,
                highlightthickness=1, highlightbackground=p.border,
                highlightcolor=p.focus, borderwidth=0, relief="flat",
                activestyle="none", font=self.fonts.body,
            )
        elif role == "text":
            widget.configure(
                background=p.surface, foreground=p.ink,
                selectbackground=p.selection_bg, selectforeground=p.selection_ink,
                insertbackground=p.ink, highlightthickness=1,
                highlightbackground=p.border, highlightcolor=p.focus,
                borderwidth=0, relief="flat",
            )
        elif role == "log":
            widget.configure(
                background=p.canvas, foreground=p.ink_muted,
                selectbackground=p.selection_bg, selectforeground=p.selection_ink,
                insertbackground=p.ink, highlightthickness=0,
                borderwidth=0, relief="flat", font=self.fonts.mono,
            )
        elif role == "toplevel":
            widget.configure(background=p.canvas)

    # --------------------------------------------------------------- styles
    def _apply(self) -> None:
        p, f, s = self.palette, self.fonts, self.style
        self.root.configure(background=p.canvas)

        # Classic-widget defaults, for widgets created after this point.
        # Existing ones are handled by _restyle_classic; option_add only
        # affects construction.
        self.root.option_add("*Font", f.body)
        self.root.option_add("*Toplevel.background", p.canvas)
        self.root.option_add("*Menu.background", p.raised)
        self.root.option_add("*Menu.foreground", p.ink)
        self.root.option_add("*Menu.activeBackground", p.accent)
        self.root.option_add("*Menu.activeForeground", p.ink_on_accent)
        self.root.option_add("*Menu.borderWidth", 0)
        # The combobox dropdown is a classic Listbox inside a Toplevel that ttk
        # builds internally; styling it is only reachable through the option DB.
        self.root.option_add("*TCombobox*Listbox.background", p.raised)
        self.root.option_add("*TCombobox*Listbox.foreground", p.ink)
        self.root.option_add("*TCombobox*Listbox.selectBackground", p.selection_bg)
        self.root.option_add("*TCombobox*Listbox.selectForeground", p.selection_ink)
        self.root.option_add("*TCombobox*Listbox.borderWidth", 0)
        self.root.option_add("*TCombobox*Listbox.font", f.body)

        s.configure(".", background=p.surface, foreground=p.ink,
                    fieldbackground=p.raised, troughcolor=p.track,
                    bordercolor=p.border, darkcolor=p.surface,
                    lightcolor=p.surface, focuscolor=p.focus,
                    font=f.body, borderwidth=0, relief="flat")

        self._frames()
        self._labels()
        self._buttons()
        self._inputs()
        self._indicators()
        self._containers()

    def _frames(self) -> None:
        p, s = self.palette, self.style
        for name, color in (
            ("TFrame", p.surface),
            ("Canvas.TFrame", p.canvas),
            ("Surface.TFrame", p.surface),
            ("Raised.TFrame", p.raised),
            ("Overlay.TFrame", p.overlay),
            # Chrome regions. The nav and header sit on the canvas tone so the
            # content surface reads as the thing in front.
            ("Nav.TFrame", p.canvas),
            ("Header.TFrame", p.canvas),
            ("Drawer.TFrame", p.canvas),
        ):
            s.configure(name, background=color, borderwidth=0, relief="flat")

        # Hairline rules. A 1px frame is the only divider primitive ttk offers
        # that can also span a grid cell, which TSeparator cannot do reliably
        # inside a padded container.
        s.configure("Rule.TFrame", background=p.border)
        s.configure("StrongRule.TFrame", background=p.border_strong)
        s.configure("AccentRule.TFrame", background=p.accent)

        # Bordered content panel: replaces LabelFrame for sections that want a
        # real heading row rather than a label notched into the border.
        s.configure("Card.TFrame", background=p.surface,
                    borderwidth=1, relief="solid", bordercolor=p.border)
        s.configure("Inset.TFrame", background=p.raised,
                    borderwidth=1, relief="solid", bordercolor=p.border)

    def _labels(self) -> None:
        p, f, s = self.palette, self.fonts, self.style
        # (style, font, foreground, background)
        specs = (
            ("TLabel", f.body, p.ink, p.surface),
            ("Display.TLabel", f.display, p.ink, p.canvas),
            ("Title.TLabel", f.title, p.ink, p.surface),
            ("Heading.TLabel", f.heading, p.ink, p.surface),
            ("Body.TLabel", f.body, p.ink, p.surface),
            ("Strong.TLabel", f.body_strong, p.ink, p.surface),
            ("Field.TLabel", f.label, p.ink_muted, p.surface),
            ("Muted.TLabel", f.label, p.ink_muted, p.surface),
            ("Faint.TLabel", f.label, p.ink_faint, p.surface),
            ("Micro.TLabel", f.micro, p.ink_faint, p.surface),
            # Data readouts.
            ("Mono.TLabel", f.mono, p.ink, p.surface),
            ("MonoMuted.TLabel", f.mono, p.ink_muted, p.surface),
            ("MonoMicro.TLabel", f.mono_micro, p.ink_faint, p.surface),
            # Semantic.
            ("Ok.TLabel", f.label_strong, p.ok, p.surface),
            ("Warn.TLabel", f.label_strong, p.warn, p.surface),
            ("Danger.TLabel", f.label_strong, p.danger, p.surface),
            ("Info.TLabel", f.label_strong, p.info, p.surface),
            ("Accent.TLabel", f.label_strong, p.accent, p.surface),
            # Chrome variants, drawn on the canvas tone.
            ("OnCanvas.TLabel", f.body, p.ink, p.canvas),
            ("CanvasMuted.TLabel", f.label, p.ink_muted, p.canvas),
            ("CanvasFaint.TLabel", f.label, p.ink_faint, p.canvas),
            ("CanvasMono.TLabel", f.mono, p.ink_muted, p.canvas),
            ("CanvasStrong.TLabel", f.body_strong, p.ink, p.canvas),
            ("CanvasOk.TLabel", f.label_strong, p.ok, p.canvas),
            ("CanvasWarn.TLabel", f.label_strong, p.warn, p.canvas),
            ("CanvasDanger.TLabel", f.label_strong, p.danger, p.canvas),
            ("CanvasAccent.TLabel", f.label_strong, p.accent, p.canvas),
            ("NavSection.TLabel", f.micro, p.ink_faint, p.canvas),
        )
        for name, font, fg, bg in specs:
            s.configure(name, font=font, foreground=fg, background=bg)

        # Soft-tinted status chips: a filled pill reads as a status at a glance
        # where colored text alone does not.
        for name, fg, bg in (
            ("OkChip.TLabel", p.ok, p.ok_soft),
            ("WarnChip.TLabel", p.warn, p.warn_soft),
            ("DangerChip.TLabel", p.danger, p.danger_soft),
            ("AccentChip.TLabel", p.accent, p.accent_soft),
        ):
            s.configure(name, font=f.label_strong, foreground=fg,
                        background=bg, padding=(SPACE["snug"], SPACE["hair"]))

    def _buttons(self) -> None:
        p, f, s = self.palette, self.fonts, self.style

        # Secondary / default: bordered, surface-filled. The common case.
        s.configure("TButton", font=f.body, foreground=p.ink,
                    background=p.raised, bordercolor=p.border_strong,
                    lightcolor=p.raised, darkcolor=p.raised,
                    borderwidth=1, relief="solid", focusthickness=1,
                    focuscolor=p.focus, padding=(SPACE["roomy"], SPACE["snug"]),
                    anchor="center")
        s.map("TButton",
              background=[("disabled", p.surface), ("pressed", p.overlay),
                          ("active", p.overlay)],
              foreground=[("disabled", p.ink_faint)],
              bordercolor=[("disabled", p.border), ("focus", p.focus),
                           ("active", p.border_strong)])

        # Primary: one per surface, for the action the surface exists to do.
        s.configure("Accent.TButton", font=f.body_strong, foreground=p.ink_on_accent,
                    background=p.accent, bordercolor=p.accent,
                    lightcolor=p.accent, darkcolor=p.accent,
                    borderwidth=1, relief="solid",
                    padding=(SPACE["loose"], SPACE["snug"]))
        s.map("Accent.TButton",
              background=[("disabled", p.overlay), ("pressed", p.accent_pressed),
                          ("active", p.accent_hover)],
              foreground=[("disabled", p.ink_faint)],
              bordercolor=[("disabled", p.border), ("pressed", p.accent_pressed),
                           ("active", p.accent_hover)])

        # Destructive: stop / cancel / delete.
        s.configure("Danger.TButton", font=f.body, foreground=p.danger,
                    background=p.raised, bordercolor=p.border_strong,
                    lightcolor=p.raised, darkcolor=p.raised,
                    borderwidth=1, relief="solid",
                    padding=(SPACE["roomy"], SPACE["snug"]))
        s.map("Danger.TButton",
              background=[("disabled", p.surface), ("pressed", p.danger_soft),
                          ("active", p.danger_soft)],
              foreground=[("disabled", p.ink_faint)],
              bordercolor=[("disabled", p.border), ("active", p.danger)])

        # Ghost: borderless, for tertiary actions in dense rows (browse, scan).
        for name, bg in (("Ghost.TButton", p.surface),
                         ("CanvasGhost.TButton", p.canvas)):
            s.configure(name, font=f.label, foreground=p.ink_muted,
                        background=bg, bordercolor=bg, lightcolor=bg,
                        darkcolor=bg, borderwidth=1, relief="solid",
                        padding=(SPACE["base"], SPACE["tight"]))
            s.map(name,
                  background=[("disabled", bg), ("pressed", p.overlay),
                              ("active", p.overlay)],
                  foreground=[("disabled", p.ink_faint), ("active", p.ink)],
                  bordercolor=[("active", p.border)])

        # Nav rail items. Left-anchored and full-width so the rail reads as a
        # list of destinations rather than a column of buttons.
        s.configure("Nav.TButton", font=f.body, foreground=p.ink_muted,
                    background=p.canvas, bordercolor=p.canvas,
                    lightcolor=p.canvas, darkcolor=p.canvas,
                    borderwidth=1, relief="solid", anchor="w",
                    padding=(SPACE["cozy"], SPACE["snug"]))
        s.map("Nav.TButton",
              background=[("pressed", p.overlay), ("active", p.overlay)],
              foreground=[("active", p.ink)],
              bordercolor=[("active", p.overlay)])

        s.configure("NavActive.TButton", font=f.body_strong, foreground=p.ink,
                    background=p.surface, bordercolor=p.border,
                    lightcolor=p.surface, darkcolor=p.surface,
                    borderwidth=1, relief="solid", anchor="w",
                    padding=(SPACE["cozy"], SPACE["snug"]))
        s.map("NavActive.TButton",
              background=[("pressed", p.surface), ("active", p.surface)],
              foreground=[("active", p.ink)])

        # Segmented control: adjacent toggles sharing one border run.
        for name, font in (("Segment.TButton", f.label),
                           ("SegmentActive.TButton", f.label_strong)):
            active = name.endswith("Active.TButton")
            s.configure(name, font=font,
                        foreground=p.ink_on_accent if active else p.ink_muted,
                        background=p.accent if active else p.raised,
                        bordercolor=p.accent if active else p.border,
                        lightcolor=p.accent if active else p.raised,
                        darkcolor=p.accent if active else p.raised,
                        borderwidth=1, relief="solid",
                        padding=(SPACE["cozy"], SPACE["tight"]))
            s.map(name,
                  background=[("active", p.accent_hover if active else p.overlay)],
                  foreground=[("active", p.ink_on_accent if active else p.ink)])

        s.configure("Toolbutton", font=f.label, foreground=p.ink_muted,
                    background=p.canvas, borderwidth=0, relief="flat",
                    padding=(SPACE["snug"], SPACE["hair"]))
        s.map("Toolbutton",
              background=[("selected", p.accent_soft), ("active", p.overlay)],
              foreground=[("selected", p.accent), ("active", p.ink)])

    def _inputs(self) -> None:
        p, f, s = self.palette, self.fonts, self.style

        s.configure("TEntry", foreground=p.ink, fieldbackground=p.raised,
                    insertcolor=p.ink, bordercolor=p.border,
                    lightcolor=p.border, darkcolor=p.border,
                    borderwidth=1, relief="solid",
                    padding=(SPACE["snug"], SPACE["tight"]))
        s.map("TEntry",
              bordercolor=[("focus", p.focus), ("invalid", p.danger)],
              lightcolor=[("focus", p.focus)], darkcolor=[("focus", p.focus)],
              fieldbackground=[("disabled", p.surface), ("readonly", p.surface)],
              foreground=[("disabled", p.ink_faint)])

        # Monospace entry for paths and ports, where character-level accuracy
        # matters more than prose legibility.
        s.configure("Mono.TEntry", font=f.mono, foreground=p.ink,
                    fieldbackground=p.raised, insertcolor=p.ink,
                    bordercolor=p.border, lightcolor=p.border,
                    darkcolor=p.border, borderwidth=1, relief="solid",
                    padding=(SPACE["snug"], SPACE["tight"]))
        s.map("Mono.TEntry",
              bordercolor=[("focus", p.focus), ("invalid", p.danger)],
              lightcolor=[("focus", p.focus)], darkcolor=[("focus", p.focus)],
              fieldbackground=[("disabled", p.surface)],
              foreground=[("disabled", p.ink_faint)])

        # Invalid state for inline field validation (the manga tab's numeric
        # fields are parsed with int()/float() and need somewhere to show it).
        s.configure("Invalid.TEntry", foreground=p.ink,
                    fieldbackground=p.danger_soft, insertcolor=p.ink,
                    bordercolor=p.danger, lightcolor=p.danger,
                    darkcolor=p.danger, borderwidth=1, relief="solid",
                    padding=(SPACE["snug"], SPACE["tight"]))

        for name, font in (("TCombobox", f.body), ("Mono.TCombobox", f.mono)):
            s.configure(name, font=font, foreground=p.ink,
                        fieldbackground=p.raised, background=p.raised,
                        bordercolor=p.border, lightcolor=p.border,
                        darkcolor=p.border, arrowcolor=p.ink_muted,
                        arrowsize=12, borderwidth=1, relief="solid",
                        padding=(SPACE["snug"], SPACE["tight"]))
            s.map(name,
                  bordercolor=[("focus", p.focus), ("hover", p.border_strong)],
                  lightcolor=[("focus", p.focus)], darkcolor=[("focus", p.focus)],
                  fieldbackground=[("disabled", p.surface),
                                   ("readonly", p.raised)],
                  foreground=[("disabled", p.ink_faint)],
                  arrowcolor=[("disabled", p.ink_faint), ("active", p.ink)],
                  selectbackground=[("readonly", p.raised)],
                  selectforeground=[("readonly", p.ink)])

        s.configure("TSpinbox", foreground=p.ink, fieldbackground=p.raised,
                    background=p.raised, bordercolor=p.border,
                    lightcolor=p.border, darkcolor=p.border,
                    arrowcolor=p.ink_muted, arrowsize=12,
                    borderwidth=1, relief="solid",
                    padding=(SPACE["snug"], SPACE["tight"]))
        s.map("TSpinbox",
              bordercolor=[("focus", p.focus)],
              fieldbackground=[("disabled", p.surface)],
              foreground=[("disabled", p.ink_faint)],
              arrowcolor=[("disabled", p.ink_faint), ("active", p.ink)])

        for name, bg in (("TCheckbutton", p.surface),
                         ("Canvas.TCheckbutton", p.canvas)):
            s.configure(name, font=f.body, foreground=p.ink, background=bg,
                        indicatorbackground=p.raised, indicatorforeground=p.accent,
                        bordercolor=p.border_strong, lightcolor=p.raised,
                        darkcolor=p.raised, focuscolor=p.focus,
                        padding=(SPACE["tight"], SPACE["hair"]))
            s.map(name,
                  background=[("active", bg)],
                  foreground=[("disabled", p.ink_faint)],
                  indicatorbackground=[("selected", p.accent),
                                       ("disabled", p.surface),
                                       ("active", p.overlay)],
                  indicatorforeground=[("selected", p.ink_on_accent)],
                  bordercolor=[("selected", p.accent), ("focus", p.focus),
                               ("disabled", p.border)])

        for name, bg in (("TRadiobutton", p.surface),
                         ("Canvas.TRadiobutton", p.canvas)):
            s.configure(name, font=f.body, foreground=p.ink, background=bg,
                        indicatorbackground=p.raised, indicatorforeground=p.accent,
                        bordercolor=p.border_strong, lightcolor=p.raised,
                        darkcolor=p.raised, focuscolor=p.focus,
                        padding=(SPACE["tight"], SPACE["hair"]))
            s.map(name,
                  background=[("active", bg)],
                  foreground=[("disabled", p.ink_faint)],
                  indicatorbackground=[("selected", p.accent),
                                       ("disabled", p.surface),
                                       ("active", p.overlay)],
                  indicatorforeground=[("selected", p.ink_on_accent)],
                  bordercolor=[("selected", p.accent), ("focus", p.focus)])

        for name, bg in (("TScale", p.surface), ("Canvas.TScale", p.canvas),
                         ("Horizontal.TScale", p.surface)):
            s.configure(name, background=bg, troughcolor=p.track,
                        bordercolor=p.border, lightcolor=p.accent,
                        darkcolor=p.accent, sliderthickness=14,
                        sliderrelief="flat")
            s.map(name,
                  background=[("disabled", bg)],
                  lightcolor=[("disabled", p.border_strong),
                              ("active", p.accent_hover)],
                  darkcolor=[("disabled", p.border_strong),
                             ("active", p.accent_hover)],
                  troughcolor=[("disabled", p.track)])

    def _indicators(self) -> None:
        p, s = self.palette, self.style

        for name in ("Horizontal.TProgressbar", "Accent.Horizontal.TProgressbar"):
            s.configure(name, background=p.accent, troughcolor=p.track,
                        bordercolor=p.track, lightcolor=p.accent,
                        darkcolor=p.accent, borderwidth=0, thickness=6)
        # A finished or failed run recolors its own bar: the bar is the thing
        # the eye is already on, so it is the cheapest place to report outcome.
        s.configure("Ok.Horizontal.TProgressbar", background=p.ok,
                    troughcolor=p.track, bordercolor=p.track,
                    lightcolor=p.ok, darkcolor=p.ok, borderwidth=0, thickness=6)
        s.configure("Danger.Horizontal.TProgressbar", background=p.danger,
                    troughcolor=p.track, bordercolor=p.track,
                    lightcolor=p.danger, darkcolor=p.danger,
                    borderwidth=0, thickness=6)
        s.configure("Warn.Horizontal.TProgressbar", background=p.warn,
                    troughcolor=p.track, bordercolor=p.track,
                    lightcolor=p.warn, darkcolor=p.warn,
                    borderwidth=0, thickness=6)
        # Thin variant for the always-visible header readout.
        s.configure("Header.Horizontal.TProgressbar", background=p.accent,
                    troughcolor=p.overlay, bordercolor=p.overlay,
                    lightcolor=p.accent, darkcolor=p.accent,
                    borderwidth=0, thickness=3)

        for name in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
            s.configure(name, background=p.border, troughcolor=p.canvas,
                        bordercolor=p.canvas, arrowcolor=p.ink_faint,
                        lightcolor=p.border, darkcolor=p.border,
                        borderwidth=0, relief="flat", gripcount=0, arrowsize=12)
            s.map(name,
                  background=[("pressed", p.accent), ("active", p.border_strong)],
                  arrowcolor=[("active", p.ink_muted)])

    def _containers(self) -> None:
        p, f, s = self.palette, self.fonts, self.style

        # LabelFrame is retained for the surfaces that still use it, restyled
        # so it does not look like a different design language than Card.TFrame.
        s.configure("TLabelframe", background=p.surface, bordercolor=p.border,
                    lightcolor=p.border, darkcolor=p.border,
                    borderwidth=1, relief="solid", padding=SPACE["roomy"])
        s.configure("TLabelframe.Label", background=p.surface,
                    foreground=p.ink_muted, font=f.label_strong)
        s.configure("Canvas.TLabelframe", background=p.canvas,
                    bordercolor=p.border, lightcolor=p.border,
                    darkcolor=p.border, borderwidth=1, relief="solid")
        s.configure("Canvas.TLabelframe.Label", background=p.canvas,
                    foreground=p.ink_muted, font=f.label_strong)

        s.configure("TNotebook", background=p.canvas, bordercolor=p.border,
                    borderwidth=0, tabmargins=(0, 0, 0, 0))
        s.configure("TNotebook.Tab", background=p.canvas, foreground=p.ink_muted,
                    bordercolor=p.border, lightcolor=p.canvas,
                    darkcolor=p.canvas, borderwidth=0, font=f.body,
                    padding=(SPACE["loose"], SPACE["snug"]))
        s.map("TNotebook.Tab",
              background=[("selected", p.surface), ("active", p.overlay)],
              foreground=[("selected", p.ink), ("active", p.ink)])

        s.configure("TSeparator", background=p.border)
        s.configure("TPanedwindow", background=p.canvas)
        s.configure("Sash", sashthickness=6, gripcount=0,
                    background=p.canvas, bordercolor=p.border,
                    lightcolor=p.canvas, darkcolor=p.canvas)

        s.configure("Treeview", background=p.surface, fieldbackground=p.surface,
                    foreground=p.ink, bordercolor=p.border, borderwidth=0,
                    font=f.body, rowheight=24)
        s.configure("Treeview.Heading", background=p.raised,
                    foreground=p.ink_muted, font=f.label_strong,
                    bordercolor=p.border, relief="flat",
                    padding=(SPACE["snug"], SPACE["tight"]))
        s.map("Treeview",
              background=[("selected", p.selection_bg)],
              foreground=[("selected", p.selection_ink)])
        s.map("Treeview.Heading", background=[("active", p.overlay)])


# ---------------------------------------------------------------------------
# Module-level access
# ---------------------------------------------------------------------------

_THEME: Optional[Theme] = None


def apply_theme(root: tk.Misc, name: str = DEFAULT_THEME) -> Theme:
    """Install the theme on ``root`` and make it the process-wide current one.

    Returns the :class:`Theme` rather than a ``ttk.Style`` (the old signature
    returned the style object). Callers that only need colors should prefer
    :func:`theme` so they are not holding a stale reference across a switch.
    """
    global _THEME
    _THEME = Theme(root, name)
    return _THEME


def theme() -> Theme:
    """The current theme.

    Raises if called before :func:`apply_theme`, which is a programming error:
    a widget cannot be styled before the shell exists.
    """
    if _THEME is None:
        raise RuntimeError("apply_theme(root) must run before theme() is used")
    return _THEME


def palette() -> Palette:
    """Shorthand for ``theme().palette``."""
    return theme().palette


def color(role: str) -> str:
    """Look up one palette role by name, for data-driven styling."""
    return getattr(palette(), role)
