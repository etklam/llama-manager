"""Reusable widgets for the desktop shell.

Every surface in this app was previously hand-assembling the same four things —
a titled panel, a label-plus-control row, a status line, a button row — with
different padding and different colors each time. That is why no two tabs lined
up. These components are the single implementation of each, so consistency is
structural rather than something to remember.

Everything here reads color and type from :mod:`ui_theme`, and anything that
cannot be expressed as a ttk style name subscribes to theme changes so a
light/dark switch repaints it too.
"""
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ui_theme import theme
from ui_tokens import GLYPH, SPACE, SPINNER_FRAMES, SPINNER_INTERVAL_MS, STATIC_BUSY_GLYPH

# Which label style a given semantic state uses, on each of the two
# backgrounds a component can sit on.
_STATE_STYLE = {
    "idle": ("Faint.TLabel", "CanvasFaint.TLabel"),
    "busy": ("Accent.TLabel", "CanvasAccent.TLabel"),
    "ok": ("Ok.TLabel", "CanvasOk.TLabel"),
    "warn": ("Warn.TLabel", "CanvasWarn.TLabel"),
    "error": ("Danger.TLabel", "CanvasDanger.TLabel"),
}


def hairline(parent: tk.Misc, orient: str = "horizontal",
             style: str = "Rule.TFrame") -> ttk.Frame:
    """A 1px divider.

    ``ttk.Separator`` draws with the theme's border element and ignores the
    palette in clam; a 1px styled frame is predictable and can span grid cells.
    """
    if orient == "horizontal":
        return ttk.Frame(parent, height=1, style=style)
    return ttk.Frame(parent, width=1, style=style)


class Section(ttk.Frame):
    """A titled content panel: heading, optional description, body.

    Replaces ``ttk.LabelFrame`` as the grouping primitive. A LabelFrame notches
    its title into the border, which at this type size reads as a form from
    1998 and gives no room for the one-line explanation most of these groups
    need. Here the heading is a real row, so a section can carry a description
    and a trailing control (a toggle, a count, a status) without fighting the
    border.

    Compose into ``section.body``, never into the Section itself.
    """

    def __init__(self, parent: tk.Misc, title: str, description: str = "",
                 trailing: Optional[Callable[[tk.Misc], tk.Misc]] = None,
                 padding: Optional[int] = None, flush: bool = False):
        super().__init__(parent, style="Surface.TFrame" if flush else "Card.TFrame")
        pad = SPACE["roomy"] if padding is None else padding
        self.columnconfigure(0, weight=1)

        header = ttk.Frame(self, style="Surface.TFrame")
        header.grid(row=0, column=0, sticky="ew",
                    padx=(pad, pad), pady=(pad, SPACE["snug"]))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text=title, style="Heading.TLabel").grid(
            row=0, column=0, sticky="w")
        if trailing is not None:
            trailing(header).grid(row=0, column=1, sticky="e")
        if description:
            ttk.Label(header, text=description, style="Muted.TLabel",
                      wraplength=640, justify="left").grid(
                row=1, column=0, columnspan=2, sticky="w", pady=(SPACE["hair"], 0))

        self.body = ttk.Frame(self, style="Surface.TFrame")
        self.body.grid(row=1, column=0, sticky="nsew", padx=(pad, pad),
                       pady=(0, pad))
        self.body.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)


class Disclosure(ttk.Frame):
    """A section that starts collapsed. For settings touched once a month.

    Better than the old "Show Advanced" button, which lived in the run-control
    row and reflowed the surface when toggled: the trigger now sits on the
    group it opens, so the layout above it never moves and the operator can see
    that the group exists without opening it.

    Compose into ``disclosure.body``.
    """

    def __init__(self, parent: tk.Misc, title: str, description: str = "",
                 expanded: bool = False):
        super().__init__(parent, style="Card.TFrame")
        self.columnconfigure(0, weight=1)
        self._expanded = expanded
        self._description = description

        self._button = ttk.Button(self, style="Ghost.TButton",
                                  command=self.toggle)
        self._button.grid(row=0, column=0, sticky="w",
                          padx=(SPACE["roomy"], SPACE["roomy"]),
                          pady=(SPACE["cozy"], SPACE["cozy"]))
        self._title = title

        self.body = ttk.Frame(self, style="Surface.TFrame")
        self.body.columnconfigure(0, weight=1)
        self._render()

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._render()

    @property
    def expanded(self) -> bool:
        return self._expanded

    def _render(self) -> None:
        self._button.configure(
            text=f"{'▾' if self._expanded else '▸'}  {self._title}")
        if self._expanded:
            self.body.grid(row=1, column=0, sticky="ew",
                           padx=(SPACE["roomy"], SPACE["roomy"]),
                           pady=(0, SPACE["roomy"]))
        else:
            self.body.grid_remove()


class FieldGrid(ttk.Frame):
    """Label-and-control rows on one aligned grid.

    The alignment is the point: every surface here has a form, and they
    previously each chose their own label column width, so the controls in the
    Server tab did not line up with the controls in the Whisper tab. ``add``
    returns the cell frame, so a caller can put a control plus a Browse button
    plus a hint in one row without breaking that alignment.
    """

    def __init__(self, parent: tk.Misc, label_width: Optional[int] = None,
                 columns: int = 1, gap: int = SPACE["snug"]):
        super().__init__(parent, style="Surface.TFrame")
        self._gap = gap
        self._columns = columns
        self._row = 0
        self._label_width = label_width
        for index in range(columns):
            self.columnconfigure(index * 2 + 1, weight=1)

    def add(self, label: str, column: int = 0, hint: str = "",
            span: int = 1) -> ttk.Frame:
        """Add one row and return the frame the control should be built in."""
        row = self._row
        label_col = column * 2
        widget = ttk.Label(self, text=label, style="Field.TLabel", anchor="w")
        if self._label_width:
            widget.configure(width=self._label_width)
        widget.grid(row=row, column=label_col, sticky="w",
                    padx=(0, SPACE["base"]), pady=(0, self._gap))

        cell = ttk.Frame(self, style="Surface.TFrame")
        cell.grid(row=row, column=label_col + 1,
                  columnspan=span * 2 - 1, sticky="ew",
                  padx=(0, SPACE["loose"] if column < self._columns - 1 else 0),
                  pady=(0, self._gap))
        cell.columnconfigure(0, weight=1)

        if hint:
            ttk.Label(self, text=hint, style="Micro.TLabel",
                      wraplength=420, justify="left").grid(
                row=row + 1, column=label_col + 1, sticky="w",
                pady=(0, self._gap))
            self._row += 1

        if column == self._columns - 1:
            self._row += 1
        return cell

    def newline(self) -> None:
        """Force the next ``add`` onto a fresh row (for partial multi-column rows)."""
        self._row += 1


class StatusLine(ttk.Frame):
    """Glyph + text, in a semantic color. The app's one status primitive.

    State is carried three ways at once — glyph, color, and words — because
    color alone fails for red/green confusion, and a glyph alone does not say
    what happened. Everything that previously set ``foreground="red"`` by hand
    goes through here.
    """

    def __init__(self, parent: tk.Misc, text: str = "", state: str = "idle",
                 on_canvas: bool = False, mono: bool = False):
        super().__init__(parent, style="Canvas.TFrame" if on_canvas else "Surface.TFrame")
        self._on_canvas = on_canvas
        self._glyph = ttk.Label(self, text=GLYPH.get(state, GLYPH["idle"]))
        self._glyph.pack(side="left", padx=(0, SPACE["snug"]))
        base = ("CanvasMono.TLabel" if on_canvas else "MonoMuted.TLabel") if mono else \
               ("OnCanvas.TLabel" if on_canvas else "Body.TLabel")
        self._text_style = base
        self._text = ttk.Label(self, text=text, style=base, anchor="w")
        self._text.pack(side="left", fill="x", expand=True)
        self.set(state, text)

    def set(self, state: str, text: Optional[str] = None) -> None:
        styles = _STATE_STYLE.get(state, _STATE_STYLE["idle"])
        self._glyph.configure(
            text=GLYPH.get(state, GLYPH["idle"]),
            style=styles[1] if self._on_canvas else styles[0])
        if text is not None:
            self._text.configure(text=text)

    @property
    def text(self) -> str:
        return self._text.cget("text")


class Spinner(ttk.Label):
    """The one animated element: says "work is happening", nothing else.

    Tk has no transitions, so this is a periodic ``after`` callback cycling
    braille frames. ``reduce_motion`` replaces the animation with a static
    glyph — Tk has no ``prefers-reduced-motion``, so the preference arrives from
    config instead.
    """

    def __init__(self, parent: tk.Misc, on_canvas: bool = True,
                 reduce_motion: bool = False):
        super().__init__(parent, text=" ",
                         style="CanvasAccent.TLabel" if on_canvas else "Accent.TLabel")
        self._frame = 0
        self._job: Optional[str] = None
        self._reduce_motion = reduce_motion
        self.bind("<Destroy>", self._on_destroy, add="+")

    def start(self) -> None:
        if self._reduce_motion:
            self.configure(text=STATIC_BUSY_GLYPH)
            return
        if self._job is None:
            self._tick()

    def stop(self) -> None:
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        self.configure(text=" ")

    def set_reduce_motion(self, reduce_motion: bool) -> None:
        self._reduce_motion = reduce_motion
        if reduce_motion and self._job is not None:
            self.after_cancel(self._job)
            self._job = None
            self.configure(text=STATIC_BUSY_GLYPH)

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(SPINNER_FRAMES)
        try:
            self.configure(text=SPINNER_FRAMES[self._frame])
            self._job = self.after(SPINNER_INTERVAL_MS, self._tick)
        except tk.TclError:
            self._job = None

    def _on_destroy(self, event) -> None:
        if event.widget is self and self._job is not None:
            self.after_cancel(self._job)
            self._job = None


class SegmentedControl(ttk.Frame):
    """A row of mutually exclusive options bound to a StringVar.

    Exists to kill a specific bug class: the manga tab stored its display text
    *as* its state (``tk.StringVar(value='同一文件')``) and compared against
    that literal, so renaming a label changed behavior. Here the value and the
    label are separate — ``options`` is ``(value, label)`` — and the variable
    only ever holds the stable value.

    Preferred over a readonly Combobox for 2-4 options: the choices are visible
    without a click, which matters for settings that change what a run does.
    """

    def __init__(self, parent: tk.Misc, variable: tk.StringVar,
                 options: Sequence[Tuple[str, str]],
                 command: Optional[Callable[[str], None]] = None):
        super().__init__(parent, style="Surface.TFrame")
        self._variable = variable
        self._command = command
        self._buttons: Dict[str, ttk.Button] = {}
        for index, (value, label) in enumerate(options):
            button = ttk.Button(self, text=label, style="Segment.TButton",
                                command=lambda v=value: self._select(v))
            button.grid(row=0, column=index,
                        padx=(0, SPACE["hair"] if index < len(options) - 1 else 0))
            self._buttons[value] = button
        self._refresh()
        # Follow external writes to the variable (config load, job restore).
        self._trace = variable.trace_add("write", lambda *_: self._refresh())

    def _select(self, value: str) -> None:
        if self._variable.get() != value:
            self._variable.set(value)
        self._refresh()
        if self._command is not None:
            self._command(value)

    def _refresh(self) -> None:
        current = self._variable.get()
        for value, button in self._buttons.items():
            try:
                button.configure(
                    style="SegmentActive.TButton" if value == current
                    else "Segment.TButton")
            except tk.TclError:
                pass

    def set_enabled(self, enabled: bool) -> None:
        for button in self._buttons.values():
            button.state(("!disabled",) if enabled else ("disabled",))


class ActionBar(ttk.Frame):
    """The run controls for a surface: primary right, secondary left of it.

    One consistent shape so "the button that starts the thing" is in the same
    place on every surface. Buttons are created through :meth:`primary`,
    :meth:`secondary` and :meth:`danger` rather than passed in, so a surface
    cannot accidentally make its start button look like someone else's.
    """

    def __init__(self, parent: tk.Misc, on_canvas: bool = False):
        super().__init__(parent, style="Canvas.TFrame" if on_canvas else "Surface.TFrame")
        self.columnconfigure(0, weight=1)
        self._leading = ttk.Frame(self, style=self.cget("style"))
        self._leading.grid(row=0, column=0, sticky="w")
        self._trailing = ttk.Frame(self, style=self.cget("style"))
        self._trailing.grid(row=0, column=1, sticky="e")

    def primary(self, text: str, command: Callable[[], None]) -> ttk.Button:
        button = ttk.Button(self._trailing, text=text, command=command,
                            style="Accent.TButton")
        button.pack(side="left", padx=(SPACE["snug"], 0))
        return button

    def danger(self, text: str, command: Callable[[], None]) -> ttk.Button:
        button = ttk.Button(self._trailing, text=text, command=command,
                            style="Danger.TButton")
        button.pack(side="left", padx=(SPACE["snug"], 0))
        return button

    def secondary(self, text: str, command: Callable[[], None]) -> ttk.Button:
        button = ttk.Button(self._leading, text=text, command=command)
        button.pack(side="left", padx=(0, SPACE["snug"]))
        return button

    def ghost(self, text: str, command: Callable[[], None]) -> ttk.Button:
        button = ttk.Button(self._leading, text=text, command=command,
                            style="Ghost.TButton")
        button.pack(side="left", padx=(0, SPACE["snug"]))
        return button

    def note(self, textvariable: Optional[tk.StringVar] = None,
             text: str = "") -> ttk.Label:
        """A hint beside the controls — why the primary is disabled, usually."""
        label = ttk.Label(self._leading, text=text, textvariable=textvariable,
                          style="Muted.TLabel")
        label.pack(side="left", padx=(0, SPACE["snug"]))
        return label


class Readout(ttk.Frame):
    """A labelled numeric value in mono. For rates, counts, budgets, VRAM.

    Value above label, value in the data face: the numbers are what the
    operator came to read, so they get the stronger position and a font where
    a changing digit does not reflow the row.
    """

    def __init__(self, parent: tk.Misc, label: str, value: str = "—",
                 on_canvas: bool = False):
        super().__init__(parent, style="Canvas.TFrame" if on_canvas else "Surface.TFrame")
        self._value = ttk.Label(
            self, text=value,
            style="CanvasMono.TLabel" if on_canvas else "Mono.TLabel")
        self._value.pack(anchor="w")
        ttk.Label(self, text=label,
                  style="CanvasFaint.TLabel" if on_canvas else "Micro.TLabel").pack(
            anchor="w")

    def set(self, value: str) -> None:
        self._value.configure(text=value)


class EmptyState(ttk.Frame):
    """What a surface shows before it has anything to show.

    The product register asks for empty states that teach the interface rather
    than announcing emptiness, so this takes a title, an explanation, and the
    action that resolves it — which makes it the natural home for first-run
    guidance (no model directory configured, no whisper-cli path set).
    """

    def __init__(self, parent: tk.Misc, title: str, body: str,
                 action: Optional[Tuple[str, Callable[[], None]]] = None,
                 glyph: str = "◌"):
        super().__init__(parent, style="Inset.TFrame", padding=SPACE["section"])
        self.columnconfigure(0, weight=1)
        ttk.Label(self, text=glyph, style="Faint.TLabel").grid(row=0, column=0)
        ttk.Label(self, text=title, style="Strong.TLabel").grid(
            row=1, column=0, pady=(SPACE["snug"], SPACE["hair"]))
        ttk.Label(self, text=body, style="Muted.TLabel", wraplength=420,
                  justify="center").grid(row=2, column=0)
        if action is not None:
            text, command = action
            ttk.Button(self, text=text, command=command,
                       style="Accent.TButton").grid(
                row=3, column=0, pady=(SPACE["roomy"], 0))


class ValidatedSpin(ttk.Frame):
    """A numeric field that reports its own bad input instead of raising.

    The manga surface read eight free-text entries straight into ``int()`` and
    ``float()`` inside the start handler, so one typo aborted the run with a
    generic dialog that did not name the field. Validation belongs on the
    field: :meth:`value` returns ``None`` and marks the entry when the text is
    not a number in range, and :meth:`error` names it for the caller.
    """

    def __init__(self, parent: tk.Misc, variable: tk.StringVar, label: str,
                 minimum: float = 0, maximum: float = 1_000_000,
                 integer: bool = True, width: int = 9):
        super().__init__(parent, style="Surface.TFrame")
        self._variable = variable
        self._label = label
        self._minimum = minimum
        self._maximum = maximum
        self._integer = integer
        self._entry = ttk.Entry(self, textvariable=variable, width=width,
                                style="Mono.TEntry")
        self._entry.pack(side="left")
        self._error: Optional[str] = None

    @property
    def label(self) -> str:
        return self._label

    def value(self):
        """Parsed value, or ``None`` when invalid (and the field is marked)."""
        raw = self._variable.get().strip()
        try:
            parsed = int(raw) if self._integer else float(raw)
        except ValueError:
            self._mark(f"{self._label}：「{raw or '空白'}」不是數字")
            return None
        if not self._minimum <= parsed <= self._maximum:
            self._mark(
                f"{self._label}：{parsed} 超出範圍 "
                f"{self._minimum:g}–{self._maximum:g}")
            return None
        self._clear()
        return parsed

    def error(self) -> Optional[str]:
        return self._error

    def _mark(self, message: str) -> None:
        self._error = message
        self._entry.configure(style="Invalid.TEntry")

    def _clear(self) -> None:
        self._error = None
        self._entry.configure(style="Mono.TEntry")


class ScrollArea(ttk.Frame):
    """A vertically scrollable content region.

    Needed because the shell now spends fixed height on a persistent header and
    an optional log drawer, and the Server surface has more fields than the
    remaining space on a 1366x768 screen. Tk has no scrollable frame, so this
    is the standard Canvas-plus-inner-frame construction with the three
    platform wheel bindings.

    Compose into ``area.body``.
    """

    def __init__(self, parent: tk.Misc, style: str = "Canvas.TFrame"):
        super().__init__(parent, style=style)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        palette = theme().palette
        self._canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0,
                                 background=palette.canvas)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._scrollbar = ttk.Scrollbar(self, orient="vertical",
                                        command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._on_scroll_set)

        self.body = ttk.Frame(self._canvas, style=style)
        self._window = self._canvas.create_window(
            (0, 0), window=self.body, anchor="nw")

        self.body.bind("<Configure>", self._on_body_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_wheel()
        self._unsubscribe = theme().on_change(
            lambda p: self._canvas.configure(background=p.canvas))
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _on_scroll_set(self, first: str, last: str) -> None:
        # Hide the scrollbar when everything fits: a permanently visible
        # disabled scrollbar is noise on a surface that rarely overflows.
        if float(first) <= 0.0 and float(last) >= 1.0:
            self._scrollbar.grid_remove()
        else:
            self._scrollbar.grid(row=0, column=1, sticky="ns")
        self._scrollbar.set(first, last)

    def _on_body_configure(self, _event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        # Match the inner frame to the viewport width so content wraps rather
        # than scrolling horizontally.
        self._canvas.itemconfigure(self._window, width=event.width)

    def _bind_wheel(self) -> None:
        # bind_all would steal the wheel from the log drawer and every listbox;
        # enter/leave scoping keeps it to the hovered region.
        self._canvas.bind("<Enter>", lambda _e: self._wheel_bind())
        self._canvas.bind("<Leave>", lambda _e: self._wheel_unbind())

    def _wheel_bind(self) -> None:
        if sys.platform == "linux":
            self._canvas.bind_all("<Button-4>", self._on_wheel_linux)
            self._canvas.bind_all("<Button-5>", self._on_wheel_linux)
        else:
            self._canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _wheel_unbind(self) -> None:
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                self._canvas.unbind_all(sequence)
            except tk.TclError:
                pass

    def _on_wheel(self, event) -> None:
        # Windows reports multiples of 120; macOS reports small signed deltas.
        step = -1 * (event.delta // 120) if abs(event.delta) >= 120 else -1 * event.delta
        self._canvas.yview_scroll(int(step), "units")

    def _on_wheel_linux(self, event) -> None:
        self._canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

    def _on_destroy(self, event) -> None:
        if event.widget is self:
            self._wheel_unbind()
            self._unsubscribe()


def chip(parent: tk.Misc, text: str, kind: str = "accent") -> ttk.Label:
    """A small filled status pill. ``kind`` is ok / warn / danger / accent."""
    styles = {"ok": "OkChip.TLabel", "warn": "WarnChip.TLabel",
              "danger": "DangerChip.TLabel", "accent": "AccentChip.TLabel"}
    return ttk.Label(parent, text=text, style=styles.get(kind, styles["accent"]))


def labelled_scale(parent: tk.Misc, variable, minimum: int, maximum: int,
                   length: int = 150, fmt: str = "{:d}",
                   on_canvas: bool = False) -> Tuple[ttk.Scale, ttk.Label]:
    """A scale with a mono value readout that tracks it.

    ttk.Scale writes floats into its variable even when that variable is an
    IntVar, so binding a label straight to it renders "23.548387096774196".
    Three surfaces each rediscovered that and wrote their own formatter; this
    is the shared one.
    """
    holder = ttk.Frame(parent, style="Canvas.TFrame" if on_canvas else "Surface.TFrame")
    scale = ttk.Scale(holder, from_=minimum, to=maximum, variable=variable,
                      orient="horizontal", length=length)
    scale.pack(side="left")
    readout = ttk.Label(holder, width=5,
                        style="CanvasMono.TLabel" if on_canvas else "Mono.TLabel")

    def render(raw) -> None:
        value = float(raw)
        readout.configure(text=fmt.format(int(value) if "d" in fmt else value))

    readout.pack(side="left", padx=(SPACE["snug"], 0))
    scale.configure(command=render)
    render(variable.get())
    holder.scale = scale          # type: ignore[attr-defined]
    holder.readout = readout      # type: ignore[attr-defined]
    return holder, readout
