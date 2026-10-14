"""The application shell: status header, navigation rail, unified log drawer.

This replaces a bare ``ttk.Notebook`` whose five tabs each owned a slice of the
interface. Three problems came from that arrangement, and the shell exists to
fix each one:

* **Run state was local.** A translation started on the Subtitle tab reported
  its progress only on the Subtitle tab. The operator's actual question — "is
  it still running, how much is left" — could not be answered without first
  remembering where they had started it. :class:`StatusHeader` is always
  visible and subscribes to ``run_bus``, so every surface reports to the same
  place.
* **The log was in five places.** Four tabs each built a ``ScrolledText`` and
  subscribed it to their own channel, and a fifth floating window subscribed to
  everything. Diagnosing a pipeline run meant watching three of them at once.
  :class:`LogDrawer` is one view with the channel as a filter.
* **Navigation hid state.** Notebook tabs cannot show that the work behind them
  is still going. :class:`NavRail` puts a live indicator on each destination.

The surfaces themselves stay ordinary ``ttk.Frame`` subclasses, so they remain
independently constructible and testable.
"""
from __future__ import annotations

import queue
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from tkinter import ttk
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

from ui_helpers import (
    CHANNEL_APP,
    CHANNEL_LABELS,
    LOG_DRAIN_MAX,
    LOG_POLL_MS,
    LogBuffer,
    RunState,
    log_bus,
    run_bus,
)
from ui_kit import Spinner, StatusLine, hairline
from ui_theme import theme
from ui_tokens import GLYPH, LOG_LEVEL_ROLE, METRICS, SPACE

_HISTORY_MAX = 4000
_FILTER_ALL = "__all__"


@dataclass(frozen=True)
class LogEntry:
    """One rendered log line, with the time it was emitted.

    The timestamp is captured on arrival rather than at render time because the
    drawer re-renders history whenever the channel filter changes; formatting
    "now" during a replay would relabel every old line with the current clock.
    (``ui_helpers.append_log_lines`` stamps at render time, which is correct for
    its append-only callers, so this view does its own formatting rather than
    changing a helper three other things depend on.)
    """

    at: str
    level: str
    message: str
    channel: str


class LogView(ttk.Frame):
    """Filterable log text. Used by the drawer and by the detached window."""

    def __init__(self, parent: tk.Misc, history: Deque[LogEntry],
                 channels: Sequence[str], max_lines: int = 2000):
        super().__init__(parent, style="Drawer.TFrame")
        self._history = history
        self._max_lines = max_lines
        self._filter = _FILTER_ALL
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self, style="Drawer.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, SPACE["snug"]))

        self._filter_var = tk.StringVar(value=_FILTER_ALL)
        self._buttons: Dict[str, ttk.Radiobutton] = {}
        for value, label in [(_FILTER_ALL, "全部")] + [
            (channel, CHANNEL_LABELS.get(channel, channel)) for channel in channels
        ]:
            # Toolbutton radios: one obvious current filter, no dropdown to open.
            button = ttk.Radiobutton(
                toolbar, text=label, value=value, variable=self._filter_var,
                style="Toolbutton", command=self._on_filter_changed)
            button.pack(side="left", padx=(0, SPACE["hair"]))
            self._buttons[value] = button

        self._text = tk.Text(self, wrap="none", height=8, borderwidth=0,
                             state="disabled", takefocus=False)
        theme().register_classic(self._text, "log")
        self._text.grid(row=1, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(self, orient="vertical",
                                  command=self._text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self._text.configure(yscrollcommand=scrollbar.set)

        self._apply_tags(theme().palette)
        self._unsubscribe = theme().on_change(self._apply_tags)
        self.bind("<Destroy>", self._on_destroy, add="+")
        self.rerender()

    # ------------------------------------------------------------- styling
    def _apply_tags(self, palette) -> None:
        """Level colors live in tags, so a theme switch repaints old lines too."""
        for level, role in LOG_LEVEL_ROLE.items():
            self._text.tag_config(level, foreground=getattr(palette, role))
        self._text.tag_config("channel", foreground=palette.ink_faint)
        self._text.tag_config("time", foreground=palette.ink_faint)

    # -------------------------------------------------------------- render
    def _passes(self, entry: LogEntry) -> bool:
        return self._filter == _FILTER_ALL or entry.channel == self._filter

    def _on_filter_changed(self) -> None:
        self._filter = self._filter_var.get()
        self.rerender()

    def rerender(self) -> None:
        """Rebuild from history. Called on filter change and on first draw."""
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")
        self.append([entry for entry in self._history if self._passes(entry)],
                    prefiltered=True)

    def append(self, entries: Sequence[LogEntry], prefiltered: bool = False) -> None:
        """Append lines in one Tcl call, preserving the reader's scroll position."""
        if not prefiltered:
            entries = [entry for entry in entries if self._passes(entry)]
        if not entries:
            return
        # Only auto-follow when the reader is already at the bottom: scrolling
        # back to read an error must not be yanked away by the next line.
        follow = self._text.yview()[1] >= 0.999
        parts: List = []
        show_channel = self._filter == _FILTER_ALL
        for entry in entries:
            parts.extend((f"{entry.at} ", ("time",)))
            if show_channel:
                parts.extend((f"[{CHANNEL_LABELS.get(entry.channel, entry.channel)}] ",
                              ("channel",)))
            parts.extend((f"{entry.message}\n", (entry.level,)))
        self._text.configure(state="normal")
        try:
            self._text.insert("end", *parts)
            count = int(self._text.index("end-1c").split(".")[0])
            if count > self._max_lines:
                self._text.delete("1.0", f"{count - self._max_lines}.0")
            if follow:
                self._text.see("end")
        finally:
            self._text.configure(state="disabled")

    def clear(self) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")

    def _on_destroy(self, event) -> None:
        if event.widget is self:
            self._unsubscribe()


class LogDrawer(ttk.Frame):
    """Collapsible log region at the foot of the window.

    Collapsed by default: for a healthy run the operator wants the header's
    one-line status, and the log is what they open when something looks wrong.
    """

    def __init__(self, parent: tk.Misc, channels: Sequence[str],
                 on_visibility_changed: Optional[Callable[[bool], None]] = None):
        super().__init__(parent, style="Drawer.TFrame")
        self._channels = list(channels)
        self._on_visibility_changed = on_visibility_changed or (lambda _v: None)
        self._history: Deque[LogEntry] = deque(maxlen=_HISTORY_MAX)
        self._visible = False
        self._detached: Optional[tk.Toplevel] = None
        self._detached_view: Optional[LogView] = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        hairline(self).grid(row=0, column=0, sticky="ew")

        bar = ttk.Frame(self, style="Drawer.TFrame")
        bar.grid(row=1, column=0, sticky="ew",
                 padx=(SPACE["roomy"], SPACE["roomy"]), pady=(SPACE["tight"], 0))
        bar.columnconfigure(1, weight=1)

        self._toggle_button = ttk.Button(
            bar, text="▸  執行日誌   Ctrl+L", style="CanvasGhost.TButton",
            command=self.toggle)
        self._toggle_button.grid(row=0, column=0, sticky="w")

        self._summary = ttk.Label(bar, text="", style="CanvasFaint.TLabel")
        self._summary.grid(row=0, column=1, sticky="w", padx=(SPACE["base"], 0))

        actions = ttk.Frame(bar, style="Drawer.TFrame")
        actions.grid(row=0, column=2, sticky="e")
        ttk.Button(actions, text="另開視窗", style="CanvasGhost.TButton",
                   command=self.detach).pack(side="left", padx=(0, SPACE["tight"]))
        ttk.Button(actions, text="清空", style="CanvasGhost.TButton",
                   command=self.clear).pack(side="left")

        self._body = ttk.Frame(self, style="Drawer.TFrame")
        self._body.columnconfigure(0, weight=1)
        self._body.rowconfigure(0, weight=1)
        self._view = LogView(self._body, self._history, self._channels)
        self._view.grid(row=0, column=0, sticky="nsew",
                        padx=(SPACE["roomy"], SPACE["roomy"]),
                        pady=(SPACE["snug"], SPACE["roomy"]))

        # Worker threads emit log lines; Tcl is not thread-safe, so lines park
        # on a queue and the Tk thread's poll loop does every widget write.
        self._queue = LogBuffer()
        self._unsubscribe = log_bus.subscribe_all(self._on_log)
        self._poll_job: Optional[str] = self.after(LOG_POLL_MS, self._poll)
        self._pending_errors = 0
        self.bind("<Destroy>", self._on_destroy, add="+")

    # ---------------------------------------------------------- visibility
    @property
    def visible(self) -> bool:
        return self._visible

    def toggle(self) -> None:
        self.show(not self._visible)

    def show(self, visible: bool) -> None:
        if visible == self._visible:
            return
        self._visible = visible
        if visible:
            self._body.grid(row=2, column=0, sticky="nsew")
            self._toggle_button.configure(text="▾  執行日誌   Ctrl+L")
            self._pending_errors = 0
            self._refresh_summary()
            self._view.rerender()
        else:
            self._body.grid_remove()
            self._toggle_button.configure(text="▸  執行日誌   Ctrl+L")
        self._on_visibility_changed(visible)

    # ------------------------------------------------------------- detach
    def detach(self) -> tk.Toplevel:
        """Open the log in its own window (second monitor, or alongside a run)."""
        if self._detached is not None and self._detached.winfo_exists():
            self._detached.lift()
            return self._detached
        window = tk.Toplevel(self)
        window.title("執行日誌")
        window.geometry("860x460")
        theme().register_classic(window, "toplevel")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        self._detached_view = LogView(window, self._history, self._channels)
        self._detached_view.grid(row=0, column=0, sticky="nsew",
                                 padx=SPACE["roomy"], pady=SPACE["roomy"])
        window.protocol("WM_DELETE_WINDOW", self.close_detached)
        self._detached = window
        return window

    def close_detached(self) -> None:
        if self._detached is not None and self._detached.winfo_exists():
            self._detached.destroy()
        self._detached = None
        self._detached_view = None

    @property
    def detached(self) -> Optional[tk.Toplevel]:
        return self._detached

    # ---------------------------------------------------------------- data
    def clear(self) -> None:
        self._history.clear()
        self._view.clear()
        if self._detached_view is not None:
            self._detached_view.clear()
        self._pending_errors = 0
        self._refresh_summary()

    def _on_log(self, level: str, message: str, channel: str = CHANNEL_APP) -> None:
        self._queue.put((level, message, channel))

    def _poll(self) -> None:
        self._poll_job = None
        try:
            batch: List[LogEntry] = []
            dropped = self._queue.take_dropped()
            stamp = datetime.now().strftime("%H:%M:%S")
            if dropped:
                batch.append(LogEntry(
                    stamp, "WARNING",
                    f"日誌緩衝已滿，略過 {dropped} 筆較舊訊息。", CHANNEL_APP))
            while len(batch) < LOG_DRAIN_MAX:
                try:
                    level, message, channel = self._queue.get_nowait()
                except queue.Empty:
                    break
                batch.append(LogEntry(stamp, level, message, channel))

            if batch:
                self._history.extend(batch)
                self._view.append(batch)
                if self._detached_view is not None:
                    self._detached_view.append(batch)
                if not self._visible:
                    # While collapsed, the drawer still has to tell the operator
                    # that something worth reading arrived.
                    self._pending_errors += sum(
                        1 for entry in batch
                        if entry.level in ("ERROR", "WARNING"))
                    self._refresh_summary()
            self._poll_job = self.after(LOG_POLL_MS, self._poll)
        except tk.TclError:
            self._poll_job = None

    def _refresh_summary(self) -> None:
        if self._visible or not self._pending_errors:
            self._summary.configure(text="")
            return
        self._summary.configure(
            text=f"{GLYPH['warn']} {self._pending_errors} 筆警告／錯誤")

    def _on_destroy(self, event) -> None:
        if event.widget is not self:
            return
        self._unsubscribe()
        if self._poll_job is not None:
            self.after_cancel(self._poll_job)
            self._poll_job = None


class StatusHeader(ttk.Frame):
    """Always-visible state: the server, and whatever is running.

    Answers the two questions the operator has on walking back to the machine,
    in the two places the eye lands first, without requiring them to remember
    which surface started the work.
    """

    def __init__(self, parent: tk.Misc, on_toggle_theme: Callable[[], None],
                 on_toggle_log: Callable[[], None], reduce_motion: bool = False):
        super().__init__(parent, style="Header.TFrame")
        self.columnconfigure(1, weight=1)

        identity = ttk.Frame(self, style="Header.TFrame")
        identity.grid(row=0, column=0, sticky="w",
                      padx=(SPACE["page"], SPACE["section"]), pady=SPACE["cozy"])
        ttk.Label(identity, text="llama.cpp Manager",
                  style="CanvasStrong.TLabel").pack(anchor="w")
        self._server = StatusLine(identity, text="伺服器未啟動", state="idle",
                                  on_canvas=True, mono=True)
        self._server.pack(anchor="w", pady=(SPACE["hair"], 0))

        run = ttk.Frame(self, style="Header.TFrame")
        run.grid(row=0, column=1, sticky="ew", pady=SPACE["cozy"])
        run.columnconfigure(1, weight=1)

        self._spinner = Spinner(run, on_canvas=True, reduce_motion=reduce_motion)
        self._spinner.grid(row=0, column=0, sticky="w", padx=(0, SPACE["snug"]))

        self._run_label = ttk.Label(run, text="閒置", style="CanvasMuted.TLabel",
                                    anchor="w")
        self._run_label.grid(row=0, column=1, sticky="ew")

        self._run_detail = ttk.Label(run, text="", style="CanvasMono.TLabel",
                                     anchor="e")
        self._run_detail.grid(row=0, column=2, sticky="e", padx=(SPACE["base"], 0))

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress = ttk.Progressbar(
            run, variable=self._progress_var, maximum=100,
            style="Header.Horizontal.TProgressbar")
        # Gridded only while a measurable run is in flight; an empty track
        # sitting under an idle header is furniture.

        actions = ttk.Frame(self, style="Header.TFrame")
        actions.grid(row=0, column=2, sticky="e",
                     padx=(SPACE["section"], SPACE["page"]))
        self._theme_button = ttk.Button(actions, text="◐ 淺色",
                                        style="CanvasGhost.TButton",
                                        command=on_toggle_theme)
        self._theme_button.pack(side="left", padx=(0, SPACE["tight"]))
        ttk.Button(actions, text="日誌", style="CanvasGhost.TButton",
                   command=on_toggle_log).pack(side="left")

        hairline(self).grid(row=1, column=0, columnspan=3, sticky="ew")

    # --------------------------------------------------------------- server
    def set_server(self, running: bool, port: Optional[int] = None,
                   model: str = "") -> None:
        if running:
            name = model or "(未命名模型)"
            self._server.set("ok", f":{port}  {name}")
        else:
            self._server.set("idle", "伺服器未啟動")

    # ------------------------------------------------------------------ run
    def set_run(self, state: RunState) -> None:
        if state.state == "idle" or not state.label:
            self.clear_run()
            return
        self._run_label.configure(
            text=state.label,
            style={
                "busy": "CanvasAccent.TLabel",
                "ok": "CanvasOk.TLabel",
                "warn": "CanvasWarn.TLabel",
                "error": "CanvasDanger.TLabel",
            }.get(state.state, "CanvasMuted.TLabel"))
        self._run_detail.configure(text=state.detail or "")

        if state.active:
            self._spinner.start()
        else:
            self._spinner.stop()

        if state.percent is not None and state.active:
            self._progress_var.set(max(0.0, min(100.0, state.percent)))
            self._progress.grid(row=1, column=0, columnspan=3, sticky="ew",
                                pady=(SPACE["tight"], 0))
        else:
            self._progress.grid_remove()

    def clear_run(self) -> None:
        self._spinner.stop()
        self._run_label.configure(text="閒置", style="CanvasMuted.TLabel")
        self._run_detail.configure(text="")
        self._progress.grid_remove()

    def set_theme_label(self, dark: bool) -> None:
        # Names the destination, not the current state: a button labelled with
        # where it takes you is the one people read correctly.
        self._theme_button.configure(text="◐ 淺色" if dark else "◑ 深色")

    def set_reduce_motion(self, reduce_motion: bool) -> None:
        self._spinner.set_reduce_motion(reduce_motion)


class NavRail(ttk.Frame):
    """Vertical destination list with a live indicator per surface."""

    def __init__(self, parent: tk.Misc, on_select: Callable[[str], None]):
        super().__init__(parent, style="Nav.TFrame", width=METRICS["nav_width"])
        self.grid_propagate(False)
        self.columnconfigure(0, weight=1)
        self._on_select = on_select
        self._rows: Dict[str, Tuple[ttk.Button, ttk.Label]] = {}
        self._order: List[str] = []
        self._current: Optional[str] = None
        self._next_row = 0

    def add(self, key: str, label: str, accelerator: str = "") -> None:
        row = ttk.Frame(self, style="Nav.TFrame")
        row.grid(row=self._next_row, column=0, sticky="ew",
                 padx=(SPACE["base"], SPACE["base"]), pady=(0, SPACE["hair"]))
        row.columnconfigure(0, weight=1)

        button = ttk.Button(row, text=label, style="Nav.TButton",
                            command=lambda k=key: self._on_select(k))
        button.grid(row=0, column=0, sticky="ew")

        # Keyboard hint in its own column: a left-anchored button cannot also
        # right-align a shortcut, and an undiscoverable shortcut is one nobody
        # uses.
        if accelerator:
            ttk.Label(row, text=accelerator, style="NavSection.TLabel").grid(
                row=0, column=1, padx=(SPACE["tight"], 0))

        # Per-destination activity dot. A notebook tab cannot show that the
        # work behind it is still running; this is the whole reason the rail
        # replaced the tab strip.
        indicator = ttk.Label(row, text=" ", style="CanvasAccent.TLabel")
        indicator.grid(row=0, column=2, padx=(SPACE["tight"], 0))

        self._rows[key] = (button, indicator)
        self._order.append(key)
        self._next_row += 1

    def add_spacer(self) -> None:
        spacer = ttk.Frame(self, style="Nav.TFrame")
        spacer.grid(row=self._next_row, column=0, sticky="nsew")
        self.rowconfigure(self._next_row, weight=1)
        self._next_row += 1

    def add_footer(self, widget_factory: Callable[[tk.Misc], tk.Misc]) -> tk.Misc:
        widget = widget_factory(self)
        widget.grid(row=self._next_row, column=0, sticky="ew",
                    padx=SPACE["base"], pady=(0, SPACE["base"]))
        self._next_row += 1
        return widget

    def select(self, key: str) -> None:
        self._current = key
        for item, (button, _indicator) in self._rows.items():
            button.configure(
                style="NavActive.TButton" if item == key else "Nav.TButton")

    def set_busy(self, key: str, busy: bool, state: str = "busy") -> None:
        row = self._rows.get(key)
        if row is None:
            return
        _button, indicator = row
        if not busy:
            indicator.configure(text=" ", style="CanvasAccent.TLabel")
            return
        indicator.configure(
            text=GLYPH.get("busy" if state == "busy" else state, GLYPH["busy"]),
            style={
                "busy": "CanvasAccent.TLabel",
                "ok": "CanvasOk.TLabel",
                "warn": "CanvasWarn.TLabel",
                "error": "CanvasDanger.TLabel",
            }.get(state, "CanvasAccent.TLabel"))

    @property
    def keys(self) -> List[str]:
        return list(self._order)


class AppShell(ttk.Frame):
    """Window chrome plus a registry of surfaces.

    Surfaces are built by the caller as children of :attr:`content` and then
    registered, which keeps the shell ignorant of what they contain and keeps
    them constructible on their own.
    """

    def __init__(self, root: tk.Misc, config_manager=None,
                 channels: Sequence[str] = tuple(CHANNEL_LABELS)):
        super().__init__(root, style="Canvas.TFrame")
        # Not `self._root`: tkinter.Misc defines a `_root()` method that
        # nametowidget() calls internally, and shadowing it with the Tk
        # instance breaks winfo_toplevel() for this widget and every child.
        self._window = root
        self._config = config_manager
        self._surfaces: Dict[str, tk.Misc] = {}
        self._channel_owner: Dict[str, str] = {}
        self._current: Optional[str] = None
        reduce_motion = bool(
            config_manager.get("ui.reduce_motion", False)) if config_manager else False

        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        self.header = StatusHeader(
            self, on_toggle_theme=self.toggle_theme,
            on_toggle_log=self.toggle_log, reduce_motion=reduce_motion)
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.header.set_theme_label(theme().palette.dark)

        self.nav = NavRail(self, on_select=self.select)
        self.nav.grid(row=1, column=0, sticky="nsw")

        divider = hairline(self, orient="vertical")
        divider.grid(row=1, column=0, sticky="nse")

        self.content = ttk.Frame(self, style="Canvas.TFrame")
        self.content.grid(row=1, column=1, sticky="nsew")
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(0, weight=1)

        self.drawer = LogDrawer(self, channels)
        self.drawer.grid(row=2, column=0, columnspan=2, sticky="ew")

        run_bus.subscribe(self._on_run_state)
        self._bind_keys()

    # ------------------------------------------------------------ surfaces
    def register(self, key: str, widget: tk.Misc, label: str,
                 channels: Sequence[str] = ()) -> tk.Misc:
        """Register a surface and its nav entry.

        ``channels`` associates run-bus channels with this destination so the
        rail can show that work behind it is still in flight.
        """
        self._surfaces[key] = widget
        widget.grid(row=0, column=0, sticky="nsew")
        widget.grid_remove()
        self.nav.add(key, label, accelerator=f"⌃{len(self._surfaces)}")
        for channel in channels:
            self._channel_owner[channel] = key
        if self._current is None:
            self.select(key)
        return widget

    def select(self, key_or_widget=None):
        """Show a surface, or (with no argument) return the current one.

        The no-argument form returns the widget, mirroring
        ``ttk.Notebook.select()`` closely enough that the shell can stand in
        for the notebook it replaced.
        """
        if key_or_widget is None:
            return self._surfaces.get(self._current) if self._current else None

        key = key_or_widget
        if not isinstance(key_or_widget, str):
            key = next((candidate for candidate, widget in self._surfaces.items()
                        if widget is key_or_widget), None)
        if key is None or key not in self._surfaces:
            return None
        if key == self._current:
            return self._surfaces[key]

        if self._current is not None:
            self._surfaces[self._current].grid_remove()
        self._current = key
        surface = self._surfaces[key]
        surface.grid()
        self.nav.select(key)
        # Surfaces refresh themselves on entry (re-reading config a sibling
        # surface may have changed). The notebook used a <<NotebookTabChanged>>
        # binding for this; an explicit hook is one less indirection.
        hook = getattr(surface, "on_surface_shown", None)
        if callable(hook):
            hook()
        return surface

    @property
    def current(self) -> Optional[str]:
        return self._current

    # --------------------------------------------------------------- state
    def _on_run_state(self, state: RunState) -> None:
        self.header.set_run(state)
        key = self._channel_owner.get(state.channel)
        if key is not None:
            self.nav.set_busy(key, state.state != "idle", state.state)

    # -------------------------------------------------------------- chrome
    def toggle_theme(self) -> None:
        name = theme().toggle()
        self.header.set_theme_label(theme().palette.dark)
        if self._config is not None:
            self._config.set("ui.theme", name)

    def toggle_log(self) -> None:
        self.drawer.toggle()

    def _bind_keys(self) -> None:
        # Bound on the toplevel so a focused entry does not swallow them.
        top = self.winfo_toplevel()
        top.bind("<Control-l>", lambda _e: (self.toggle_log(), "break")[1])
        top.bind("<Control-L>", lambda _e: (self.toggle_log(), "break")[1])
        top.bind("<Control-Shift-T>",
                 lambda _e: (self.toggle_theme(), "break")[1])
        for index in range(1, 10):
            top.bind(
                f"<Control-Key-{index}>",
                lambda _e, i=index: (self._select_index(i - 1), "break")[1])

    def _select_index(self, index: int) -> None:
        keys = self.nav.keys
        if 0 <= index < len(keys):
            self.select(keys[index])
