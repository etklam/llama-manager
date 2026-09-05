import queue
import re
from datetime import datetime
from typing import Callable, Optional, Sequence
import tkinter as tk
from tkinter import ttk, scrolledtext

# Log widget pump: how often the main thread drains the per-widget queue, and
# the cap per tick so one chatty emitter can't monopolize a redraw.
LOG_POLL_MS = 80
LOG_DRAIN_MAX = 200
LOG_QUEUE_MAX = 4000


class LogBuffer(queue.Queue):
    """Nonblocking, thread-safe recent history with explicit overflow reporting."""

    def __init__(self, capacity=LOG_QUEUE_MAX):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        super().__init__()
        self.capacity = capacity
        self._dropped = 0

    def _put(self, item):
        # Queue holds its mutex while calling this hook.
        if len(self.queue) >= self.capacity:
            self.queue.popleft()
            self.unfinished_tasks -= 1
            self._dropped += 1
        self.queue.append(item)

    def take_dropped(self):
        with self.mutex:
            count, self._dropped = self._dropped, 0
            return count


def append_log_lines(widget, lines, max_lines):
    """Render a batch in one Tcl call and preserve the reader's scroll position."""
    if not lines:
        return
    follow = widget.yview()[1] >= 0.999
    timestamp = datetime.now().strftime("%H:%M:%S")
    parts = []
    for entry in lines:
        level, message = entry[:2]
        channel = f" [{entry[2]}]" if len(entry) > 2 else ""
        parts.extend((f"[{timestamp}] [{level}]{channel} {message}\n", (level,)))
    widget.config(state="normal")
    try:
        widget.insert(tk.END, *parts)
        count = int(widget.index('end-1c').split('.')[0])
        if count > max_lines:
            widget.delete(1.0, f"{count - max_lines}.0")
        if follow:
            widget.see(tk.END)
    finally:
        widget.config(state="disabled")


# Log channels. Every emitter names the subsystem it speaks for so a log widget
# can show only its own traffic: whisper emits one diagnostic line per decoded
# segment and the translator one per subtitle, and fanning both into all four
# sinks made every tab unreadable while tripling the Tk inserts per line.
CHANNEL_APP = "app"
CHANNEL_SERVER = "server"
CHANNEL_TRANSLATE = "translate"
CHANNEL_WHISPER = "whisper"
CHANNEL_PIPELINE = "pipeline"


class _LogBus:
    """Fan-out log bus with per-channel subscriptions.

    Subscribers registered via subscribe() take (level, message) and may filter
    to a set of channels. subscribe_all() takes (level, message, channel) and
    always sees everything — that is what the debug window uses.
    """

    def __init__(self):
        self._subscribers = []

    def subscribe(self, fn, channels: Optional[Sequence[str]] = None):
        """Subscribe to `channels` (all channels when None)."""
        entry = (fn, frozenset(channels) if channels else None, False)
        self._subscribers.append(entry)

        def _unsub():
            try:
                self._subscribers.remove(entry)
            except ValueError:
                pass

        return _unsub

    def subscribe_all(self, fn):
        """Subscribe to every channel with the channel name passed through."""
        entry = (fn, None, True)
        self._subscribers.append(entry)

        def _unsub():
            try:
                self._subscribers.remove(entry)
            except ValueError:
                pass

        return _unsub

    def emit(self, level, msg, channel: str = CHANNEL_APP):
        for fn, channels, wants_channel in list(self._subscribers):
            if channels is not None and channel not in channels:
                continue
            if wants_channel:
                fn(level, msg, channel)
            else:
                fn(level, msg)


log_bus = _LogBus()


def parse_dropped_paths(raw_data):
    paths = re.findall(r'\{([^}]+)\}', raw_data)
    if not paths:
        paths = raw_data.split()
    return paths


def populate_language_combo(combo, languages, saved_value=None):
    codes = list(languages.keys())
    names = [f"{c} - {languages[c]}" for c in codes]
    combo['values'] = names
    if saved_value:
        for i, n in enumerate(names):
            if n.startswith(saved_value):
                combo.current(i)
                break
        else:
            combo.current(0)
    elif names:
        combo.current(0)
    return combo


def extract_combo_code(var_value):
    return var_value.split(' - ')[0] if ' - ' in var_value else var_value


class LogMixin:
    _log_text: scrolledtext.ScrolledText
    _log_max_lines: int = 2000
    # Channel this widget emits on and subscribes to. Subclasses override it;
    # the default keeps standalone LogMixin users on the shared app channel.
    log_channel: str = CHANNEL_APP

    def _init_log_widget(self, parent, row=0, column=0, label="Log",
                         height=10, max_lines=2000, channels=None):
        frame = ttk.LabelFrame(parent, text=label, padding=8)
        frame.grid(row=row, column=column, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self._log_text = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD, height=height,
            font=("Consolas", 9), state="disabled")
        self._log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self._log_max_lines = max_lines

        self._log_text.tag_config("INFO", foreground="blue")
        self._log_text.tag_config("ERROR", foreground="red")
        self._log_text.tag_config("SUCCESS", foreground="green")
        self._log_text.tag_config("WARNING", foreground="orange")

        # Subscribe to this widget's own channel only, so a tab shows its own
        # subsystem's traffic. `channels` overrides it for widgets that need to
        # watch more than one (the Server tab also carries app-level notices).
        self._log_channels = tuple(channels) if channels else (self.log_channel,)
        self._log_queue = LogBuffer()
        self._log_unsub = log_bus.subscribe(
            self._log_bus_handler, channels=self._log_channels)
        self._log_after = self._log_text.after(LOG_POLL_MS, self._poll_log_queue)
        self._log_text.bind("<Destroy>", self._dispose_log, add="+")

    def _dispose_log(self, event):
        if event.widget is not self._log_text:
            return
        self._log_unsub()
        if self._log_after:
            self._log_text.after_cancel(self._log_after)
            self._log_after = None

    def _log_bus_handler(self, level, message):
        # Emitters include background threads (server stdout monitor, workers),
        # and Tk calls from those threads are unsafe. Park the line on the
        # queue; the main-thread poll loop does all widget work.
        self._log_queue.put((level, message))

    def _poll_log_queue(self):
        try:
            pending = []
            if isinstance(self._log_queue, LogBuffer):
                dropped = self._log_queue.take_dropped()
                if dropped:
                    pending.append(("WARNING", f"Log buffer full: skipped {dropped} older messages."))
            while len(pending) < LOG_DRAIN_MAX:
                try:
                    pending.append(self._log_queue.get_nowait())
                except queue.Empty:
                    break
            if pending:
                self._insert_log_lines(pending)
            self._log_after = self._log_text.after(LOG_POLL_MS, self._poll_log_queue)
        except tk.TclError:
            # Widget destroyed; stop polling.
            pass

    def _log(self, level, message):
        log_bus.emit(level, message, self.log_channel)

    def _insert_log(self, level, message):
        self._insert_log_lines([(level, message)])

    def _insert_log_lines(self, lines):
        append_log_lines(self._log_text, lines, self._log_max_lines)

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete(1.0, tk.END)
        self._log_text.config(state="disabled")
