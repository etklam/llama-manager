import re
import threading
from datetime import datetime
from typing import Any, Callable, List, Optional
import tkinter as tk
from tkinter import ttk, scrolledtext


class _LogBus:
    def __init__(self):
        self._subscribers = []

    def subscribe(self, fn):
        self._subscribers.append(fn)

        def _unsub():
            try:
                self._subscribers.remove(fn)
            except ValueError:
                pass

        return _unsub

    def emit(self, level, msg):
        for fn in list(self._subscribers):
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

    def _init_log_widget(self, parent, row=0, column=0, label="Log",
                         height=10, max_lines=2000):
        frame = tk.LabelFrame(parent, text=label)
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

        self._log_unsub = log_bus.subscribe(self._log_bus_handler)

    def _log_bus_handler(self, level, message):
        self.winfo_toplevel().after(0, lambda: self._insert_log(level, message))

    def _log(self, level, message):
        log_bus.emit(level, message)

    def _insert_log(self, level, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] [{level}] {message}\n"
        self._log_text.config(state="normal")
        self._log_text.insert(tk.END, line, level)
        self._log_text.see(tk.END)
        lines = int(self._log_text.index('end-1c').split('.')[0])
        if lines > self._log_max_lines:
            self._log_text.delete(1.0, f"{lines - self._log_max_lines}.0")
        self._log_text.config(state="disabled")

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete(1.0, tk.END)
        self._log_text.config(state="disabled")


class BatchRunner:
    """Shared lifecycle shell for batch UI loops (translate/whisper/etc).

    Owns: button toggling, stop flag, file iteration with stop check,
    per-file exception catching, progress+done callbacks marshalled via
    schedule_fn. Each caller supplies per_file_fn(file) for the real work.
    """

    def __init__(self, start_btn, stop_btn, log_fn, schedule_fn):
        self._start_btn = start_btn
        self._stop_btn = stop_btn
        self._log = log_fn
        self._schedule = schedule_fn
        self._running = False
        self._stop_requested = False
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._running

    def run(self, files: List[str], per_file_fn: Callable[[str], Any],
            on_progress: Optional[Callable[[int, str], None]] = None,
            on_done: Optional[Callable[..., None]] = None) -> None:
        """Run the batch loop synchronously. Caller typically invokes in a thread."""
        self._running = True
        self._stop_requested = False
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")

        try:
            for i, filepath in enumerate(files):
                if self._stop_requested:
                    self._log("WARNING", "Stopped by user")
                    break
                if on_progress:
                    on_progress(i, filepath)
                try:
                    per_file_fn(filepath)
                except Exception as e:
                    self._log("ERROR", f"Failed: {filepath} - {e}")
        finally:
            self._running = False
            stopped = self._stop_requested
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            if on_done:
                self._schedule(lambda: on_done(stopped=stopped))

    def run_async(self, files: List[str], per_file_fn: Callable[[str], Any],
                  on_progress=None, on_done=None) -> threading.Thread:
        """Spawn run() in a daemon thread. Returns the thread handle."""
        self._thread = threading.Thread(
            target=self.run,
            args=(files, per_file_fn),
            kwargs={"on_progress": on_progress, "on_done": on_done},
            daemon=True,
        )
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop_requested = True
        self._log("WARNING", "Stopping after current file...")
