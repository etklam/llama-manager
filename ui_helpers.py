import re
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext


def parse_dropped_paths(raw_data):
    paths = re.findall(r'\{([^}]+)\}', raw_data)
    if not paths:
        paths = raw_data.split()
    return paths


class LogMixin:
    _log_text: scrolledtext.ScrolledText

    def _init_log_widget(self, parent, row=0, column=0):
        frame = tk.LabelFrame(parent, text="Log", padding="5")
        frame.grid(row=row, column=column, sticky=(tk.W, tk.E, tk.N, tk.S))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self._log_text = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD, height=10,
            font=("Consolas", 9), state="disabled")
        self._log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self._log_text.tag_config("INFO", foreground="blue")
        self._log_text.tag_config("ERROR", foreground="red")
        self._log_text.tag_config("SUCCESS", foreground="green")
        self._log_text.tag_config("WARNING", foreground="orange")

    def _log(self, level, message):
        self.winfo_toplevel().after(0, lambda: self._insert_log(level, message))

    def _insert_log(self, level, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] [{level}] {message}\n"
        self._log_text.config(state="normal")
        self._log_text.insert(tk.END, line, level)
        self._log_text.see(tk.END)
        self._log_text.config(state="disabled")

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete(1.0, tk.END)
        self._log_text.config(state="disabled")
