import re
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext


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

    def _log(self, level, message):
        self.winfo_toplevel().after(0, lambda: self._insert_log(level, message))

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
