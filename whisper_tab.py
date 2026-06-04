"""Whisper Tab - tkinter GUI for whisper.cpp speech recognition."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tkinterdnd2 import DND_FILES

from whisper_controller import WhisperController

from constants import SUPPORTED_MEDIA, WHISPER_LANGUAGES
from ui_helpers import LogMixin, parse_dropped_paths


class WhisperTab(LogMixin, ttk.Frame):

    def __init__(self, parent, config_manager, get_whisper_models, scan_whisper_models, on_srt_generated):
        super().__init__(parent)
        self._config_manager = config_manager
        self._get_whisper_models = get_whisper_models
        self._scan_whisper_models = scan_whisper_models
        self._on_srt_generated = on_srt_generated
        self._file_list: list = []
        self._transcribing = False
        self._stop_requested = False

        self._create_ui()
        self._load_settings()
        self._populate_models()

    def _create_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        self._create_settings_section()
        self._create_file_section()
        self._create_control_section()
        self._create_progress_section()
        self._init_log_widget(self, row=4, column=0)

    def _create_settings_section(self):
        frame = ttk.LabelFrame(self, text="Settings", padding="5")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="whisper-cli:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5))
        self._cli_var = tk.StringVar()
        self._cli_entry = ttk.Entry(frame, textvariable=self._cli_var)
        self._cli_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 2))
        ttk.Button(frame, text="Browse",
                   command=self._browse_cli).grid(row=0, column=2)

        ttk.Label(frame, text="Model:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self._model_var = tk.StringVar()
        self._model_combo = ttk.Combobox(frame, textvariable=self._model_var,
                                          state="readonly", width=30)
        self._model_combo.grid(row=1, column=1, sticky=tk.W, padx=(0, 2), pady=(5, 0))

        ttk.Label(frame, text="Model Dir:").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self._model_dir_var = tk.StringVar()
        self._model_dir_entry = ttk.Entry(frame, textvariable=self._model_dir_var)
        self._model_dir_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(0, 2), pady=(5, 0))
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=2, pady=(5, 0))
        ttk.Button(btn_frame, text="Browse",
                   command=self._browse_model_dir).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Scan",
                   command=self._scan_models).pack(side=tk.LEFT, padx=2)

        row3 = ttk.Frame(frame)
        row3.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Label(row3, text="Language:").pack(side=tk.LEFT, padx=(0, 2))
        self._lang_var = tk.StringVar(value="auto")
        self._lang_combo = ttk.Combobox(row3, textvariable=self._lang_var,
                                         state="readonly", width=20)
        self._lang_combo.pack(side=tk.LEFT, padx=(0, 20))

        self._populate_languages()

        ttk.Label(row3, text="Threads:").pack(side=tk.LEFT, padx=(0, 2))
        self._threads_var = tk.IntVar(value=8)
        self._threads_scale = ttk.Scale(row3, from_=1, to=16,
                                         variable=self._threads_var,
                                         orient=tk.HORIZONTAL, length=120)
        self._threads_scale.pack(side=tk.LEFT, padx=(0, 5))
        self._threads_label = ttk.Label(row3, text="8")
        self._threads_label.pack(side=tk.LEFT)
        self._threads_scale.configure(
            command=lambda v: self._threads_label.config(text=str(int(float(v)))))

    def _create_file_section(self):
        frame = ttk.LabelFrame(self, text="File Selection", padding="5")
        frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        frame.columnconfigure(0, weight=1)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=0, column=0, sticky=tk.W)

        ttk.Button(btn_frame, text="Choose Files",
                   command=self._choose_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear",
                   command=self._clear_files).pack(side=tk.LEFT, padx=2)

        list_frame = ttk.Frame(frame)
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        list_frame.columnconfigure(0, weight=1)

        self._file_listbox = tk.Listbox(list_frame, height=4,
                                         selectmode=tk.EXTENDED)
        self._file_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._file_listbox.drop_target_register(DND_FILES)
        self._file_listbox.dnd_bind('<<Drop>>', self._on_files_dropped)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                  command=self._file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._file_listbox.config(yscrollcommand=scrollbar.set)

    def _create_control_section(self):
        frame = ttk.Frame(self)
        frame.grid(row=2, column=0, pady=(0, 5), sticky=tk.W)

        self._start_btn = ttk.Button(frame, text="Start Transcription",
                                      command=self._start_transcription)
        self._start_btn.pack(side=tk.LEFT, padx=2)

        self._stop_btn = ttk.Button(frame, text="Stop",
                                     command=self._stop_transcription,
                                     state="disabled")
        self._stop_btn.pack(side=tk.LEFT, padx=2)

    def _create_progress_section(self):
        frame = ttk.Frame(self)
        frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        frame.columnconfigure(0, weight=1)

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            frame, variable=self._progress_var, maximum=100)
        self._progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self._progress_label = ttk.Label(frame, text="Ready")
        self._progress_label.grid(row=1, column=0, sticky=tk.W)

    def _populate_languages(self):
        codes = list(WHISPER_LANGUAGES.keys())
        names = [f"{code} - {WHISPER_LANGUAGES[code]}" for code in codes]
        self._lang_combo['values'] = names

        saved_lang = self._config_manager.get("whisper.language", "auto") if self._config_manager else "auto"
        for i, name in enumerate(names):
            if name.startswith(saved_lang):
                self._lang_combo.current(i)
                break
        else:
            self._lang_combo.current(0)

        self._lang_combo.bind('<<ComboboxSelected>>', self._on_setting_changed)

    def _populate_models(self):
        models = self._get_whisper_models()
        names = [m.get('name', str(m)) for m in models]
        self._model_combo['values'] = names

        last_model = self._config_manager.get("whisper.last_model", "") if self._config_manager else ""
        if last_model:
            for i, name in enumerate(names):
                if name == last_model:
                    self._model_combo.current(i)
                    break
        elif names:
            self._model_combo.current(0)

        self._model_combo.bind('<<ComboboxSelected>>', self._on_setting_changed)

    def _load_settings(self):
        cm = self._config_manager
        if not cm:
            return

        self._cli_var.set(cm.get("whisper.cli_path", ""))
        self._model_dir_var.set(cm.get("whisper.model_dir", ""))
        self._threads_var.set(cm.get("whisper.threads", 8))
        self._threads_label.config(text=str(self._threads_var.get()))

    def _save_settings(self):
        cm = self._config_manager
        if not cm:
            return

        cm.set("whisper.cli_path", self._cli_var.get())
        cm.set("whisper.model_dir", self._model_dir_var.get())
        cm.set("whisper.last_model", self._model_var.get())
        cm.set("whisper.language", self._get_language_code())
        cm.set("whisper.threads", int(float(self._threads_var.get())))

    def _on_setting_changed(self, event=None):
        self._save_settings()

    def _get_language_code(self):
        val = self._lang_var.get()
        return val.split(' - ')[0] if ' - ' in val else val

    def _browse_cli(self):
        path = filedialog.askopenfilename(
            title="Select whisper-cli executable",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if path:
            self._cli_var.set(path)
            self._save_settings()

    def _browse_model_dir(self):
        path = filedialog.askdirectory(title="Select model directory")
        if path:
            self._model_dir_var.set(path)
            self._save_settings()

    def _scan_models(self):
        self._save_settings()
        models = self._scan_whisper_models()
        names = [m.get('name', str(m)) for m in models]
        self._model_combo['values'] = names
        if names:
            self._model_combo.current(0)
        self._log("INFO", f"Found {len(names)} model(s)")

    def _choose_files(self):
        files = filedialog.askopenfilenames(
            title="Select audio/video files",
            filetypes=[("Audio/Video", "*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac;"
                                         "*.mp4;*.mkv;*.avi;*.mov;*.wmv;*.webm;*.ts"),
                       ("All files", "*.*")])
        for f in files:
            if f not in self._file_list and self._is_supported_file(f):
                self._file_list.append(f)
                self._file_listbox.insert(tk.END, Path(f).name)
        self._log("INFO", f"Added {len(files)} file(s)")

    def _clear_files(self):
        self._file_list.clear()
        self._file_listbox.delete(0, tk.END)

    def _on_files_dropped(self, event):
        raw_data = event.data
        paths = parse_dropped_paths(raw_data)
        added = 0
        for path in paths:
            path = path.strip()
            if not path:
                continue
            if path not in self._file_list and self._is_supported_file(path):
                self._file_list.append(path)
                self._file_listbox.insert(tk.END, Path(path).name)
                added += 1
        self._log("INFO", f"Added {added} file(s) via drag-and-drop")

    def _is_supported_file(self, filepath: str) -> bool:
        ext = Path(filepath).suffix.lower()
        return ext in SUPPORTED_MEDIA

    def _start_transcription(self):
        if not self._file_list:
            messagebox.showwarning("Warning", "Please select files first!")
            return

        cli_path = self._cli_var.get().strip()
        if not cli_path:
            messagebox.showwarning("Warning", "Please set whisper-cli path!")
            return

        model = self._model_var.get()
        if not model:
            messagebox.showwarning("Warning", "Please select a model!")
            return

        if self._transcribing:
            return

        self._transcribing = True
        self._stop_requested = False
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._progress_var.set(0)
        self._clear_log()

        self._save_settings()

        thread = threading.Thread(target=self._run_transcription, daemon=True)
        thread.start()

    def _stop_transcription(self):
        self._stop_requested = True
        self._log("WARNING", "Stopping after current file...")

    def _run_transcription(self):
        total = len(self._file_list)

        for i, filepath in enumerate(self._file_list):
            if self._stop_requested:
                self._log("WARNING", "Transcription stopped by user")
                break

            self.winfo_toplevel().after(0, lambda f=Path(filepath).name, idx=i:
                self._progress_label.config(text=f"Processing: {f} ({idx + 1}/{total})"))

            try:
                srt_path = self._transcribe_file(filepath)
                if srt_path:
                    self._log("SUCCESS", f"Generated: {srt_path}")
                    if self._on_srt_generated:
                        self._on_srt_generated(srt_path)
            except Exception as e:
                self._log("ERROR", f"Failed: {Path(filepath).name} - {e}")

            pct = ((i + 1) / total) * 100
            self.winfo_toplevel().after(0, lambda p=pct: self._progress_var.set(p))

        self.winfo_toplevel().after(0, self._on_transcription_done)

    def _transcribe_file(self, filepath):
        completed = threading.Event()
        result = {'srt_path': None, 'error': None}

        def on_complete(srt_path):
            result['srt_path'] = srt_path
            completed.set()

        def on_error(msg):
            result['error'] = msg
            completed.set()

        controller = WhisperController(
            cli_path=Path(self._cli_var.get().strip()),
            on_log=lambda msg: self._log("INFO", msg),
            on_progress=lambda msg: self._log("INFO", msg),
            on_complete=on_complete,
            on_error=on_error
        )

        model_dir = self._model_dir_var.get().strip()
        model_name = self._model_var.get()
        model_path = self._resolve_model_path(model_dir, model_name)
        language = self._get_language_code()
        threads = int(float(self._threads_var.get()))

        self._log("INFO", f"CLI: {self._cli_var.get().strip()}")
        self._log("INFO", f"Model: {model_path}")
        self._log("INFO", f"File: {filepath}")

        controller.start(filepath, model_path, language, threads)

        while not completed.is_set():
            completed.wait(timeout=0.1)
            if self._stop_requested:
                controller.stop()
                return None

        if result['error']:
            raise RuntimeError(result['error'])

        return result['srt_path']

    def _resolve_model_path(self, model_dir, model_name):
        if not model_dir or not model_name:
            return model_name
        models = self._get_whisper_models()
        for m in models:
            if m.get('name') == model_name:
                return m.get('path', model_name)
        return str(Path(model_dir) / model_name)

    def _on_transcription_done(self):
        self._transcribing = False
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._progress_label.config(text="Completed")

        if self._stop_requested:
            self._log("INFO", "Transcription stopped")
        else:
            self._log("SUCCESS", "All files transcribed!")
