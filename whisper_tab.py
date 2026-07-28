"""Whisper Tab - tkinter GUI for whisper.cpp speech recognition."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tkinterdnd2 import DND_FILES

from whisper_transcription import (
    CancellationToken,
    Cancelled,
    Completed,
    Failed,
    TranscriptionEvent,
    TranscriptionRequest,
    WhisperTranscriber,
    resolve_model_path,
)

from constants import SUPPORTED_MEDIA, WHISPER_LANGUAGES
from ui_helpers import (
    CHANNEL_WHISPER, LogMixin, populate_language_combo, extract_combo_code,
)
from file_listbox import FileListbox


class WhisperTab(LogMixin, ttk.Frame):
    log_channel = CHANNEL_WHISPER

    def __init__(self, parent, config_manager, get_whisper_models,
                 scan_whisper_models, on_srt_generated, transcriber=None):
        super().__init__(parent)
        self._config_manager = config_manager
        self._get_whisper_models = get_whisper_models
        self._scan_whisper_models = scan_whisper_models
        self._on_srt_generated = on_srt_generated
        self._transcriber = transcriber or WhisperTranscriber()
        self._file_list: list = []
        self._transcribing = False
        self._stop_requested = False
        self._active_cancellation: Optional[CancellationToken] = None

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

        self._chunk_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row3, text="Chunk long audio (30 min, anti-repeat)",
            variable=self._chunk_var,
            command=self._on_setting_changed,
        ).pack(side=tk.LEFT, padx=(20, 0))

    def _create_file_section(self):
        frame = ttk.LabelFrame(self, text="File Selection", padding="5")
        frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        frame.columnconfigure(0, weight=1)

        self._file_listbox_widget = FileListbox(
            frame, valid_extensions=SUPPORTED_MEDIA,
            filetypes_label="Audio/Video",
            filetypes_exts=[".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
                            ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".ts"],
            on_change=self._sync_file_list,
        )
        self._file_listbox_widget.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self._file_listbox = self._file_listbox_widget.listbox

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
        saved_lang = self._config_manager.get("whisper.language", "auto") if self._config_manager else "auto"
        populate_language_combo(self._lang_combo, WHISPER_LANGUAGES, saved_lang)
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
        self._chunk_var.set(bool(cm.get("whisper.chunk_long_audio", False)))

    def _save_settings(self):
        cm = self._config_manager
        if not cm:
            return

        cm.set("whisper.cli_path", self._cli_var.get())
        cm.set("whisper.model_dir", self._model_dir_var.get())
        cm.set("whisper.last_model", self._model_var.get())
        cm.set("whisper.language", self._get_language_code())
        cm.set("whisper.threads", int(float(self._threads_var.get())))
        cm.set("whisper.chunk_long_audio", bool(self._chunk_var.get()))

    def _on_setting_changed(self, event=None):
        self._save_settings()

    def _get_language_code(self):
        return extract_combo_code(self._lang_var.get())

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

    def _sync_file_list(self):
        """Mirror the widget's file list so _start_transcription reads it."""
        self._file_list = self._file_listbox_widget.files

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
        self._active_cancellation = CancellationToken()
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._progress_var.set(0)
        self._clear_log()

        self._save_settings()

        thread = threading.Thread(target=self._run_transcription, daemon=True)
        thread.start()

    def _stop_transcription(self):
        self._stop_requested = True
        if self._active_cancellation is not None:
            self._active_cancellation.cancel()
        self._log("WARNING", "Cancelling current transcription...")

    def _run_transcription(self):
        total = len(self._file_list)

        for i, filepath in enumerate(self._file_list):
            if self._stop_requested:
                self._log("WARNING", "Transcription stopped by user")
                break

            self.winfo_toplevel().after(0, lambda f=Path(filepath).name, idx=i:
                self._progress_label.config(text=f"Processing: {f} ({idx + 1}/{total})"))

            try:
                outcome = self._transcribe_file(filepath)
                if isinstance(outcome, Cancelled):
                    self._stop_requested = True
                    self._log("WARNING", "Transcription cancelled by user")
                    break
                if isinstance(outcome, Failed):
                    self._log(
                        "ERROR",
                        f"Failed: {Path(filepath).name} - "
                        f"{outcome.stage}: {outcome.message}",
                    )
                elif isinstance(outcome, Completed):
                    srt_path = str(outcome.srt_path)
                    self._log("SUCCESS", f"Generated: {srt_path}")
                    if self._on_srt_generated:
                        self._on_srt_generated(srt_path)
            except Exception as e:
                self._log("ERROR", f"Failed: {Path(filepath).name} - {e}")

            pct = ((i + 1) / total) * 100
            self.winfo_toplevel().after(0, lambda p=pct: self._progress_var.set(p))

        self.winfo_toplevel().after(0, self._on_transcription_done)

    def _transcribe_file(self, filepath):
        model_dir = self._model_dir_var.get().strip()
        model_name = self._model_var.get()
        model_path = self._resolve_model_path(model_dir, model_name)
        language = self._get_language_code()
        threads = int(float(self._threads_var.get()))

        self._log("INFO", f"CLI: {self._cli_var.get().strip()}")
        self._log("INFO", f"Model: {model_path}")
        self._log("INFO", f"File: {filepath}")

        cancellation = self._active_cancellation
        if cancellation is None:
            cancellation = CancellationToken()
            if self._stop_requested:
                cancellation.cancel()
            self._active_cancellation = cancellation

        return self._transcriber.transcribe(
            TranscriptionRequest(
                source=Path(filepath),
                cli_path=Path(self._cli_var.get().strip()),
                model_path=Path(model_path),
                language=language,
                threads=threads,
                chunk_long_audio=bool(self._chunk_var.get()),
            ),
            cancellation=cancellation,
            emit=self._on_transcription_event,
        )

    def _on_transcription_event(self, event: TranscriptionEvent):
        if event.kind == "diagnostic" and event.message:
            self._log("INFO", event.message)
            return
        if event.kind == "progress" and event.total:
            text = f"{event.stage}: {event.current}/{event.total}"
        elif event.kind == "started":
            text = event.stage.capitalize()
        else:
            return
        self.winfo_toplevel().after(
            0, lambda value=text: self._progress_label.config(text=value))

    def _resolve_model_path(self, model_dir, model_name):
        return resolve_model_path(
            model_dir, model_name, self._get_whisper_models)

    def _on_transcription_done(self):
        self._active_cancellation = None
        self._transcribing = False
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

        if self._stop_requested:
            self._progress_label.config(text="Stopped")
            self._log("INFO", "Transcription stopped")
        else:
            self._progress_label.config(text="Completed")
            self._log("SUCCESS", "All files transcribed!")
