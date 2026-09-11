"""PipelineCard - UI card for the Whisper -> Translate pipeline."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path

from tkinterdnd2 import DND_FILES

from base_registry import BaseModelRegistry
from pipeline_runner import PipelineRunner
from constants import (
    SUPPORTED_MEDIA, SUPPORTED_SUBTITLE, WHISPER_LANGUAGES, TARGET_LANGUAGES,
)
from ui_helpers import populate_language_combo, extract_combo_code
from config_helpers import CONTEXT_MODE_LABELS, context_mode_from_label
from file_listbox import FileListbox


class PipelineCard(ttk.LabelFrame):
    def __init__(self, parent, config_manager, get_server_running,
                 get_port, get_current_model, get_whisper_models,
                 on_log, on_debug_log=None):
        super().__init__(parent, text="🎤📝 Quick Whisper → Translate", padding="10")
        self._config = config_manager
        self._get_server_running = get_server_running
        self._get_port = get_port
        self._get_current_model = get_current_model
        self._get_whisper_models = get_whisper_models
        self._on_log = on_log
        self._on_debug_log = on_debug_log or (lambda m: None)

        self._pipeline_runner = None
        self._pipe_files = []
        self._completed_files = []

        self._create_ui()

    # ---------------------------------------------------------------- UI
    def _create_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(6, weight=1)

        lists_frame = ttk.Frame(self)
        lists_frame.grid(
            row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        lists_frame.columnconfigure(0, weight=1)
        lists_frame.columnconfigure(1, weight=1)

        pending_frame = ttk.LabelFrame(lists_frame, text="Pending", padding="5")
        pending_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 4))
        pending_frame.columnconfigure(0, weight=1)

        # ponytail: subtitle+media union — pipeline accepts both kinds.
        valid = SUPPORTED_SUBTITLE | SUPPORTED_MEDIA
        self._file_listbox_widget = FileListbox(
            pending_frame, valid_extensions=valid,
            filetypes_label="Subtitle + Media",
            filetypes_exts=[".srt", ".txt", *sorted(SUPPORTED_MEDIA)],
            on_change=self._sync_pipe_files,
            on_clear=self._clear_pipeline_files,
        )
        self._file_listbox_widget.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self._pipe_listbox = self._file_listbox_widget.listbox

        # ponytail: extra control rides along the widget's btn row.
        self._pipe_replace_var = tk.BooleanVar(
            value=self._config.get("pipeline.replace_original", False))
        ttk.Checkbutton(self._file_listbox_widget.button_row,
                        text="Replace original",
                        variable=self._pipe_replace_var).pack(side=tk.LEFT, padx=10)

        completed_frame = ttk.LabelFrame(
            lists_frame, text="Completed", padding="5")
        completed_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(4, 0))
        completed_frame.columnconfigure(0, weight=1)
        self._completed_listbox = tk.Listbox(
            completed_frame, height=4, selectmode=tk.EXTENDED)
        self._completed_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        completed_scrollbar = ttk.Scrollbar(
            completed_frame, orient=tk.VERTICAL,
            command=self._completed_listbox.yview)
        completed_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._completed_listbox.config(yscrollcommand=completed_scrollbar.set)

        ttk.Label(self, text="Whisper Model:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self._pipe_wmodel_var = tk.StringVar()
        self._pipe_wmodel_combo = ttk.Combobox(self, textvariable=self._pipe_wmodel_var,
                                                state="readonly", width=25)
        self._pipe_wmodel_combo.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))

        row3 = ttk.Frame(self)
        row3.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Label(row3, text="Language:").pack(side=tk.LEFT, padx=(0, 2))
        saved_lang = self._config.get("pipeline.language", "auto")
        self._pipe_lang_var = tk.StringVar(value=saved_lang)
        lang_combo = ttk.Combobox(row3, textvariable=self._pipe_lang_var,
                                   state="readonly", width=12)
        lang_combo.pack(side=tk.LEFT, padx=(0, 10))
        populate_language_combo(lang_combo, WHISPER_LANGUAGES, saved_lang)

        ttk.Label(row3, text="Target:").pack(side=tk.LEFT, padx=(0, 2))
        saved_target = self._config.get("pipeline.target_lang", "zh-cn")
        self._pipe_target_var = tk.StringVar(value=saved_target)
        target_combo = ttk.Combobox(row3, textvariable=self._pipe_target_var,
                                     state="readonly", width=20)
        target_combo.pack(side=tk.LEFT)
        populate_language_combo(target_combo, TARGET_LANGUAGES, saved_target)

        mode_row = ttk.Frame(self)
        mode_row.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        ttk.Label(mode_row, text="翻譯模式:").pack(side=tk.LEFT, padx=(0, 2))
        saved_context_mode = self._config.get("ui.context_mode", "none")
        self._pipe_context_mode_var = tk.StringVar(
            value=CONTEXT_MODE_LABELS.get(saved_context_mode, CONTEXT_MODE_LABELS['none'])
        )
        context_combo = ttk.Combobox(
            mode_row, textvariable=self._pipe_context_mode_var,
            values=list(CONTEXT_MODE_LABELS.values()), state="readonly", width=16,
        )
        context_combo.pack(side=tk.LEFT)

        self._pipe_status_label = ttk.Label(self, text="Ready", foreground="gray")
        self._pipe_status_label.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=(8, 0))
        self._pipe_start_btn = ttk.Button(btn_frame, text="▶ Start",
                                           command=self._start_pipeline)
        self._pipe_start_btn.pack(side=tk.LEFT, padx=2)
        self._pipe_stop_btn = ttk.Button(btn_frame, text="⏹ Stop",
                                          command=self._stop_pipeline, state="disabled")
        self._pipe_stop_btn.pack(side=tk.LEFT, padx=2)

    # ------------------------------------------------------- Model population
    def populate_models(self):
        models = self._get_whisper_models()
        names = [m.get('name', str(m)) for m in models]
        self._pipe_wmodel_combo['values'] = names
        last = self._config.get("whisper.last_model", "")
        if last and last in names:
            self._pipe_wmodel_var.set(last)
        elif names:
            self._pipe_wmodel_combo.current(0)

    # ------------------------------------------------------- File management
    def _sync_pipe_files(self):
        """Mirror the Pending list so _run_pipeline reads it."""
        self._pipe_files = self._file_listbox_widget.files

    def _clear_pipeline_files(self):
        """Clear both Pending and Completed with the shared Clear button."""
        self._file_listbox_widget.clear()
        self._completed_files.clear()
        self._completed_listbox.delete(0, tk.END)

    def _pipeline_file_completed(self, filepath):
        self.winfo_toplevel().after(
            0, lambda: self._move_pipeline_file_to_completed(filepath))

    def _move_pipeline_file_to_completed(self, filepath):
        if not self._file_listbox_widget.remove(filepath):
            return
        if filepath not in self._completed_files:
            self._completed_files.append(filepath)
            self._completed_listbox.insert(tk.END, Path(filepath).name)

    # ------------------------------------------------------- Pipeline control
    def _start_pipeline(self):
        if self._pipeline_runner and self._pipeline_runner.running:
            return
        if not self._get_server_running():
            messagebox.showwarning("Warning", "Start the llama server first!")
            return
        if not self._pipe_files:
            messagebox.showwarning("Warning", "Select files first!")
            return

        has_media = any(Path(f).suffix.lower() in SUPPORTED_MEDIA for f in self._pipe_files)
        if has_media:
            cli_path = self._config.get("whisper.cli_path", "")
            if not cli_path:
                messagebox.showwarning("Warning", "Set whisper-cli path in Speech Recognition tab first!")
                return
            if not self._pipe_wmodel_var.get():
                messagebox.showwarning("Warning", "Select a whisper model!")
                return

        self._pipe_start_btn.config(state="disabled")
        self._pipe_stop_btn.config(state="normal")
        self._pipe_status_label.config(text="Checking llama-server...", foreground="blue")

        self._pipeline_runner = PipelineRunner(
            config_manager=self._config,
            get_port=self._get_port,
            get_current_model=self._get_current_model,
            resolve_whisper_model_path=self._resolve_whisper_model_path,
            get_whisper_models=self._get_whisper_models,
            on_log=lambda msg: self._on_log("INFO", msg),
            on_progress=self._pipeline_step,
            on_done=self._pipeline_done,
            on_file_completed=self._pipeline_file_completed,
        )

        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _stop_pipeline(self):
        if self._pipeline_runner:
            self._pipeline_runner.stop()
        self._pipeline_step("Stopping...")

    def _run_pipeline(self):
        files = list(self._pipe_files)
        cli_path = Path(self._config.get("whisper.cli_path", ""))
        model_name = self._pipe_wmodel_var.get()
        model_dir = self._config.get("whisper.model_dir", "")
        lang_val = self._pipe_lang_var.get()
        language = extract_combo_code(lang_val)
        target_val = self._pipe_target_var.get()
        target_lang = extract_combo_code(target_val)
        context_var = getattr(self, '_pipe_context_mode_var', None)
        context_mode = context_mode_from_label(
            context_var.get() if context_var else self._config.get('ui.context_mode', 'none')
        )

        self._config.set("pipeline.language", language)
        self._config.set("pipeline.target_lang", target_lang)
        self._config.set("pipeline.replace_original", self._pipe_replace_var.get())
        self._config.set("ui.context_mode", context_mode)
        if model_name:
            self._config.set("whisper.last_model", model_name)

        self._pipeline_runner.run(
            files=files,
            target_lang=target_lang,
            language=language,
            replace_original=self._pipe_replace_var.get(),
            whisper_cli_path=cli_path,
            whisper_model_name=model_name,
            whisper_model_dir=model_dir,
            context_mode=context_mode,
        )

    def _pipeline_step(self, msg):
        color = "red" if msg.startswith("Error") else "blue"
        self.winfo_toplevel().after(0, lambda: self._pipe_status_label.config(
            text=msg, foreground=color))
        # One emission is enough: _on_log publishes on CHANNEL_PIPELINE and the
        # debug window subscribes to all channels. Re-emitting through the legacy
        # debug callback duplicates every line in Debug and, because that path
        # defaults to CHANNEL_APP, also leaks pipeline progress into Server.
        self._on_log("INFO", msg)

    def _pipeline_done(self, stopped=False):
        self.winfo_toplevel().after(0, lambda: self._update_pipeline_done(stopped))

    def _update_pipeline_done(self, stopped):
        self._pipe_start_btn.config(state="normal")
        self._pipe_stop_btn.config(state="disabled")
        if self._pipeline_runner.startup_error:
            self._pipe_status_label.config(
                text=f"Cannot start: {self._pipeline_runner.startup_error}",
                foreground="red")
        elif stopped:
            self._pipe_status_label.config(text="Stopped", foreground="orange")
        elif self._pipeline_runner.failed_files:
            self._pipe_status_label.config(
                text=f"{self._pipeline_runner.failed_files} file(s) failed — see log",
                foreground="red")
        else:
            self._pipe_status_label.config(text="All done!", foreground="green")

    def _resolve_whisper_model_path(self, model_dir, model_name):
        return BaseModelRegistry.resolve_model_path(
            self._get_whisper_models(), model_dir, model_name)
