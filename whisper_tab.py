"""Whisper Tab - tkinter GUI for whisper.cpp speech recognition."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path

from whisper_transcription import (
    TranscriptionRequest,
    WhisperTranscriber,
)

from transcription_runner import TranscriptionPresenter, TranscriptionRunner

from base_registry import BaseModelRegistry
from constants import SUPPORTED_MEDIA, WHISPER_LANGUAGES
from ui_helpers import (
    CHANNEL_WHISPER, LogMixin, populate_language_combo, extract_combo_code,
    run_bus,
)
from ui_kit import ActionBar, FieldGrid, ScrollArea, Section, labelled_scale
from ui_tokens import SPACE
from file_listbox import FileListbox, scan_directory
from whisper_policy import (
    chunking_enabled,
    chunking_label,
    existing_srt_for,
    set_chunking_enabled,
    set_skip_existing_enabled,
    skip_existing_enabled,
)


class _WhisperTabPresenter(TranscriptionPresenter):
    """Marshals the run loop's presentation onto this tab's widgets.

    The loop runs on a worker thread; every widget update goes through the
    Tk thread via after(0, ...).
    """

    def __init__(self, tab):
        self._tab = tab

    def log(self, level, message):
        self._tab._log(level, message)

    def status(self, message):
        # Mirrored to the shell header: a transcription is long enough that the
        # operator will be on another surface for most of it.
        self._tab._last_status = message
        run_bus.progress(CHANNEL_WHISPER, message, self._tab._last_percent)
        self._tab._schedule_ui(
            lambda: self._tab._progress_label.config(text=message))

    def progress(self, percent):
        self._tab._last_percent = percent
        run_bus.progress(CHANNEL_WHISPER, self._tab._last_status, percent)
        self._tab._schedule_ui(lambda: self._tab._progress_var.set(percent))

    def srt_generated(self, filepath, srt_path):
        self._tab._schedule_ui(
            lambda: self._tab._on_srt_generated(srt_path))

    def finished(self, stopped):
        self._tab._schedule_ui(
            lambda: self._tab._on_transcription_done(stopped))


class WhisperTab(LogMixin, ttk.Frame):
    log_channel = CHANNEL_WHISPER

    def __init__(self, parent, config_manager, get_whisper_models,
                 scan_whisper_models, on_srt_generated, transcriber=None):
        super().__init__(parent, style="Canvas.TFrame")
        self._config_manager = config_manager
        self._get_whisper_models = get_whisper_models
        self._scan_whisper_models = scan_whisper_models
        self._on_srt_generated = on_srt_generated
        self._runner = TranscriptionRunner(
            transcriber or WhisperTranscriber(), _WhisperTabPresenter(self))
        self._file_list: list = []
        # Last values the presenter reported, so status and progress (which
        # arrive on separate callbacks) can be published to the header together.
        self._last_status = "轉錄中"
        self._last_percent = None

        self._create_ui()
        self._load_settings()
        self._populate_models()

    def _schedule_ui(self, fn):
        """Run `fn` on the Tk thread from a worker thread."""
        self.winfo_toplevel().after(0, fn)

    def _create_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        heading = ttk.Frame(self, style="Canvas.TFrame")
        heading.grid(row=0, column=0, sticky="ew",
                     padx=SPACE["page"], pady=(SPACE["section"], SPACE["loose"]))
        ttk.Label(heading, text="語音轉錄", style="Display.TLabel").pack(anchor="w")
        ttk.Label(heading,
                  text="用 whisper.cpp 把影音批次轉成 SRT；完成的字幕會送進翻譯佇列。",
                  style="CanvasMuted.TLabel").pack(anchor="w",
                                                   pady=(SPACE["hair"], 0))

        body = ScrollArea(self)
        body.grid(row=1, column=0, sticky="nsew")
        content = body.body
        content.columnconfigure(0, weight=1)

        self._create_file_section(content, row=0)
        self._create_settings_section(content, row=1)
        self._create_control_section(content, row=2)

    def _create_file_section(self, parent, row):
        """The queue leads: on this surface the operator's first act is adding files."""
        section = Section(parent, "來源影音",
                          "可整個資料夾加入（含子目錄）。")
        section.grid(row=row, column=0, sticky="ew", padx=SPACE["page"],
                     pady=(0, SPACE["loose"]))

        self._file_listbox_widget = FileListbox(
            section.body, valid_extensions=SUPPORTED_MEDIA,
            filetypes_label="影音檔",
            filetypes_exts=[".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
                            ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".ts"],
            on_change=self._sync_file_list,
            empty_hint="拖放影音檔或整個資料夾到這裡",
            height=7,
        )
        self._file_listbox_widget.grid(row=0, column=0, sticky="ew")
        self._file_listbox = self._file_listbox_widget.listbox

        ttk.Button(self._file_listbox_widget.button_row, text="加入資料夾",
                   style="Ghost.TButton",
                   command=self._browse_directory).pack(
            side=tk.LEFT, padx=(0, SPACE["tight"]))

        self._skip_exist_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self._file_listbox_widget.button_row,
                        text="略過已有 SRT 的檔案",
                        variable=self._skip_exist_var,
                        command=self._on_setting_changed).pack(
            side=tk.LEFT, padx=(SPACE["base"], 0))

    def _create_settings_section(self, parent, row):
        section = Section(parent, "whisper.cpp 設定")
        section.grid(row=row, column=0, sticky="ew", padx=SPACE["page"],
                     pady=(0, SPACE["loose"]))

        grid = FieldGrid(section.body, label_width=12)
        grid.grid(row=0, column=0, sticky="ew")

        cli_cell = grid.add("whisper-cli")
        self._cli_var = tk.StringVar()
        self._cli_entry = ttk.Entry(cli_cell, textvariable=self._cli_var,
                                    style="Mono.TEntry")
        self._cli_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(cli_cell, text="瀏覽…", style="Ghost.TButton",
                   command=self._browse_cli).pack(side="left",
                                                  padx=(SPACE["snug"], 0))

        dir_cell = grid.add("模型目錄")
        self._model_dir_var = tk.StringVar()
        self._model_dir_entry = ttk.Entry(dir_cell,
                                          textvariable=self._model_dir_var,
                                          style="Mono.TEntry")
        self._model_dir_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(dir_cell, text="瀏覽…", style="Ghost.TButton",
                   command=self._browse_model_dir).pack(
            side="left", padx=(SPACE["snug"], 0))
        ttk.Button(dir_cell, text="重新掃描", style="Ghost.TButton",
                   command=self._scan_models).pack(side="left")

        self._model_var = tk.StringVar()
        self._model_combo = ttk.Combobox(grid.add("轉錄模型"),
                                         textvariable=self._model_var,
                                         state="readonly")
        self._model_combo.pack(fill="x")

        self._lang_var = tk.StringVar(value="auto")
        self._lang_combo = ttk.Combobox(grid.add("語言"),
                                        textvariable=self._lang_var,
                                        state="readonly")
        self._lang_combo.pack(fill="x")
        self._populate_languages()

        self._threads_var = tk.IntVar(value=8)
        threads_holder, self._threads_label = labelled_scale(
            grid.add("執行緒"), self._threads_var, 1, 16, length=140)
        threads_holder.pack(anchor="w")
        self._threads_scale = threads_holder.scale

        self._chunk_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            section.body, text=chunking_label(),
            variable=self._chunk_var,
            command=self._on_setting_changed,
        ).grid(row=1, column=0, sticky="w", pady=(SPACE["snug"], 0))
        ttk.Label(section.body,
                  text="長音訊（約兩小時以上）分段轉錄可避免 context 崩壞導致整段重複。",
                  style="Micro.TLabel", wraplength=560, justify="left").grid(
            row=2, column=0, sticky="w")

    def _create_control_section(self, parent, row):
        section = Section(parent, "執行", flush=True)
        section.grid(row=row, column=0, sticky="ew", padx=SPACE["page"],
                     pady=(0, SPACE["page"]))

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            section.body, variable=self._progress_var, maximum=100,
            style="Accent.Horizontal.TProgressbar")
        self._progress_bar.grid(row=0, column=0, sticky="ew")

        self._progress_label = ttk.Label(section.body, text="就緒",
                                         style="MonoMuted.TLabel", anchor="w")
        self._progress_label.grid(row=1, column=0, sticky="ew",
                                  pady=(SPACE["snug"], SPACE["base"]))

        bar = ActionBar(section.body)
        bar.grid(row=2, column=0, sticky="ew")
        self._start_btn = bar.primary("開始轉錄", self._start_transcription)
        self._stop_btn = bar.danger("停止", self._stop_transcription)
        self._stop_btn.state(("disabled",))

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
        self._chunk_var.set(chunking_enabled(cm))
        self._skip_exist_var.set(skip_existing_enabled(cm))

    def _save_settings(self):
        cm = self._config_manager
        if not cm:
            return

        cm.set("whisper.cli_path", self._cli_var.get())
        cm.set("whisper.model_dir", self._model_dir_var.get())
        cm.set("whisper.last_model", self._model_var.get())
        cm.set("whisper.language", self._get_language_code())
        cm.set("whisper.threads", int(float(self._threads_var.get())))
        set_chunking_enabled(cm, bool(self._chunk_var.get()))
        set_skip_existing_enabled(cm, bool(self._skip_exist_var.get()))

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

    def _browse_directory(self):
        directory = filedialog.askdirectory(
            title="Select a directory (subdirectories included)")
        if not directory:
            return
        files = scan_directory(directory, SUPPORTED_MEDIA)
        skipped = 0
        if self._skip_exist_var.get():
            kept = []
            for f in files:
                if existing_srt_for(f).exists():
                    skipped += 1
                else:
                    kept.append(f)
            files = kept
        added = self._file_listbox_widget.add_many(files)
        self._log("INFO", f"Added {added} file(s) from {directory}")
        if skipped:
            self._log("INFO", f"Skipped {skipped} file(s) with existing SRT")

    def _sync_file_list(self):
        """Mirror the widget's file list so _start_transcription reads it."""
        self._file_list = self._file_listbox_widget.files

    def _start_transcription(self):
        if not self._file_list:
            messagebox.showwarning("沒有檔案", "請先選擇要轉錄的影音檔。")
            return

        cli_path = self._cli_var.get().strip()
        if not cli_path:
            messagebox.showwarning(
                "尚未設定 whisper-cli", "請先設定 whisper-cli 執行檔路徑。")
            return

        model = self._model_var.get()
        if not model:
            messagebox.showwarning("尚未選擇模型", "請選擇一個轉錄模型。")
            return

        if self._runner.running:
            return

        if self._skip_exist_var.get() and not self._transcribable_files():
            messagebox.showinfo(
                "全部已有字幕", "所選檔案都已經有對應的 SRT，沒有需要轉錄的項目。")
            return

        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._progress_var.set(0)
        self._progress_bar.configure(style="Accent.Horizontal.TProgressbar")
        self._last_percent = None
        run_bus.begin(CHANNEL_WHISPER,
                      f"轉錄 {len(self._transcribable_files())} 個檔案", "準備中…")

        self._save_settings()

        thread = threading.Thread(target=self._run_transcription, daemon=True)
        thread.start()

    def _stop_transcription(self):
        self._runner.stop()
        self._log("WARNING", "正在取消目前的轉錄…")

    def _run_transcription(self):
        files = self._transcribable_files()
        for f in self._file_list:
            if f not in files:
                self._log("INFO", f"Skipped (SRT exists): {Path(f).name}")
        self._runner.run_files(files, self._build_request)

    def _transcribable_files(self):
        """The file list for a run: without SRT-already-exists files when skipping.

        Re-checked at run time (not just at directory-scan time) so a file
        whose SRT appeared since it was added is still protected.
        """
        if not self._skip_exist_var.get():
            return self._file_list
        return [f for f in self._file_list
                if not existing_srt_for(f).exists()]

    def _build_request(self, filepath):
        """Build one TranscriptionRequest from the tab's current settings."""
        model_dir = self._model_dir_var.get().strip()
        model_name = self._model_var.get()
        model_path = BaseModelRegistry.resolve_model_path(
            self._get_whisper_models(), model_dir, model_name)
        language = self._get_language_code()
        threads = int(float(self._threads_var.get()))

        self._log("INFO", f"CLI: {self._cli_var.get().strip()}")
        self._log("INFO", f"Model: {model_path}")
        self._log("INFO", f"File: {filepath}")

        return TranscriptionRequest(
            source=Path(filepath),
            cli_path=Path(self._cli_var.get().strip()),
            model_path=Path(model_path),
            language=language,
            threads=threads,
            chunk_long_audio=bool(self._chunk_var.get()),
        )

    def _on_transcription_done(self, stopped: bool):
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

        if stopped:
            outcome, message = "warn", "已停止"
            self._log("INFO", "轉錄已停止")
        else:
            outcome, message = "ok", "全部完成"
            self._log("SUCCESS", "所有檔案轉錄完成！")
        self._progress_label.config(text=message)
        self._progress_bar.configure(
            style={"ok": "Ok.Horizontal.TProgressbar",
                   "warn": "Warn.Horizontal.TProgressbar"}[outcome])
        run_bus.end(CHANNEL_WHISPER, "語音轉錄", outcome, message)

    def on_surface_shown(self) -> None:
        """Shell hook: pick up a model directory the Server surface changed."""
        self._populate_models()
