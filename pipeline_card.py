"""PipelineCard - the Whisper -> Translate pipeline, the Workbench's main control.

Presentation note: this is a plain frame rather than a titled LabelFrame so the
Workbench surface owns the heading. It is the one authoritative place to start a
transcribe-then-translate run; the Subtitle and Whisper surfaces handle the
single-stage cases.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from pathlib import Path

from base_registry import BaseModelRegistry
from pipeline_runner import PipelineRunner
from constants import (
    SUPPORTED_MEDIA, SUPPORTED_SUBTITLE, WHISPER_LANGUAGES, TARGET_LANGUAGES,
)
from llm_target import current_mode, resolve_llm_target
from ui_helpers import (
    CHANNEL_PIPELINE, populate_language_combo, extract_combo_code, run_bus,
)
from config_helpers import CONTEXT_MODE_LABELS, context_mode_from_label, build_translation_config
from file_listbox import FileListbox
from ui_kit import ActionBar, FieldGrid, StatusLine
from ui_theme import theme as _theme
from ui_tokens import SPACE


class PipelineCard(ttk.Frame):
    def __init__(self, parent, config_manager, get_server_running,
                 get_port, get_current_model, get_whisper_models,
                 on_log, on_debug_log=None):
        super().__init__(parent, style="Surface.TFrame")
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
        """Queue on the left, settings on the right, controls along the bottom.

        The file queue is the thing the operator interacts with most (drag in a
        folder, watch it drain), so it gets the larger half and the vertical
        space. Settings are read once per batch and then left alone.
        """
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        self._create_queue(self)
        self._create_settings(self)
        self._create_controls(self)

    def _create_queue(self, parent):
        queue_frame = ttk.Frame(parent, style="Surface.TFrame")
        queue_frame.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE["loose"]))
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(1, weight=1)
        queue_frame.rowconfigure(3, weight=1)

        header = ttk.Frame(queue_frame, style="Surface.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="待處理", style="Field.TLabel").grid(
            row=0, column=0, sticky="w")
        self._pending_count = ttk.Label(header, text="0", style="MonoMuted.TLabel")
        self._pending_count.grid(row=0, column=1, sticky="e")

        # Subtitle + media union: the pipeline accepts an SRT to translate or
        # media to transcribe first, and decides per file.
        valid = SUPPORTED_SUBTITLE | SUPPORTED_MEDIA
        self._file_listbox_widget = FileListbox(
            queue_frame, valid_extensions=valid,
            filetypes_label="字幕與影音",
            filetypes_exts=[".srt", ".txt", *sorted(SUPPORTED_MEDIA)],
            browse_text="選擇檔案", clear_text="清空",
            on_change=self._sync_pipe_files,
            on_clear=self._clear_pipeline_files,
            empty_hint="拖放影音或字幕檔到這裡",
            height=7,
        )
        self._file_listbox_widget.grid(row=1, column=0, sticky="nsew",
                                       pady=(SPACE["tight"], SPACE["base"]))
        self._pipe_listbox = self._file_listbox_widget.listbox

        self._pipe_replace_var = tk.BooleanVar(
            value=self._config.get("pipeline.replace_original", False))
        ttk.Checkbutton(self._file_listbox_widget.button_row,
                        text="覆寫原檔",
                        variable=self._pipe_replace_var).pack(
            side=tk.LEFT, padx=(SPACE["base"], 0))

        done_header = ttk.Frame(queue_frame, style="Surface.TFrame")
        done_header.grid(row=2, column=0, sticky="ew")
        done_header.columnconfigure(0, weight=1)
        ttk.Label(done_header, text="已完成", style="Field.TLabel").grid(
            row=0, column=0, sticky="w")
        self._completed_count = ttk.Label(done_header, text="0",
                                          style="MonoMuted.TLabel")
        self._completed_count.grid(row=0, column=1, sticky="e")

        done_frame = ttk.Frame(queue_frame, style="Surface.TFrame")
        done_frame.grid(row=3, column=0, sticky="nsew", pady=(SPACE["tight"], 0))
        done_frame.columnconfigure(0, weight=1)
        done_frame.rowconfigure(0, weight=1)
        self._completed_listbox = tk.Listbox(done_frame, height=4,
                                             selectmode=tk.EXTENDED,
                                             exportselection=False)
        _theme().register_classic(self._completed_listbox, "listbox")
        self._completed_listbox.grid(row=0, column=0, sticky="nsew")
        completed_scrollbar = ttk.Scrollbar(
            done_frame, orient=tk.VERTICAL,
            command=self._completed_listbox.yview)
        completed_scrollbar.grid(row=0, column=1, sticky="ns")
        self._completed_listbox.config(yscrollcommand=completed_scrollbar.set)

    def _create_settings(self, parent):
        settings = ttk.Frame(parent, style="Surface.TFrame")
        settings.grid(row=0, column=1, sticky="nsew")
        settings.columnconfigure(0, weight=1)

        grid = FieldGrid(settings, label_width=9)
        grid.grid(row=0, column=0, sticky="ew")

        self._pipe_wmodel_var = tk.StringVar()
        self._pipe_wmodel_combo = ttk.Combobox(
            grid.add("轉錄模型"), textvariable=self._pipe_wmodel_var,
            state="readonly")
        self._pipe_wmodel_combo.pack(fill="x")

        saved_lang = self._config.get("pipeline.language", "auto")
        self._pipe_lang_var = tk.StringVar(value=saved_lang)
        lang_combo = ttk.Combobox(grid.add("來源語言"),
                                  textvariable=self._pipe_lang_var,
                                  state="readonly")
        lang_combo.pack(fill="x")
        populate_language_combo(lang_combo, WHISPER_LANGUAGES, saved_lang)

        saved_target = self._config.get("pipeline.target_lang", "zh-cn")
        self._pipe_target_var = tk.StringVar(value=saved_target)
        target_combo = ttk.Combobox(grid.add("目標語言"),
                                    textvariable=self._pipe_target_var,
                                    state="readonly")
        target_combo.pack(fill="x")
        populate_language_combo(target_combo, TARGET_LANGUAGES, saved_target)

        saved_context_mode = self._config.get("ui.context_mode", "none")
        self._pipe_context_mode_var = tk.StringVar(
            value=CONTEXT_MODE_LABELS.get(saved_context_mode,
                                          CONTEXT_MODE_LABELS['none'])
        )
        self._pipe_context_mode_combo = ttk.Combobox(
            grid.add("翻譯模式", hint="全文理解翻譯會先讀完整部字幕再翻，較慢但前後一致。"),
            textvariable=self._pipe_context_mode_var,
            values=list(CONTEXT_MODE_LABELS.values()), state="readonly")
        self._pipe_context_mode_combo.pack(fill="x")

        # Read-only mirror of the Subtitle surface's selector: this card
        # consumes whichever LLM backend was chosen there and deliberately
        # does not offer a second switch for the same setting.
        self._llm_label = ttk.Label(settings, text="", style="Micro.TLabel",
                                    wraplength=260, justify="left")
        self._llm_label.grid(row=1, column=0, sticky="w", pady=(SPACE["snug"], 0))

    def _create_controls(self, parent):
        controls = ttk.Frame(parent, style="Surface.TFrame")
        controls.grid(row=1, column=0, columnspan=2, sticky="ew",
                      pady=(SPACE["loose"], 0))
        controls.columnconfigure(0, weight=1)

        self._pipe_status = StatusLine(controls, text="就緒", state="idle")
        self._pipe_status.grid(row=0, column=0, sticky="ew",
                               pady=(0, SPACE["base"]))

        bar = ActionBar(controls)
        bar.grid(row=1, column=0, sticky="ew")
        self._pipe_start_btn = bar.primary("開始處理", self._start_pipeline)
        self._pipe_stop_btn = bar.danger("停止", self._stop_pipeline)
        self._pipe_stop_btn.state(("disabled",))

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
        self._refresh_llm_label()

    def _refresh_llm_label(self):
        """Show which LLM backend a run will use (remote name, or local)."""
        try:
            target = resolve_llm_target(
                self._config, self._get_port, self._get_current_model)
            if target.mode == 'remote':
                self._llm_label.config(
                    text=f"翻譯後端：{target.name} · {target.model or '(未設定模型)'}"
                         "（在「字幕翻譯」切換）")
            else:
                self._llm_label.config(
                    text="翻譯後端：本地 llama-server（在「字幕翻譯」切換）")
        except Exception:
            self._llm_label.config(text="")

    # ------------------------------------------------------- File management
    def _sync_pipe_files(self):
        """Mirror the Pending list so _run_pipeline reads it."""
        self._pipe_files = self._file_listbox_widget.files
        self._refresh_counts()

    def _clear_pipeline_files(self):
        """Clear both Pending and Completed with the shared Clear button."""
        self._file_listbox_widget.clear()
        self._completed_files.clear()
        self._completed_listbox.delete(0, tk.END)
        self._refresh_counts()

    def _refresh_counts(self):
        self._pending_count.config(text=str(len(self._pipe_files)))
        self._completed_count.config(text=str(len(self._completed_files)))

    def _pipeline_file_completed(self, filepath):
        self.winfo_toplevel().after(
            0, lambda: self._move_pipeline_file_to_completed(filepath))

    def _move_pipeline_file_to_completed(self, filepath):
        if not self._file_listbox_widget.remove(filepath):
            return
        if filepath not in self._completed_files:
            self._completed_files.append(filepath)
            self._completed_listbox.insert(tk.END, Path(filepath).name)
        self._refresh_counts()

    # ------------------------------------------------------- Pipeline control
    def _start_pipeline(self):
        if self._pipeline_runner and self._pipeline_runner.running:
            return
        # The local llama-server is part of the run only in local mode; a
        # remote profile needs no process to be up.
        if current_mode(self._config) == 'local' and not self._get_server_running():
            messagebox.showwarning("Warning", "Start the llama server first!")
            return
        if not self._pipe_files:
            messagebox.showwarning("沒有檔案", "請先選擇要處理的檔案。")
            return

        has_media = any(Path(f).suffix.lower() in SUPPORTED_MEDIA for f in self._pipe_files)
        if has_media:
            cli_path = self._config.get("whisper.cli_path", "")
            if not cli_path:
                messagebox.showwarning(
                    "尚未設定 whisper-cli",
                    "佇列中有影音檔，請先在「語音轉錄」設定 whisper-cli 路徑。")
                return
            if not self._pipe_wmodel_var.get():
                messagebox.showwarning("尚未選擇模型", "請選擇一個轉錄模型。")
                return

        self._pipe_start_btn.state(("disabled",))
        self._pipe_stop_btn.state(("!disabled",))
        checking = ("檢查遠端 LLM 連線…" if current_mode(self._config) == 'remote'
                    else "檢查本地 llama-server…")
        self._pipe_status.set("busy", checking)
        run_bus.begin(CHANNEL_PIPELINE, "管線啟動中", checking)

        # Read every Tk variable here, on the UI thread, and freeze it for
        # the run: the worker thread never touches a mutable Tk variable.
        run = {
            'files': list(self._pipe_files),
            'cli_path': Path(self._config.get("whisper.cli_path", "")),
            'model_name': self._pipe_wmodel_var.get(),
            'model_dir': self._config.get("whisper.model_dir", ""),
            'language': extract_combo_code(self._pipe_lang_var.get()),
            'target_lang': extract_combo_code(self._pipe_target_var.get()),
            'replace_original': self._pipe_replace_var.get(),
            'context_mode': context_mode_from_label(
                self._pipe_context_mode_var.get()
                if hasattr(self, '_pipe_context_mode_var')
                else self._config.get('ui.context_mode', 'none')
            ),
        }

        self._config.set("pipeline.language", run['language'])
        self._config.set("pipeline.target_lang", run['target_lang'])
        self._config.set("pipeline.replace_original", run['replace_original'])
        self._config.set("ui.context_mode", run['context_mode'])
        if run['model_name']:
            self._config.set("whisper.last_model", run['model_name'])

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

        self._pipeline_runner.local_server = getattr(self, 'local_server', None)
        self._pipeline_runner.run_target = resolve_llm_target(self._config, self._get_port, self._get_current_model)
        self._pipeline_runner.run_config = build_translation_config(self._config, self._pipeline_runner.run_target)
        threading.Thread(
            target=self._run_pipeline, args=(run,), daemon=True).start()

    def _stop_pipeline(self):
        if self._pipeline_runner:
            self._pipeline_runner.stop()
        self._pipeline_step("正在停止…（等待目前這次請求結束）")

    def _run_pipeline(self, run: dict):
        """Execute one run from the UI thread's frozen parameters."""
        self._pipeline_runner.run(
            files=run['files'],
            target_lang=run['target_lang'],
            language=run['language'],
            replace_original=run['replace_original'],
            whisper_cli_path=run['cli_path'],
            whisper_model_name=run['model_name'],
            whisper_model_dir=run['model_dir'],
            context_mode=run['context_mode'],
        )

    def _pipeline_step(self, msg):
        state = "error" if msg.startswith("Error") else "busy"
        self.winfo_toplevel().after(
            0, lambda: self._pipe_status.set(state, msg))
        # Progress has no numeric total here (the runner reports stage strings),
        # so the header shows the stage and its spinner rather than a bar that
        # would have to invent a percentage.
        run_bus.progress(CHANNEL_PIPELINE, "語音轉錄 → 翻譯", None, msg)
        # One emission is enough: _on_log publishes on CHANNEL_PIPELINE and the
        # log drawer subscribes to all channels. Re-emitting through the legacy
        # debug callback duplicates every line and, because that path defaults
        # to CHANNEL_APP, also leaks pipeline progress into the server channel.
        self._on_log("INFO", msg)

    def _pipeline_done(self, stopped=False):
        self.winfo_toplevel().after(0, lambda: self._update_pipeline_done(stopped))

    def _update_pipeline_done(self, stopped):
        self._pipe_start_btn.state(("!disabled",))
        self._pipe_stop_btn.state(("disabled",))
        if self._pipeline_runner.startup_error:
            state, message = "error", f"無法啟動：{self._pipeline_runner.startup_error}"
        elif stopped:
            state, message = "warn", "已停止"
        elif self._pipeline_runner.failed_files:
            state = "error"
            message = f"{self._pipeline_runner.failed_files} 個檔案失敗 — 詳見日誌"
        else:
            state, message = "ok", "全部完成"
        self._pipe_status.set(state, message)
        run_bus.end(CHANNEL_PIPELINE, "語音轉錄 → 翻譯", state, message)

    def _resolve_whisper_model_path(self, model_dir, model_name):
        return BaseModelRegistry.resolve_model_path(
            self._get_whisper_models(), model_dir, model_name)
