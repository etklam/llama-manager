"""The Workbench: the surface that does the whole job in one pass.

Replaces the old "工作台" tab, which held two cards that each duplicated another
tab: a quick-start card reimplementing the Server tab's model and port
controls, and the pipeline card. Three surfaces could therefore start a
translation, each with a slightly different option set, and the operator had no
way to know which settings a given run would actually use.

What changed, per PRODUCT.md's third design principle:

* The pipeline is the Workbench's reason to exist, so it gets the surface.
* Server *configuration* moved out entirely. What remains is a start/stop
  switch and a readout, because starting a server is a switch rather than a job
  with options. Port, GPU layers, context size and the rest live on the Server
  surface alone, and this strip links there instead of offering a second copy.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from pipeline_card import PipelineCard
from whisper_transcription import WhisperTranscriber
from ui_kit import Section, StatusLine
from ui_tokens import SPACE


class WorkbenchSurface(ttk.Frame):
    """Server switch plus the transcribe-then-translate pipeline."""

    def __init__(self, parent, config_manager, get_server_running,
                 get_port, get_current_model, get_whisper_models,
                 on_start_server: Callable[[], None],
                 on_stop_server: Callable[[], None],
                 on_goto_server: Callable[[], None],
                 on_log, on_debug_log=None,
                 transcriber: WhisperTranscriber | None = None):
        super().__init__(parent, style="Canvas.TFrame")
        self._get_server_running = get_server_running
        self._get_port = get_port
        self._get_current_model = get_current_model
        self._on_start_server = on_start_server
        self._on_stop_server = on_stop_server
        self._on_goto_server = on_goto_server

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        heading = ttk.Frame(self, style="Canvas.TFrame")
        heading.grid(row=0, column=0, sticky="ew",
                     padx=SPACE["page"], pady=(SPACE["section"], SPACE["loose"]))
        ttk.Label(heading, text="工作台", style="Display.TLabel").pack(anchor="w")
        ttk.Label(heading,
                  text="把日文影音轉成繁體中文字幕：轉錄與翻譯一次完成。",
                  style="CanvasMuted.TLabel").pack(anchor="w",
                                                   pady=(SPACE["hair"], 0))

        self._create_server_strip()
        self._create_pipeline(config_manager, get_whisper_models,
                              on_log, on_debug_log, transcriber)

    # ------------------------------------------------------- server strip
    def _create_server_strip(self):
        section = Section(
            self, "模型伺服器",
            "翻譯需要一個執行中的 llama-server（或在「字幕翻譯」改用遠端 API）。",
            trailing=lambda parent: ttk.Button(
                parent, text="伺服器設定 →", style="Ghost.TButton",
                command=self._on_goto_server),
        )
        section.grid(row=1, column=0, sticky="ew", padx=SPACE["page"],
                     pady=(0, SPACE["loose"]))

        row = ttk.Frame(section.body, style="Surface.TFrame")
        row.grid(row=0, column=0, sticky="ew")
        row.columnconfigure(0, weight=1)

        self._server_status = StatusLine(row, text="未啟動", state="idle",
                                         mono=True)
        self._server_status.grid(row=0, column=0, sticky="w")

        buttons = ttk.Frame(row, style="Surface.TFrame")
        buttons.grid(row=0, column=1, sticky="e")
        self._start_button = ttk.Button(buttons, text="啟動伺服器",
                                        style="Accent.TButton",
                                        command=self._on_start_server)
        self._start_button.pack(side="left", padx=(0, SPACE["snug"]))
        self._stop_button = ttk.Button(buttons, text="停止", style="Danger.TButton",
                                       command=self._on_stop_server)
        self._stop_button.pack(side="left")
        self._stop_button.state(("disabled",))

    # ----------------------------------------------------------- pipeline
    def _create_pipeline(self, config_manager, get_whisper_models,
                         on_log, on_debug_log, transcriber):
        section = Section(
            self, "語音轉錄 → 翻譯",
            "影音檔會先轉錄成字幕再翻譯；已有的 SRT／TXT 會直接翻譯。")
        section.grid(row=2, column=0, sticky="nsew", padx=SPACE["page"],
                     pady=(0, SPACE["page"]))

        self.pipeline_card = PipelineCard(
            section.body,
            config_manager=config_manager,
            get_server_running=self._get_server_running,
            get_port=self._get_port,
            get_current_model=self._get_current_model,
            get_whisper_models=get_whisper_models,
            on_log=on_log,
            on_debug_log=on_debug_log,
            transcriber=transcriber,
        )
        self.pipeline_card.grid(row=0, column=0, sticky="nsew")
        section.body.rowconfigure(0, weight=1)

    # -------------------------------------------------------------- state
    def set_server_state(self, running: bool, model_name: str = "",
                         port=None) -> None:
        """Mirror the real server state; the Server surface is the owner."""
        if running:
            self._server_status.set("ok", f"執行中  :{port}  {model_name or ''}".rstrip())
            self._start_button.state(("disabled",))
            self._stop_button.state(("!disabled",))
        else:
            self._server_status.set("idle", "未啟動")
            self._start_button.state(("!disabled",))
            self._stop_button.state(("disabled",))

    def on_surface_shown(self) -> None:
        """Shell hook: refresh what a sibling surface may have changed."""
        self.pipeline_card.populate_models()
        self.set_server_state(
            self._get_server_running(), self._get_current_model(),
            self._get_port())
