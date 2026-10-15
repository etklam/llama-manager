#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llama.cpp Manager — 本地模型服務、語音轉錄與字幕翻譯的桌面控制台。

This module is now wiring only. It resolves paths, builds the five surfaces,
registers them with :class:`app_shell.AppShell`, and routes server state into
the persistent header. Everything about how the interface looks lives in
``ui_tokens`` / ``ui_theme`` / ``ui_kit``, and everything about its structure
lives in ``app_shell``.

Previously this file also owned a notebook, a quick-start card duplicating the
Server tab, and a floating debug-log window. The shell replaced all three.
"""

import tkinter as tk
from tkinter import messagebox
from pathlib import Path

from tkinterdnd2 import TkinterDnD

from app_shell import AppShell
from subtitle_tab import SubtitleTranslationTab
from whisper_tab import WhisperTab
from server_tab import ServerTab
from workbench import WorkbenchSurface
from config_manager import ConfigManager
from server_controller import ServerController
from manga_tab import MangaTab
from model_registry import ModelRegistry
from whisper_model_registry import WhisperModelRegistry
from app_paths import default_tool_paths, resource_dir, user_data_dir
from tool_manager import ToolManager
from whisper_transcription import FfmpegMediaAdapter, WhisperTranscriber
from ui_theme import apply_theme
from ui_helpers import (
    CHANNEL_APP,
    CHANNEL_MANGA,
    CHANNEL_PIPELINE,
    CHANNEL_SERVER,
    CHANNEL_TRANSLATE,
    CHANNEL_WHISPER,
    LogMixin,
    log_bus,
)

class LlamaManager(LogMixin):
    def __init__(self, root):
        self.root = root
        self.root.title("llama.cpp Manager")
        self.root.geometry("1280x860")
        self.root.minsize(1100, 720)

        self.resource_dir = resource_dir(__file__)
        self.data_dir = user_data_dir()
        config_path = self.data_dir / "config.json"
        legacy_config = self.resource_dir / "config.json"
        self.first_run = not config_path.exists() and not legacy_config.is_file()
        self.config_manager = ConfigManager(
            str(config_path),
            legacy_paths=[legacy_config],
        )
        self.config_manager.load()
        self.tools = ToolManager(self.config_manager, self.data_dir)

        # Theme before any widget exists: ttk styles have to be registered
        # before the widgets that reference them are constructed.
        self.theme = apply_theme(
            self.root, self.config_manager.get("ui.theme", "dark"))

        self._resolve_paths()

        self.server = ServerController(
            self.server_exe,
            lambda line: log_bus.emit("INFO", line, CHANNEL_SERVER),
        )
        self.models = ModelRegistry(self.model_dir, self.config_manager)

        self.whisper_models = WhisperModelRegistry(self.whisper_model_dir,
                                                   self.config_manager)
        ffmpeg = self.tools.resolve("ffmpeg")
        ffprobe = self.tools.resolve("ffprobe")
        self.transcriber = WhisperTranscriber(
            media=FfmpegMediaAdapter(ffmpeg_path=ffmpeg, ffprobe_path=ffprobe))

        self.create_ui()
        self.scan_models()

        # ServerTab reaches back for scan_models via the toplevel.
        self.root._app = self
        self.root.protocol("WM_DELETE_WINDOW", self._request_close)
        if self.first_run:
            self.root.after(250, self.server_tab._open_setup_wizard)

    # --------------------------------------------------------------- paths
    def _resolve_paths(self):
        """Resolve configured paths against the writable user data directory."""
        cm = self.config_manager
        defaults = default_tool_paths(self.data_dir)
        base = Path(cm.get("paths.base_dir", "") or defaults["base_dir"])
        runtime = Path(cm.get("paths.runtime_dir", "") or defaults["runtime_dir"])
        model_dir = Path(cm.get("paths.model_dir", "") or defaults["model_dir"])
        server_exe = self.tools.resolve("llama-server")
        whisper_model_dir = Path(
            cm.get("whisper.model_dir", "") or defaults["whisper_model_dir"])

        self.base_dir = base
        self.runtime_dir = runtime
        self.model_dir = model_dir
        self.whisper_model_dir = whisper_model_dir
        self.server_exe = server_exe

    # ------------------------------------------------------------ lifecycle
    def _request_close(self):
        # Keep Tk dispatch alive until the manga worker releases its lease and
        # finishes restoring the previously owned server after cancellation.
        self.manga_tab._closing = True
        if self.manga_tab.active:
            self.manga_tab.cancel()
            self.root.after(100, self._request_close)
            return
        self.shell.drawer.close_detached()
        self.root.destroy()

    def scan_models(self):
        if not self.model_dir.exists():
            self.log("WARNING", f"模型目錄不存在：{self.model_dir}")
            return

        models_list, added_count, removed_count = self.models.scan()

        if removed_count > 0:
            self.log("INFO", f"移除 {removed_count} 個不存在的模型")
        for m in models_list[-added_count:] if added_count > 0 else []:
            self.log("SUCCESS", f"發現新模型: {m['name']} ({m['size']})")

        self.server_tab.refresh_model_list()

        total_count = len(models_list)
        if removed_count > 0 or added_count > 0:
            self.log("INFO", f"掃描完成: 共 {total_count} 個模型 (+{added_count}, -{removed_count})")
        else:
            self.log("INFO", f"掃描完成: 共 {total_count} 個模型 (無變更)")

    # ------------------------------------------------------------------ UI
    def create_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.shell = AppShell(
            self.root, self.config_manager,
            channels=(CHANNEL_APP, CHANNEL_SERVER, CHANNEL_TRANSLATE,
                      CHANNEL_WHISPER, CHANNEL_PIPELINE, CHANNEL_MANGA),
        )
        self.shell.grid(row=0, column=0, sticky="nsew")

        parent = self.shell.content

        # Server first: the Workbench strip and the Subtitle surface both read
        # its model selection and port.
        self.server_tab = ServerTab(
            parent,
            server_controller=self.server,
            models_registry=self.models,
            config_manager=self.config_manager,
            base_dir=self.model_dir,
            on_model_selected=self._on_server_model_selected,
            on_server_state_changed=self._on_server_state_changed,
            on_model_dir_changed=self._set_model_dir,
            tool_manager=self.tools,
        )

        self.main_tab = WorkbenchSurface(
            parent,
            config_manager=self.config_manager,
            get_server_running=lambda: self.server.running,
            get_port=lambda: self.server_tab.port_var.get(),
            get_current_model=self.get_current_model,
            get_whisper_models=self.get_whisper_models,
            on_start_server=self._quick_start_server,
            on_stop_server=self._quick_stop_server,
            on_goto_server=lambda: self.shell.select("server"),
            on_log=lambda level, msg: log_bus.emit(level, msg, CHANNEL_PIPELINE),
            on_debug_log=self._debug_log,
            transcriber=self.transcriber,
        )

        self.subtitle_tab = SubtitleTranslationTab(
            parent,
            get_port_callback=lambda: self.server_tab.port_var.get(),
            get_model_callback=self.get_current_model,
            config_manager=self.config_manager,
        )

        self.whisper_tab = WhisperTab(
            parent,
            config_manager=self.config_manager,
            get_whisper_models=self.get_whisper_models,
            scan_whisper_models=lambda: self.get_whisper_models(do_scan=True),
            on_srt_generated=self._on_srt_generated,
            transcriber=self.transcriber,
            on_tools_changed=self._on_whisper_tools_changed,
            tool_manager=self.tools,
        )

        self.manga_tab = MangaTab(parent, self.server, self.config_manager)

        self.subtitle_tab.local_server = self.server
        self.pipeline_card.local_server = self.server

        # Registration order is nav order, and drives Ctrl+1..5.
        self.shell.register("workbench", self.main_tab, "工作台",
                            channels=(CHANNEL_PIPELINE,))
        self.shell.register("server", self.server_tab, "伺服器設定",
                            channels=(CHANNEL_SERVER,))
        self.shell.register("subtitle", self.subtitle_tab, "字幕翻譯",
                            channels=(CHANNEL_TRANSLATE,))
        self.shell.register("whisper", self.whisper_tab, "語音轉錄",
                            channels=(CHANNEL_WHISPER,))
        self.shell.register("manga", self.manga_tab, "漫畫 OCR／翻譯 Beta",
                            channels=(CHANNEL_MANGA,))
        self.shell.select("workbench")

    @property
    def pipeline_card(self):
        """The Workbench's pipeline, where the old layout kept it directly."""
        return self.main_tab.pipeline_card

    @property
    def notebook(self):
        """Compatibility alias for the shell's surface registry.

        The shell exposes ``select(widget)`` / ``select()`` with the same shape
        ``ttk.Notebook`` used, so existing callers that navigated through
        ``app.notebook`` keep working.
        """
        return self.shell

    # ---------------------------------------------------- Server callbacks
    def _on_server_model_selected(self, model_name):
        self.config_manager.set("ui.last_model", model_name)

    def _on_server_state_changed(self, running, model_name, port):
        self.shell.header.set_server(running, port, model_name)
        self.main_tab.set_server_state(running, model_name, port)

    # -------------------------------------------------- Quick server switch
    def _quick_start_server(self):
        model_name = self.get_current_model()
        if not model_name:
            messagebox.showerror(
                "尚未選擇模型", "請先在「伺服器設定」選擇一個模型。")
            self.shell.select("server")
            return
        if self.server.running:
            messagebox.showwarning("伺服器已在執行", "伺服器已經啟動。")
            return
        port = self.server_tab.port_var.get()
        if self.server_tab._do_start_server(model_name, port):
            self._debug_log(f"Server started: http://localhost:{port}")

    def _quick_stop_server(self):
        self.server_tab._do_stop_server()

    # ------------------------------------------------------------- logging
    def _debug_log(self, message):
        """Single-arg legacy entry point: derive a level, then emit."""
        upper = message.upper()
        if "ERROR" in upper:
            level = "ERROR"
        elif "WARNING" in upper or "WARN" in upper:
            level = "WARNING"
        else:
            level = "INFO"
        log_bus.emit(level, message)

    def log(self, level, message):
        log_bus.emit(level, message)

    # --------------------------------------------------- Log window (compat)
    def _open_debug_win(self):
        """Open the log in its own window. Kept under the old name."""
        self.shell.drawer.detach()

    def _close_debug_win(self):
        self.shell.drawer.close_detached()

    @property
    def _debug_win(self):
        return self.shell.drawer.detached

    # ------------------------------------------------------ Shared helpers
    def get_current_model(self):
        return self.server_tab.model_var.get()

    def _set_model_dir(self, path):
        """Apply the model directory selected on the Server surface."""
        value = str(path).strip()
        if not value:
            return
        self.model_dir = Path(value).expanduser()
        self.models.scan_dir = self.model_dir
        self.config_manager.set("paths.model_dir", str(self.model_dir))
        self.server_tab._base_dir = self.model_dir
        self.scan_models()

    def _on_whisper_tools_changed(self):
        """Refresh the shared media adapter after Whisper settings change."""
        ffmpeg = self.tools.resolve("ffmpeg")
        ffprobe = self.tools.resolve("ffprobe")
        self.transcriber.set_media_tools(ffmpeg, ffprobe)
        self.log("INFO", f"媒體工具：ffmpeg={ffmpeg}；ffprobe={ffprobe}")

    def _apply_setup_wizard(self, values):
        """Refresh live surfaces after the first-run setup dialog closes."""
        model_dir = values.get("model_dir", "").strip()
        if model_dir:
            self.server_tab.model_dir_var.set(model_dir)
            self.server_tab._apply_model_dir()

        server_exe = values.get("llama-server", "").strip()
        if server_exe:
            self.server_tab.server_exe_var.set(server_exe)
            self.server_tab._apply_server_exe()

        whisper = self.whisper_tab
        whisper_dir = values.get("whisper_model_dir", "").strip()
        if whisper_dir:
            whisper._model_dir_var.set(whisper_dir)
        whisper_cli = values.get("whisper-cli", "").strip()
        if whisper_cli:
            whisper._cli_var.set(whisper_cli)
        whisper._ffmpeg_var.set(values.get("ffmpeg", "").strip())
        whisper._ffprobe_var.set(values.get("ffprobe", "").strip())
        whisper._save_settings()
        whisper._populate_models()

    def get_whisper_models(self, do_scan=False):
        model_dir = self.config_manager.get("whisper.model_dir", "")
        if model_dir:
            self.whisper_models.scan_dir = Path(model_dir)
        if do_scan:
            self.whisper_models.scan()
            if self.whisper_models.last_scan_error:
                self.log("WARNING", self.whisper_models.last_scan_error)
        return self.whisper_models.list_models()

    def _on_srt_generated(self, srt_path):
        self.subtitle_tab.load_file(srt_path)
        self.log("INFO", f"字幕已載入翻譯佇列：{Path(srt_path).name}")


def main():
    root = TkinterDnD.Tk()
    LlamaManager(root)
    root.mainloop()


if __name__ == "__main__":
    main()
