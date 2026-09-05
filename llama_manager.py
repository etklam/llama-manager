#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llama.cpp Manager - 簡單的 GUI 管理器
用於管理 llama.cpp 服務器和模型
"""

import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path

from tkinterdnd2 import TkinterDnD

from subtitle_tab import SubtitleTranslationTab
from whisper_tab import WhisperTab
from server_tab import ServerTab
from pipeline_card import PipelineCard
from config_manager import ConfigManager
from server_controller import ServerController
from model_registry import ModelRegistry
from whisper_model_registry import WhisperModelRegistry
from ui_theme import apply_theme
# constants imported indirectly by pipeline_card, server_tab, etc.
from ui_helpers import (
    CHANNEL_APP,
    CHANNEL_PIPELINE,
    CHANNEL_SERVER,
    LOG_POLL_MS,
    LOG_DRAIN_MAX,
    LogBuffer,
    append_log_lines,
    LogMixin,
    log_bus,
)

class LlamaManager(LogMixin):
    def __init__(self, root):
        self.root = root
        self.root.title("llama.cpp Manager")
        self.root.geometry("1120x820")
        self.root.minsize(960, 720)
        self.root.resizable(True, True)
        self._style = apply_theme(root)

        self.base_dir = Path(r"D:\AI\llama\llama.cpp")
        self.runtime_dir = Path(
            r"D:\AI\llama\llama.cpp-dflash\build-vulkan-msvc\bin"
        )
        self.model_dir = self.base_dir / "llama-hip"
        self.server_exe = self.runtime_dir / "llama-server.exe"

        self.config_manager = ConfigManager(str(Path(__file__).parent / "config.json"))
        self.config_manager.load()

        self.server = ServerController(
            self.server_exe,
            lambda line: log_bus.emit("INFO", line, CHANNEL_SERVER),
        )
        self.models = ModelRegistry(self.model_dir, self.config_manager)

        whisper_model_dir = Path(self.config_manager.get("whisper.model_dir", str(self.model_dir)))
        self.whisper_models = WhisperModelRegistry(whisper_model_dir, self.config_manager)

        self._debug_win = None
        self._debug_text = None
        self._debug_after = None

        self.create_ui()
        self.scan_models()

        # Allow ServerTab to call back to scan_models
        self.root._app = self

    def scan_models(self):
        if not self.model_dir.exists():
            self.log("WARNING", f"目錄不存在: {self.model_dir}")
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

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Tab 0: Main
        self.main_tab = ttk.Frame(self.notebook, padding="12")
        self.notebook.add(self.main_tab, text="工作台")
        self._create_main_tab()

        # Tab 1: Server
        self.server_tab = ServerTab(
            self.notebook,
            server_controller=self.server,
            models_registry=self.models,
            config_manager=self.config_manager,
            base_dir=self.model_dir,
            on_model_selected=self._on_server_model_selected,
            on_server_state_changed=self._on_server_state_changed,
        )
        self.notebook.add(self.server_tab, text="伺服器設定")

        # Tab 2: Subtitle Translation
        self.subtitle_tab = SubtitleTranslationTab(
            self.notebook,
            get_port_callback=lambda: self.server_tab.port_var.get(),
            get_model_callback=self.get_current_model,
            config_manager=self.config_manager
        )
        self.notebook.add(self.subtitle_tab, text="字幕翻譯")

        # Tab 3: Speech Recognition
        self.whisper_tab = WhisperTab(
            self.notebook,
            config_manager=self.config_manager,
            get_whisper_models=self.get_whisper_models,
            scan_whisper_models=lambda: self.get_whisper_models(do_scan=True),
            on_srt_generated=self._on_srt_generated
        )
        self.notebook.add(self.whisper_tab, text="語音轉錄")

        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

    # ---------------------------------------------------- Server Tab callbacks
    def _on_server_model_selected(self, model_name):
        self.config_manager.set("ui.last_model", model_name)

    def _on_server_state_changed(self, running, model_name, port):
        if running:
            self._qs_start_btn.config(state="disabled")
            self._qs_stop_btn.config(state="normal")
            self._qs_status_label.config(text=f"● Running (port {port})", foreground="green")
        else:
            self._qs_start_btn.config(state="normal")
            self._qs_stop_btn.config(state="disabled")
            self._qs_status_label.config(text="● Stopped", foreground="gray")

    # --------------------------------------------------------------- Main tab
    def _create_main_tab(self):
        self.main_tab.columnconfigure(0, weight=1)
        self.main_tab.rowconfigure(1, weight=1)

        header = ttk.Frame(self.main_tab)
        header.grid(row=0, column=0, sticky=tk.EW, pady=(0, 20))
        heading = ttk.Frame(header)
        heading.pack(side=tk.LEFT)
        ttk.Label(heading, text="llama.cpp Manager",
                  style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(heading, text="啟動本地模型，將語音轉成字幕並完成翻譯。",
                  style="Muted.TLabel").pack(anchor=tk.W, pady=(6, 0))
        self._debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(header, text="診斷日誌", variable=self._debug_var,
                        command=self._toggle_debug).pack(side=tk.RIGHT, padx=(20, 0))

        cards = ttk.Frame(self.main_tab)
        cards.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)
        cards.rowconfigure(0, weight=1)

        self._create_quick_start_card(cards, 0)

        self.pipeline_card = PipelineCard(
            cards,
            config_manager=self.config_manager,
            get_server_running=lambda: self.server.running,
            get_port=lambda: self.server_tab.port_var.get(),
            get_current_model=self.get_current_model,
            get_whisper_models=self.get_whisper_models,
            # Pipeline runs emit whisper diagnostics and per-line translation
            # logs on their own channel: the card's status label is the visible
            # feedback, and the debug window has the full detail when wanted.
            on_log=lambda level, msg: log_bus.emit(level, msg, CHANNEL_PIPELINE),
            on_debug_log=self._debug_log,
        )
        self.pipeline_card.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))

    def _create_quick_start_card(self, parent, col):
        card = ttk.LabelFrame(parent, text="啟動模型伺服器", padding="12")
        card.grid(row=0, column=col, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        card.columnconfigure(1, weight=1)

        ttk.Label(card, text="模型").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self._qs_model_var = tk.StringVar()
        self._qs_model_combo = ttk.Combobox(card, textvariable=self._qs_model_var,
                                             state="readonly", width=24)
        self._qs_model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))

        ttk.Label(card, text="連接埠").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self._qs_port_var = tk.IntVar(value=self.config_manager.get("server.port", 8080))
        ttk.Entry(card, textvariable=self._qs_port_var, width=8).grid(
            row=1, column=1, sticky=tk.W, padx=(0, 5), pady=(5, 0))

        self._qs_status_label = ttk.Label(card, text="● Stopped", foreground="gray")
        self._qs_status_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        btn_frame = ttk.Frame(card)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(8, 0))
        self._qs_start_btn = ttk.Button(btn_frame, text="啟動伺服器", style="Accent.TButton",
                                         command=self._quick_start_server)
        self._qs_start_btn.pack(side=tk.LEFT, padx=2)
        self._qs_stop_btn = ttk.Button(btn_frame, text="停止",
                                        command=self._quick_stop_server, state="disabled")
        self._qs_stop_btn.pack(side=tk.LEFT, padx=2)
        ttk.Label(card, text="GPU、上下文與加速選項可在「伺服器設定」調整。",
                  style="Muted.TLabel", wraplength=310).grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=(18, 0))

    def _populate_quick_start(self):
        model_list = self.models.list_models()
        names = [m.get("name", "Unknown") for m in model_list]
        self._qs_model_combo['values'] = names
        last = self.config_manager.get("ui.last_model", "")
        if last and last in names:
            self._qs_model_var.set(last)
        elif names:
            self._qs_model_combo.current(0)
        # Sync Quick Start selection with Server tab
        if self.server_tab.model_var.get():
            self._qs_model_var.set(self.server_tab.model_var.get())

    # -------------------------------------------------- Quick Start Llama
    def _quick_start_server(self):
        model_name = self._qs_model_var.get()
        if not model_name:
            messagebox.showerror("Error", "Select a model first!")
            return
        if self.server.running:
            messagebox.showwarning("Warning", "Server already running!")
            return

        port = self._qs_port_var.get()
        # Delegate to ServerTab's _do_start_server
        if self.server_tab._do_start_server(model_name, port):
            self._debug_log(f"Server started: http://localhost:{port}")
        else:
            self._qs_status_label.config(text="● Failed", foreground="red")

    def _quick_stop_server(self):
        # Delegate to ServerTab's _do_stop_server
        self.server_tab._do_stop_server()

    # ---------------------------------------------- Debug log window
    def _toggle_debug(self):
        if self._debug_var.get():
            self._open_debug_win()
        else:
            self._close_debug_win()

    def _open_debug_win(self):
        if self._debug_win and self._debug_win.winfo_exists():
            self._debug_win.lift()
            return
        self._debug_win = tk.Toplevel(self.root)
        self._debug_win.title("Debug Log")
        self._debug_win.geometry("700x400")
        self._debug_win.protocol("WM_DELETE_WINDOW", self._on_debug_win_close)
        self._debug_text = scrolledtext.ScrolledText(
            self._debug_win, wrap=tk.WORD, font=("Consolas", 9))
        self._debug_text.pack(fill=tk.BOTH, expand=True)
        self._debug_text.tag_config("ERROR", foreground="red")
        self._debug_text.tag_config("WARNING", foreground="orange")
        # The debug window is the one sink that sees every channel, so it stays
        # the place to watch a whole run end to end.
        self._debug_queue = LogBuffer()
        self._debug_unsub = log_bus.subscribe_all(self._debug_subscriber)
        self._debug_text.config(state="disabled")
        self._debug_after = self.root.after(LOG_POLL_MS, self._poll_debug_queue)

    def _close_debug_win(self):
        if self._debug_after is not None:
            self.root.after_cancel(self._debug_after)
            self._debug_after = None
        if getattr(self, "_debug_unsub", None):
            self._debug_unsub()
            self._debug_unsub = None
        if self._debug_win and self._debug_win.winfo_exists():
            self._debug_win.destroy()
        self._debug_win = None
        self._debug_text = None

    def _on_debug_win_close(self):
        self._debug_var.set(False)
        self._close_debug_win()

    def _debug_log(self, message):
        # ponytail: single-arg legacy entry point -> derive level + emit
        ml = message.upper()
        if "ERROR" in ml:
            level = "ERROR"
        elif "WARNING" in ml or "WARN" in ml:
            level = "WARNING"
        else:
            level = "INFO"
        log_bus.emit(level, message)

    def _debug_subscriber(self, level, message, channel=CHANNEL_APP):
        # Emitters include background threads (server stdout monitor, workers);
        # Tk calls from there are unsafe, so park on the queue and let the
        # main-thread poll loop do the widget work.
        self._debug_queue.put((level, message, channel))

    def _poll_debug_queue(self):
        self._debug_after = None
        if self._debug_text is None:
            return
        try:
            pending = []
            dropped = self._debug_queue.take_dropped()
            if dropped:
                pending.append(("WARNING", f"Log buffer full: skipped {dropped} older messages.", CHANNEL_APP))
            while len(pending) < LOG_DRAIN_MAX:
                try:
                    pending.append(self._debug_queue.get_nowait())
                except queue.Empty:
                    break
            append_log_lines(self._debug_text, pending, 2000)
            self._debug_after = self.root.after(LOG_POLL_MS, self._poll_debug_queue)
        except tk.TclError:
            # Window closed; stop polling.
            pass

    # ---------------------------------------------- Tab change
    def _on_tab_changed(self, event):
        selected = self.notebook.select()
        tab_index = self.notebook.index(selected)
        if tab_index == 0:
            self._populate_quick_start()
            self.pipeline_card.populate_models()
        elif tab_index == 2:
            self.subtitle_tab.refresh_model()

    # ---------------------------------------------- Shared helpers
    def get_current_model(self):
        return self.server_tab.model_var.get()

    def get_whisper_models(self, do_scan=False):
        model_dir = self.config_manager.get("whisper.model_dir", "")
        if model_dir:
            self.whisper_models.scan_dir = Path(model_dir)
        if do_scan:
            self.whisper_models.scan()
        return self.whisper_models.list_models()

    def _on_srt_generated(self, srt_path):
        self.subtitle_tab.load_file(srt_path)
        self.log("INFO", f"SRT loaded into translation tab: {Path(srt_path).name}")

    # ---------------------------------------------- Logging
    def log(self, level, message):
        # ponytail: was a second sink with a broken _insert_log ref; route to log_bus
        log_bus.emit(level, message)



def main():
    root = TkinterDnD.Tk()
    app = LlamaManager(root)
    root.mainloop()

if __name__ == "__main__":
    main()
