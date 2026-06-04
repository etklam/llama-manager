#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llama.cpp Manager - 簡單的 GUI 管理器
用於管理 llama.cpp 服務器和模型
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path
from datetime import datetime

from tkinterdnd2 import TkinterDnD

from subtitle_tab import SubtitleTranslationTab
from whisper_tab import WhisperTab
from server_tab import ServerTab
from pipeline_card import PipelineCard
from config_manager import ConfigManager
from server_controller import ServerController
from model_registry import ModelRegistry
from whisper_model_registry import WhisperModelRegistry
# constants imported indirectly by pipeline_card, server_tab, etc.
from ui_helpers import LogMixin

class LlamaManager(LogMixin):
    def __init__(self, root):
        self.root = root
        self.root.title("llama.cpp Manager")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        self.base_dir = Path(r"D:\AI\llama\llama.cpp")
        self.hip_dir = self.base_dir / "llama-hip"
        self.server_exe = self.hip_dir / "llama-server.exe"

        self.config_manager = ConfigManager(str(Path(__file__).parent / "config.json"))
        self.config_manager.load()

        self.server = ServerController(self.server_exe, lambda line: None)
        self.models = ModelRegistry(self.hip_dir, self.config_manager)

        whisper_model_dir = Path(self.config_manager.get("whisper.model_dir", str(self.hip_dir)))
        self.whisper_models = WhisperModelRegistry(whisper_model_dir, self.config_manager)

        self._debug_win = None
        self._debug_text = None

        self.create_ui()
        self.scan_models()

        # Allow ServerTab to call back to scan_models
        self.root._app = self

    def scan_models(self):
        if not self.hip_dir.exists():
            self.log("WARNING", f"目錄不存在: {self.hip_dir}")
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
        self.main_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.main_tab, text="Main")
        self._create_main_tab()

        # Tab 1: Server
        self.server_tab = ServerTab(
            self.notebook,
            server_controller=self.server,
            models_registry=self.models,
            config_manager=self.config_manager,
            base_dir=self.hip_dir,
            on_model_selected=self._on_server_model_selected,
            on_server_state_changed=self._on_server_state_changed,
        )
        self.notebook.add(self.server_tab, text="Server")

        # Tab 2: Subtitle Translation
        self.subtitle_tab = SubtitleTranslationTab(
            self.notebook,
            get_port_callback=lambda: self.server_tab.port_var.get(),
            get_model_callback=self.get_current_model,
            config_manager=self.config_manager
        )
        self.notebook.add(self.subtitle_tab, text="Subtitle Translation")

        # Tab 3: Speech Recognition
        self.whisper_tab = WhisperTab(
            self.notebook,
            config_manager=self.config_manager,
            get_whisper_models=self.get_whisper_models,
            scan_whisper_models=lambda: self.get_whisper_models(do_scan=True),
            on_srt_generated=self._on_srt_generated
        )
        self.notebook.add(self.whisper_tab, text="Speech Recognition")

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
        header.grid(row=0, column=0, sticky=tk.W, pady=(0, 15))
        ttk.Label(header, text="🦙 llama.cpp Manager",
                  font=("Arial", 18, "bold")).pack(side=tk.LEFT)
        self._debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(header, text="Debug", variable=self._debug_var,
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
            on_log=lambda level, msg: self.log(level, msg),
            on_debug_log=self._debug_log,
        )
        self.pipeline_card.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))

    def _create_quick_start_card(self, parent, col):
        card = ttk.LabelFrame(parent, text="🚀 Quick Start llama.cpp", padding="10")
        card.grid(row=0, column=col, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        card.columnconfigure(1, weight=1)

        ttk.Label(card, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self._qs_model_var = tk.StringVar()
        self._qs_model_combo = ttk.Combobox(card, textvariable=self._qs_model_var,
                                             state="readonly", width=35)
        self._qs_model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))

        ttk.Label(card, text="Port:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self._qs_port_var = tk.IntVar(value=self.config_manager.get("server.port", 8080))
        ttk.Entry(card, textvariable=self._qs_port_var, width=8).grid(
            row=1, column=1, sticky=tk.W, padx=(0, 5), pady=(5, 0))

        self._qs_status_label = ttk.Label(card, text="● Stopped", foreground="gray")
        self._qs_status_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        btn_frame = ttk.Frame(card)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(8, 0))
        self._qs_start_btn = ttk.Button(btn_frame, text="▶ Start Server",
                                         command=self._quick_start_server)
        self._qs_start_btn.pack(side=tk.LEFT, padx=2)
        self._qs_stop_btn = ttk.Button(btn_frame, text="⏹ Stop Server",
                                        command=self._quick_stop_server, state="disabled")
        self._qs_stop_btn.pack(side=tk.LEFT, padx=2)

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

    def _close_debug_win(self):
        if self._debug_win and self._debug_win.winfo_exists():
            self._debug_win.destroy()
        self._debug_win = None
        self._debug_text = None

    def _on_debug_win_close(self):
        self._debug_var.set(False)
        self._close_debug_win()

    def _debug_log(self, message):
        if not self._debug_var.get():
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        level = None
        ml = message.upper()
        if "ERROR" in ml:
            level = "ERROR"
        elif "WARNING" in ml or "WARN" in ml:
            level = "WARNING"
        lvl = level
        self.root.after(0, lambda: self._debug_insert(line, lvl))

    def _debug_insert(self, line, level):
        if not self._debug_text:
            return
        try:
            self._debug_text.insert(tk.END, line, level or ())
            self._debug_text.see(tk.END)
            lines = int(self._debug_text.index('end-1c').split('.')[0])
            if lines > 2000:
                self._debug_text.delete(1.0, f"{lines - 2000}.0")
        except tk.TclError:
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
        self.root.after(0, lambda: self._insert_log(level, message))



def main():
    root = TkinterDnD.Tk()
    app = LlamaManager(root)
    root.mainloop()

if __name__ == "__main__":
    main()
