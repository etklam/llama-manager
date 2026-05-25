#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llama.cpp Manager - 簡單的 GUI 管理器
用於管理 llama.cpp 服務器和模型
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import subprocess
import threading
import os
import gc
import time
import psutil
from pathlib import Path
from datetime import datetime

from tkinterdnd2 import TkinterDnD, DND_FILES

from subtitle_tab import SubtitleTranslationTab
from whisper_tab import WhisperTab
from pipeline_runner import PipelineRunner, SUPPORTED_MEDIA
from config_manager import ConfigManager
from server_controller import ServerController
from model_registry import ModelRegistry
from whisper_model_registry import WhisperModelRegistry

class LlamaManager:
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

        self.server = ServerController(self.server_exe, self._on_server_log)
        self.models = ModelRegistry(self.hip_dir, self.config_manager)

        whisper_model_dir = Path(self.config_manager.get("whisper.model_dir", str(self.hip_dir)))
        self.whisper_models = WhisperModelRegistry(whisper_model_dir, self.config_manager)

        self.monitor_thread = None
        self.monitor_running = False

        self._pipeline_runner = None
        self._debug_win = None
        self._debug_text = None

        self.create_ui()
        self.scan_models()

    def scan_models(self):
        if not self.hip_dir.exists():
            self.log("WARNING", f"目錄不存在: {self.hip_dir}")
            return

        models_list, added_count, removed_count = self.models.scan()

        if removed_count > 0:
            self.log("INFO", f"移除 {removed_count} 個不存在的模型")
        for m in models_list[-added_count:] if added_count > 0 else []:
            self.log("SUCCESS", f"發現新模型: {m['name']} ({m['size']})")

        self.refresh_model_list()

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
        self.server_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.server_tab, text="Server")
        self._create_server_tab()

        # Tab 2: Subtitle Translation
        self.subtitle_tab = SubtitleTranslationTab(
            self.notebook,
            get_port_callback=lambda: self.port_var.get(),
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
        self._create_quick_pipeline_card(cards, 1)

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

    def _create_quick_pipeline_card(self, parent, col):
        card = ttk.LabelFrame(parent, text="🎤📝 Quick Whisper → Translate", padding="10")
        card.grid(row=0, column=col, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        card.columnconfigure(1, weight=1)
        card.rowconfigure(5, weight=1)

        file_btn_row = ttk.Frame(card)
        file_btn_row.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Button(file_btn_row, text="Choose Files",
                   command=self._pipe_browse_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_btn_row, text="Clear",
                   command=self._pipe_clear_files).pack(side=tk.LEFT, padx=2)
        self._pipe_replace_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(file_btn_row, text="Replace original",
                        variable=self._pipe_replace_var).pack(side=tk.LEFT, padx=10)

        list_frame = ttk.Frame(card)
        list_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(3, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self._pipe_listbox = tk.Listbox(list_frame, height=4, selectmode=tk.EXTENDED)
        self._pipe_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._pipe_listbox.drop_target_register(DND_FILES)
        self._pipe_listbox.dnd_bind('<<Drop>>', self._pipe_on_drop)
        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._pipe_listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._pipe_listbox.config(yscrollcommand=sb.set)

        ttk.Label(card, text="Whisper Model:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self._pipe_wmodel_var = tk.StringVar()
        self._pipe_wmodel_combo = ttk.Combobox(card, textvariable=self._pipe_wmodel_var,
                                                state="readonly", width=25)
        self._pipe_wmodel_combo.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))

        row3 = ttk.Frame(card)
        row3.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Label(row3, text="Language:").pack(side=tk.LEFT, padx=(0, 2))
        self._pipe_lang_var = tk.StringVar(value="auto")
        lang_combo = ttk.Combobox(row3, textvariable=self._pipe_lang_var,
                                   state="readonly", width=12)
        lang_combo.pack(side=tk.LEFT, padx=(0, 10))
        from whisper_tab import WHISPER_LANGUAGES
        codes = list(WHISPER_LANGUAGES.keys())
        names = [f"{c} - {WHISPER_LANGUAGES[c]}" for c in codes]
        lang_combo['values'] = names
        lang_combo.current(0)

        ttk.Label(row3, text="Target:").pack(side=tk.LEFT, padx=(0, 2))
        self._pipe_target_var = tk.StringVar(value="zh-cn - Simplified Chinese")
        target_combo = ttk.Combobox(row3, textvariable=self._pipe_target_var,
                                     state="readonly", width=20)
        target_combo.pack(side=tk.LEFT)
        from subtitle_tab import TARGET_LANGUAGES
        tcodes = list(TARGET_LANGUAGES.keys())
        tnames = [f"{c} - {TARGET_LANGUAGES[c]}" for c in tcodes]
        target_combo['values'] = tnames
        target_combo.current(0)

        self._pipe_status_label = ttk.Label(card, text="Ready", foreground="gray")
        self._pipe_status_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        btn_frame = ttk.Frame(card)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(8, 0))
        self._pipe_start_btn = ttk.Button(btn_frame, text="▶ Start",
                                           command=self._start_pipeline)
        self._pipe_start_btn.pack(side=tk.LEFT, padx=2)
        self._pipe_stop_btn = ttk.Button(btn_frame, text="⏹ Stop",
                                          command=self._stop_pipeline, state="disabled")
        self._pipe_stop_btn.pack(side=tk.LEFT, padx=2)

        self._pipe_files = []

    def _populate_quick_start(self):
        model_list = self.models.list_models()
        names = [m.get("name", "Unknown") for m in model_list]
        self._qs_model_combo['values'] = names
        last = self.config_manager.get("ui.last_model", "")
        if last and last in names:
            self._qs_model_var.set(last)
        elif names:
            self._qs_model_combo.current(0)

    def _populate_pipeline_models(self):
        models = self.get_whisper_models()
        names = [m.get('name', str(m)) for m in models]
        self._pipe_wmodel_combo['values'] = names
        last = self.config_manager.get("whisper.last_model", "")
        if last and last in names:
            self._pipe_wmodel_var.set(last)
        elif names:
            self._pipe_wmodel_combo.current(0)

    # -------------------------------------------------- Quick Start Llama
    def _quick_start_server(self):
        model_name = self._qs_model_var.get()
        if not model_name:
            messagebox.showerror("Error", "Select a model first!")
            return
        if self.server.running:
            messagebox.showwarning("Warning", "Server already running!")
            return

        model_path = self.models.get_model_path(model_name)
        if not model_path or not Path(model_path).exists():
            messagebox.showerror("Error", f"Model not found: {model_path}")
            return

        port = self._qs_port_var.get()
        self.config_manager.set("server.port", port)
        self.config_manager.set("server.gpu_layers", self.config_manager.get("server.gpu_layers", 99))
        self.config_manager.set("server.context_size", self.config_manager.get("server.context_size", 131072))
        self.config_manager.set("server.batch_size", self.config_manager.get("server.batch_size", 256))
        if model_name:
            self.config_manager.set("ui.last_model", model_name)

        try:
            self._qs_status_label.config(text=f"Starting {model_name}...", foreground="blue")
            self._debug_log(f"Starting server: {model_name} (port {port})")
            self.server.start(
                model_path=model_path, port=port,
                host=self.config_manager.get("server.host", "0.0.0.0"),
                gpu_layers=self.config_manager.get("server.gpu_layers", 99),
                context_size=self.config_manager.get("server.context_size", 131072),
                batch_size=self.config_manager.get("server.batch_size", 256)
            )
            self._qs_start_btn.config(state="disabled")
            self._qs_stop_btn.config(state="normal")
            self._qs_status_label.config(text=f"● Running (port {port})", foreground="green")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_label.config(text="● 運行中", foreground="green")
            self.start_resource_monitor()
            self._debug_log(f"Server started: http://localhost:{port}")
        except Exception as e:
            self._qs_status_label.config(text=f"● Failed", foreground="red")
            self._debug_log(f"Start failed: {e}")
            messagebox.showerror("Error", f"Start failed:\n{e}")

    def _quick_stop_server(self):
        if not self.server.running:
            return
        try:
            self._debug_log("Stopping server...")
            self.server.stop()
            self._qs_start_btn.config(state="normal")
            self._qs_stop_btn.config(state="disabled")
            self._qs_status_label.config(text="● Stopped", foreground="gray")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.status_label.config(text="● 未運行", foreground="black")
            self.stop_resource_monitor()
            self._debug_log("Server stopped")
        except Exception as e:
            self._qs_status_label.config(text=f"● Stop failed", foreground="red")
            self._debug_log(f"Stop failed: {e}")

    # ---------------------------------------------- Quick Pipeline
    def _pipe_browse_file(self):
        exts = ";".join(f"*{e}" for e in SUPPORTED_MEDIA)
        files = filedialog.askopenfilenames(
            title="Select SRT / Audio / Video files",
            filetypes=[("Subtitle + Media", f"*.srt;*.txt;{exts}"),
                       ("All files", "*.*")])
        for f in files:
            if f not in self._pipe_files:
                self._pipe_files.append(f)
                self._pipe_listbox.insert(tk.END, Path(f).name)

    def _pipe_clear_files(self):
        self._pipe_files.clear()
        self._pipe_listbox.delete(0, tk.END)

    def _pipe_on_drop(self, event):
        import re
        paths = re.findall(r'\{([^}]+)\}', event.data)
        if not paths:
            paths = event.data.split()
        for p in paths:
            p = p.strip()
            if p and p not in self._pipe_files:
                self._pipe_files.append(p)
                self._pipe_listbox.insert(tk.END, Path(p).name)

    def _start_pipeline(self):
        if self._pipeline_runner and self._pipeline_runner.running:
            return
        if not self.server.running:
            messagebox.showwarning("Warning", "Start the llama server first!")
            return
        if not self._pipe_files:
            messagebox.showwarning("Warning", "Select files first!")
            return

        has_media = any(Path(f).suffix.lower() in SUPPORTED_MEDIA for f in self._pipe_files)
        if has_media:
            cli_path = self.config_manager.get("whisper.cli_path", "")
            if not cli_path:
                messagebox.showwarning("Warning", "Set whisper-cli path in Speech Recognition tab first!")
                return
            if not self._pipe_wmodel_var.get():
                messagebox.showwarning("Warning", "Select a whisper model!")
                return

        self._pipe_start_btn.config(state="disabled")
        self._pipe_stop_btn.config(state="normal")

        self._pipeline_runner = PipelineRunner(
            config_manager=self.config_manager,
            get_port=lambda: self.port_var.get(),
            get_current_model=self.get_current_model,
            resolve_whisper_model_path=self._resolve_whisper_model_path,
            get_whisper_models=self.get_whisper_models,
            on_log=self._debug_log,
            on_progress=self._pipeline_step,
            on_done=self._pipeline_done,
        )

        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _stop_pipeline(self):
        if self._pipeline_runner:
            self._pipeline_runner.stop()
        self._pipeline_step("Stopping...")

    def _run_pipeline(self):
        files = list(self._pipe_files)
        cli_path = Path(self.config_manager.get("whisper.cli_path", ""))
        model_name = self._pipe_wmodel_var.get()
        model_dir = self.config_manager.get("whisper.model_dir", "")
        lang_val = self._pipe_lang_var.get()
        language = lang_val.split(' - ')[0] if ' - ' in lang_val else lang_val
        target_val = self._pipe_target_var.get()
        target_lang = target_val.split(' - ')[0] if ' - ' in target_val else target_val

        self._pipeline_runner.run(
            files=files,
            target_lang=target_lang,
            language=language,
            replace_original=self._pipe_replace_var.get(),
            whisper_cli_path=cli_path,
            whisper_model_name=model_name,
            whisper_model_dir=model_dir,
        )

    def _pipeline_step(self, msg):
        self.root.after(0, lambda: self._pipe_status_label.config(
            text=msg, foreground="blue"))
        self._debug_log(msg)

    def _pipeline_done(self, stopped=False):
        self.root.after(0, lambda: self._update_pipeline_done(stopped))

    def _update_pipeline_done(self, stopped):
        self._pipe_start_btn.config(state="normal")
        self._pipe_stop_btn.config(state="disabled")
        if stopped:
            self._pipe_status_label.config(text="Stopped", foreground="orange")
        else:
            self._pipe_status_label.config(text="All done!", foreground="green")

    def _resolve_whisper_model_path(self, model_dir, model_name):
        if not model_dir or not model_name:
            return model_name
        models = self.get_whisper_models()
        for m in models:
            if m.get('name') == model_name:
                return m.get('path', model_name)
        return str(Path(model_dir) / model_name)

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
            self._populate_pipeline_models()
        elif tab_index == 2:
            self.subtitle_tab.refresh_model()

    # ---------------------------------------------- Shared helpers
    def get_current_model(self):
        return self.model_var.get()

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

    # ---------------------------------------------- Server tab
    def _create_server_tab(self):
        main_frame = self.server_tab

        title_label = ttk.Label(
            main_frame,
            text="🚀 llama.cpp Manager",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 10))

        model_frame = ttk.LabelFrame(main_frame, text="📦 模型選擇", padding="10")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)

        ttk.Label(model_frame, text="選擇模型:").grid(row=0, column=0, sticky=tk.W)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            state="readonly",
            width=50
        )
        self.model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 5))

        ttk.Button(
            model_frame,
            text="🔄 掃描",
            command=self.scan_models,
            width=8
        ).grid(row=0, column=2, padx=5)

        ttk.Button(
            model_frame,
            text="📂 添加",
            command=self.add_model,
            width=8
        ).grid(row=0, column=3, padx=5)

        self.model_info_label = ttk.Label(model_frame, text="")
        self.model_info_label.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))

        server_frame = ttk.LabelFrame(main_frame, text="⚙️ 服務器設置", padding="10")
        server_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        params_grid = ttk.Frame(server_frame)
        params_grid.grid(row=0, column=0, sticky=(tk.W, tk.E))

        ttk.Label(params_grid, text="端口:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.port_var = tk.IntVar(value=self.config_manager.get("server.port", 8080))
        ttk.Entry(params_grid, textvariable=self.port_var, width=10).grid(row=0, column=1, padx=(0, 20))

        ttk.Label(params_grid, text="GPU 層數:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.gpu_layers_var = tk.IntVar(value=self.config_manager.get("server.gpu_layers", 99))
        ttk.Scale(
            params_grid,
            from_=0, to=99,
            variable=self.gpu_layers_var,
            orient=tk.HORIZONTAL, length=150,
            command=lambda v: self.gpu_layers_label.config(text=f"{int(float(v))}")
        ).grid(row=0, column=3, padx=(0, 5))
        self.gpu_layers_label = ttk.Label(params_grid, text="99")
        self.gpu_layers_label.grid(row=0, column=4, padx=(0, 20))

        ttk.Label(params_grid, text="上下文大小:").grid(row=0, column=5, sticky=tk.W, padx=(0, 5))
        self.context_var = tk.IntVar(value=self.config_manager.get("server.context_size", 131072))
        context_combo = ttk.Combobox(
            params_grid, textvariable=self.context_var,
            values=[512, 1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072],
            width=10, state="readonly")
        context_combo.grid(row=0, column=6, padx=(0, 20))

        ttk.Label(params_grid, text="批次大小:").grid(row=0, column=7, sticky=tk.W, padx=(0, 5))
        self.batch_var = tk.IntVar(value=self.config_manager.get("server.batch_size", 256))
        ttk.Entry(params_grid, textvariable=self.batch_var, width=10).grid(row=0, column=8)

        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, pady=(0, 10))

        self.start_button = ttk.Button(
            control_frame, text="▶️ 啟動服務器",
            command=self.start_server, width=20)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(
            control_frame, text="⏹️ 停止服務器",
            command=self.stop_server, width=20, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)

        self.release_button = ttk.Button(
            control_frame, text="🧹 釋放內存",
            command=self.release_memory, width=15, state="normal")
        self.release_button.grid(row=0, column=2, padx=5)

        self.status_label = ttk.Label(
            control_frame, text="● 未運行", font=("Arial", 10))
        self.status_label.grid(row=0, column=2, padx=20)

        log_frame = ttk.LabelFrame(main_frame, text="📋 運行日誌", padding="10")
        log_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=15, font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.log_text.tag_config("INFO", foreground="blue")
        self.log_text.tag_config("ERROR", foreground="red")
        self.log_text.tag_config("SUCCESS", foreground="green")
        self.log_text.tag_config("WARNING", foreground="orange")

        self.resource_label = ttk.Label(
            main_frame,
            text="GPU: N/A | VRAM: N/A | RAM: N/A",
            font=("Consolas", 9))
        self.resource_label.grid(row=5, column=0, pady=(10, 0), sticky=tk.W)

        self.refresh_model_list()

        last_model = self.config_manager.get("ui.last_model", "")
        if last_model and last_model in self.model_combo['values']:
            self.model_var.set(last_model)
            self.on_model_select(None)

        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_select)

    def refresh_model_list(self):
        model_list = self.models.list_models()
        model_names = [m.get("name", "Unknown") for m in model_list]
        self.model_combo['values'] = model_names

        if model_names and not self.model_var.get():
            self.model_combo.current(0)
            self.on_model_select(None)

    def on_model_select(self, event):
        model_name = self.model_var.get()
        if model_name:
            self.config_manager.set("ui.last_model", model_name)

        for model in self.models.list_models():
            if model.get("name") == model_name:
                info = f"大小: {model.get('size', 'N/A')} | 格式: {model.get('format', 'N/A')}"
                self.model_info_label.config(text=info)
                return
        self.model_info_label.config(text="")

    def add_model(self):
        file_path = filedialog.askopenfilename(
            title="選擇模型文件",
            filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")],
            initialdir=str(self.hip_dir))
        if file_path:
            model_info = self.models.add_model(file_path)
            self.refresh_model_list()
            self.log("SUCCESS", f"已添加模型: {model_info['name']}")

    def start_server(self):
        if not self.model_var.get():
            messagebox.showerror("錯誤", "請先選擇一個模型！")
            return
        if self.server.running:
            messagebox.showwarning("警告", "服務器已在運行中！")
            return

        model_name = self.model_var.get()
        model_path = self.models.get_model_path(model_name)

        if not model_path or not Path(model_path).exists():
            messagebox.showerror("錯誤", f"找不到模型文件: {model_path}")
            return

        self.config_manager.set("server.port", self.port_var.get())
        self.config_manager.set("server.gpu_layers", self.gpu_layers_var.get())
        self.config_manager.set("server.context_size", self.context_var.get())
        self.config_manager.set("server.batch_size", self.batch_var.get())

        try:
            self.log("INFO", f"啟動服務器: {model_name}")
            self.server.start(
                model_path=model_path,
                port=self.port_var.get(),
                host=self.config_manager.get("server.host", "0.0.0.0"),
                gpu_layers=self.gpu_layers_var.get(),
                context_size=self.context_var.get(),
                batch_size=self.batch_var.get()
            )
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_label.config(text="● 運行中", foreground="green")
            self.start_resource_monitor()
            self.log("SUCCESS", f"服務器已啟動在 http://localhost:{self.port_var.get()}")
        except Exception as e:
            self.log("ERROR", f"啟動失敗: {str(e)}")
            messagebox.showerror("錯誤", f"啟動服務器失敗:\n{str(e)}")

    def stop_server(self):
        if not self.server.running:
            return
        try:
            self.server.stop()
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.status_label.config(text="● 未運行", foreground="black")
            self.stop_resource_monitor()
            self.log("SUCCESS", "服務器已停止")
        except Exception as e:
            self.log("ERROR", f"停止失敗: {str(e)}")

    def release_memory(self):
        try:
            self.log("INFO", "正在釋放系統內存...")
            gc.collect()

            killed_count = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    if proc_info['name']:
                        proc_name = proc_info['name'].lower()
                        if 'llama' in proc_name or proc_name.endswith('.exe'):
                            if self.server.running and self.server._process and proc_info['pid'] == self.server._process.pid:
                                continue
                            proc.terminate()
                            killed_count += 1
                            self.log("INFO", f"已終止進程: {proc_info['name']} (PID: {proc_info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            try:
                subprocess.run(
                    ["EmptyStandbyList.exe", "standbylist"],
                    capture_output=True, timeout=5)
                self.log("SUCCESS", "已清理系統待機列表")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            ram = psutil.virtual_memory()
            ram_available_gb = ram.available / (1024**3)
            ram_total_gb = ram.total / (1024**3)

            if killed_count > 0:
                self.log("SUCCESS", f"內存釋放完成！終止了 {killed_count} 個進程")
                self.log("INFO", f"可用 RAM: {ram_available_gb:.1f}GB / {ram_total_gb:.1f}GB")
            else:
                self.log("INFO", f"沒有發現需要清理的進程")
                self.log("INFO", f"當前可用 RAM: {ram_available_gb:.1f}GB / {ram_total_gb:.1f}GB")
        except Exception as e:
            self.log("ERROR", f"釋放內存失敗: {str(e)}")
            messagebox.showerror("錯誤", f"釋放內存失敗:\n{str(e)}")

    def _on_server_log(self, line: str):
        self.log("INFO", line)

    def start_resource_monitor(self):
        self.monitor_running = True
        self.monitor_thread = threading.Thread(target=self.monitor_resources, daemon=True)
        self.monitor_thread.start()

    def stop_resource_monitor(self):
        self.monitor_running = False

    def monitor_resources(self):
        while self.monitor_running:
            try:
                ram = psutil.virtual_memory()
                ram_used = ram.used / (1024**3)
                ram_total = ram.total / (1024**3)
                ram_percent = ram.percent

                gpu_status = "未使用"
                vram_status = "N/A"
                process_info = ""

                if self.server.running and self.server._process:
                    try:
                        proc = psutil.Process(self.server._process.pid)
                        cpu_percent = proc.cpu_percent(interval=0.1)
                        mem_info = proc.memory_info()
                        proc_mem_mb = mem_info.rss / (1024**2)
                        try:
                            num_threads = proc.num_threads()
                        except:
                            num_threads = 0
                        process_info = f"服務器: CPU {cpu_percent:.1f}% | 內存 {proc_mem_mb:.0f}MB | 線程 {num_threads}"
                        gpu_status = "運行中"
                        vram_status = "已載入"
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                if process_info:
                    status_text = f"RAM: {ram_used:.1f}GB / {ram_total:.1f}GB ({ram_percent}%) | GPU: {gpu_status} | {process_info}"
                else:
                    status_text = f"RAM: {ram_used:.1f}GB / {ram_total:.1f}GB ({ram_percent}%) | GPU: {gpu_status} | VRAM: {vram_status}"

                self.root.after(0, lambda text=status_text: self.resource_label.config(text=text))
            except Exception as e:
                print(f"Resource monitor error: {e}")
                self.root.after(0, lambda: self.resource_label.config(
                    text=f"系統監控中... (錯誤: {str(e)[:30]})"))

            time.sleep(2)

    # ---------------------------------------------- Logging
    def log(self, level, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}\n"
        self.root.after(0, lambda: self._insert_log(log_message, level))

    def _insert_log(self, message, level):
        self.log_text.insert(tk.END, message, level)
        self.log_text.see(tk.END)
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 1000:
            self.log_text.delete(1.0, f"{lines-1000}.0")



def main():
    root = TkinterDnD.Tk()
    app = LlamaManager(root)
    root.mainloop()

if __name__ == "__main__":
    main()
