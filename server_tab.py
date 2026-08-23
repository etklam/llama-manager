"""
ServerTab - Server management UI for llama.cpp
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import gc
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime

import psutil

from ui_helpers import CHANNEL_APP, CHANNEL_SERVER, LogMixin


CONTEXT_SIZE_OPTIONS = (
    512, 1024, 2048, 4096, 8192, 16384,
    32768, 49152, 65536, 98304, 131072,
)


TRANSLATION_PRESET = {
    "context_size": 16384,
    "batch_size": 512,
    "parallel": 3,
    "flash_attn": True,
    "cont_batching": True,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
}

# Chat mode serves an external tool over llama-server's OpenAI-compatible
# endpoint, so a conversation accumulates history across turns: more context
# than the per-batch translation path needs, but quantized KV keeps the VRAM
# cost of that history down. Two slots let a second client (or a retry) through
# without the deep-context cost of the long-context preset.
CHAT_PRESET = {
    "context_size": 32768,
    "batch_size": 512,
    "parallel": 2,
    "flash_attn": True,
    "cont_batching": True,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
}

LONG_CONTEXT_PRESET = {
    "context_size": 131072,
    "batch_size": 512,
    "parallel": 1,
    "flash_attn": True,
    "cont_batching": True,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
}


class ServerTab(LogMixin, ttk.Frame):
    log_channel = CHANNEL_SERVER

    def __init__(self, parent, server_controller, models_registry,
                 config_manager, base_dir, on_model_selected=None,
                 on_server_state_changed=None):
        super().__init__(parent, padding="10")
        self._server = server_controller
        self._models = models_registry
        self._config = config_manager
        self._base_dir = base_dir
        self._on_model_selected = on_model_selected
        self._on_server_state_changed = on_server_state_changed
        self._monitor_thread = None
        self._monitor_running = False
        self._monitor_generation = 0
        self._active_model_name = None

        self._create_ui()

    def _create_ui(self):
        main_frame = self

        title_label = ttk.Label(
            main_frame,
            text="\U0001F680 llama.cpp Manager",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 10))

        model_frame = ttk.LabelFrame(main_frame, text="\U0001F4E6 \u6A21\u578B\u9078\u64C7", padding="10")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)

        ttk.Label(model_frame, text="\u9078\u64C7\u6A21\u578B:").grid(row=0, column=0, sticky=tk.W)
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
            text="\U0001F504 \u6383\u63CF",
            command=self._on_scan_models,
            width=8
        ).grid(row=0, column=2, padx=5)

        ttk.Button(
            model_frame,
            text="\U0001F4C2 \u6DFB\u52A0",
            command=self.add_model,
            width=8
        ).grid(row=0, column=3, padx=5)

        self.model_info_label = ttk.Label(model_frame, text="")
        self.model_info_label.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))

        server_frame = ttk.LabelFrame(main_frame, text="\u2699\uFE0F \u670D\u52A1\u5668\u8BBE\u7F6E", padding="10")
        server_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        server_frame.columnconfigure(0, weight=1)

        params_grid = ttk.Frame(server_frame)
        params_grid.grid(row=0, column=0, sticky=(tk.W, tk.E))

        ttk.Label(params_grid, text="\u7AEF\u53E3:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.port_var = tk.IntVar(value=self._config.get("server.port", 8080))
        ttk.Entry(params_grid, textvariable=self.port_var, width=10).grid(row=0, column=1, padx=(0, 20))

        ttk.Label(params_grid, text="GPU \u5C42\u6570:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.gpu_layers_var = tk.IntVar(value=self._config.get("server.gpu_layers", 99))
        ttk.Scale(
            params_grid,
            from_=0, to=99,
            variable=self.gpu_layers_var,
            orient=tk.HORIZONTAL, length=150,
            command=lambda v: self.gpu_layers_label.config(text=f"{int(float(v))}")
        ).grid(row=0, column=3, padx=(0, 5))
        self.gpu_layers_label = ttk.Label(params_grid, text="99")
        self.gpu_layers_label.grid(row=0, column=4, padx=(0, 20))

        ttk.Label(params_grid, text="\u4E0A\u4E0B\u6587\u5927\u5C0F:").grid(row=0, column=5, sticky=tk.W, padx=(0, 5))
        self.context_var = tk.IntVar(value=self._config.get("server.context_size", 16384))
        context_combo = ttk.Combobox(
            params_grid, textvariable=self.context_var,
            values=CONTEXT_SIZE_OPTIONS,
            width=10, state="readonly")
        context_combo.grid(row=0, column=6, padx=(0, 20))

        ttk.Label(params_grid, text="\u6279\u6B21\u5927\u5C0F:").grid(row=0, column=7, sticky=tk.W, padx=(0, 5))
        self.batch_var = tk.IntVar(value=self._config.get("server.batch_size", 512))
        ttk.Entry(params_grid, textvariable=self.batch_var, width=10).grid(row=0, column=8)

        # Second row: concurrency + KV cache quantization
        ttk.Label(params_grid, text="\u4E26\u767C\u69FD (-np):").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(8, 0))
        self.parallel_var = tk.IntVar(value=self._config.get("server.parallel", 3))
        ttk.Spinbox(params_grid, from_=1, to=16, textvariable=self.parallel_var,
                    width=8).grid(row=1, column=1, padx=(0, 20), pady=(8, 0))

        self.flash_attn_var = tk.BooleanVar(value=self._config.get("server.flash_attn", True))
        ttk.Checkbutton(params_grid, text="FlashAttention",
                        variable=self.flash_attn_var).grid(row=1, column=2, columnspan=2,
                                                           sticky=tk.W, padx=(0, 20), pady=(8, 0))

        ttk.Label(params_grid, text="KV \u5FEB\u53D6 K:").grid(row=1, column=4, sticky=tk.W, padx=(0, 5), pady=(8, 0))
        self.cache_type_k_var = tk.StringVar(value=self._config.get("server.cache_type_k", "q8_0"))
        ttk.Combobox(params_grid, textvariable=self.cache_type_k_var,
                     values=["f16", "q8_0", "q4_0"], width=8,
                     state="readonly").grid(row=1, column=5, padx=(0, 20), pady=(8, 0))

        ttk.Label(params_grid, text="KV \u5FEB\u53D6 V:").grid(row=1, column=6, sticky=tk.W, padx=(0, 5), pady=(8, 0))
        self.cache_type_v_var = tk.StringVar(value=self._config.get("server.cache_type_v", "q8_0"))
        ttk.Combobox(params_grid, textvariable=self.cache_type_v_var,
                     values=["f16", "q8_0", "q4_0"], width=8,
                     state="readonly").grid(row=1, column=7, columnspan=2, sticky=tk.W, pady=(8, 0))

        dflash_frame = ttk.LabelFrame(
            server_frame, text="DFlash speculative decoding", padding="8"
        )
        dflash_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        dflash_frame.columnconfigure(2, weight=1)

        self.dflash_enabled_var = tk.BooleanVar(
            value=self._config.get("server.dflash_enabled", False)
        )
        ttk.Checkbutton(
            dflash_frame, text="Enable DFlash",
            variable=self.dflash_enabled_var,
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 12))

        ttk.Label(dflash_frame, text="Draft GGUF:").grid(
            row=0, column=1, sticky=tk.W, padx=(0, 5)
        )
        self.dflash_model_path_var = tk.StringVar(value=self._initial_auxiliary_path(
            "server.dflash_model_path", "dflash-kquant.gguf"
        ))
        ttk.Entry(
            dflash_frame, textvariable=self.dflash_model_path_var, width=48
        ).grid(row=0, column=2, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(
            dflash_frame, text="Browse...",
            command=lambda: self._browse_auxiliary_model(
                self.dflash_model_path_var, "Select DFlash draft model"
            ),
        ).grid(row=0, column=3, padx=(0, 12))

        ttk.Label(dflash_frame, text="N max:").grid(
            row=0, column=4, sticky=tk.W, padx=(0, 5)
        )
        self.dflash_n_max_var = tk.IntVar(
            value=self._config.get("server.dflash_n_max", 6)
        )
        ttk.Spinbox(
            dflash_frame, from_=1, to=15,
            textvariable=self.dflash_n_max_var, width=5,
        ).grid(row=0, column=5, sticky=tk.W)

        ttk.Label(dflash_frame, text="mmproj GGUF (optional):").grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(8, 0),
            padx=(0, 5)
        )
        self.mmproj_path_var = tk.StringVar(value=self._initial_auxiliary_path(
            "server.mmproj_path", "mmproj-kquant.gguf"
        ))
        ttk.Entry(
            dflash_frame, textvariable=self.mmproj_path_var, width=48
        ).grid(
            row=1, column=2, sticky=(tk.W, tk.E), padx=(0, 5), pady=(8, 0)
        )
        ttk.Button(
            dflash_frame, text="Browse...",
            command=lambda: self._browse_auxiliary_model(
                self.mmproj_path_var, "Select multimodal projector"
            ),
        ).grid(row=1, column=3, padx=(0, 12), pady=(8, 0))

        preset_frame = ttk.Frame(server_frame)
        preset_frame.grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        ttk.Label(preset_frame, text="用途 Preset:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            preset_frame, text="翻譯模式",
            command=self._apply_translation_preset,
            width=14,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            preset_frame, text="對話模式",
            command=self._apply_chat_preset,
            width=14,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            preset_frame, text="128K 長上下文",
            command=self._apply_long_context_preset,
            width=16,
        ).pack(side=tk.LEFT)
        ttk.Label(
            preset_frame,
            text="翻譯 16K/3 slots · 對話 32K/2 slots · 長上下文 128K/1 slot",
            foreground="gray",
        ).pack(side=tk.LEFT, padx=(10, 0))

        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, pady=(0, 10))

        self.start_button = ttk.Button(
            control_frame, text="\u25B6\uFE0F \u555F\u52D5\u670D\u52A1\u5668",
            command=self.start_server, width=20)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(
            control_frame, text="\u23F9\uFE0F \u505C\u6B62\u670D\u52A1\u5668",
            command=self.stop_server, width=20, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)

        self.release_button = ttk.Button(
            control_frame, text="\U0001F9F9 \u91CB\u653E\u5167\u5B58",
            command=self.release_memory, width=15, state="normal")
        self.release_button.grid(row=0, column=2, padx=5)

        self.status_label = ttk.Label(
            control_frame, text="\u25CF \u672A\u904B\u884C", font=("Arial", 10))
        self.status_label.grid(row=0, column=3, padx=20)

        # The server log also carries app-level notices (model scan results),
        # which the Main tab has no log widget of its own to show.
        self._init_log_widget(main_frame, row=4, column=0,
                              label="\U0001F4CB \u904B\u884C\u65E5\u8A8C", height=15, max_lines=1000,
                              channels=(CHANNEL_SERVER, CHANNEL_APP))

        self.resource_label = ttk.Label(
            main_frame,
            text="GPU: N/A | VRAM: N/A | RAM: N/A",
            font=("Consolas", 9))
        self.resource_label.grid(row=5, column=0, pady=(10, 0), sticky=tk.W)

        self.refresh_model_list()

        last_model = self._config.get("ui.last_model", "")
        if last_model and last_model in self.model_combo['values']:
            self.model_var.set(last_model)
            self.on_model_select(None)

        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_select)

    def log(self, level, message):
        self._log(level, message)

    def _initial_auxiliary_path(self, config_key, suggested_filename):
        configured = self._config.get(config_key, "")
        if configured:
            return configured
        suggested = Path(self._base_dir) / suggested_filename
        return str(suggested) if suggested.is_file() else ""

    def _browse_auxiliary_model(self, path_var, title):
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")],
            initialdir=str(self._base_dir),
        )
        if file_path:
            path_var.set(file_path)

    # --------------------------------------------------------- Model management
    def refresh_model_list(self):
        model_list = self._models.list_models()
        model_names = [m.get("name", "Unknown") for m in model_list]
        self.model_combo['values'] = model_names

        if model_names and not self.model_var.get():
            self.model_combo.current(0)
            self.on_model_select(None)

    def on_model_select(self, event):
        model_name = self.model_var.get()
        if model_name:
            self._config.set("ui.last_model", model_name)
            if self._on_model_selected:
                self._on_model_selected(model_name)

        for model in self._models.list_models():
            if model.get("name") == model_name:
                info = f"\u5927\u5C0F: {model.get('size', 'N/A')} | \u683C\u5F0F: {model.get('format', 'N/A')}"
                self.model_info_label.config(text=info)
                return
        self.model_info_label.config(text="")

    def add_model(self):
        file_path = filedialog.askopenfilename(
            title="\u9078\u64C7\u6A21\u578B\u6587\u4EF6",
            filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")],
            initialdir=str(self._base_dir))
        if file_path:
            model_info = self._models.add_model(file_path)
            self.refresh_model_list()
            self.log("SUCCESS", f"\u5DF2\u6DFB\u52A0\u6A21\u578B: {model_info['name']}")

    def _on_scan_models(self):
        """Callback for the scan button - delegates to the scan_models method
        on the parent LlamaManager via winfo_toplevel()."""
        # The parent manager's scan_models will call refresh_model_list on this tab
        root = self.winfo_toplevel()
        if hasattr(root, '_app') and hasattr(root._app, 'scan_models'):
            root._app.scan_models()

    # -------------------------------------------------------- Server presets
    def _apply_translation_preset(self):
        self._apply_server_preset("翻譯模式", TRANSLATION_PRESET)

    def _apply_chat_preset(self):
        self._apply_server_preset("對話模式", CHAT_PRESET)

    def _apply_long_context_preset(self):
        self._apply_server_preset("長上下文模式", LONG_CONTEXT_PRESET)

    def _apply_server_preset(self, name, values):
        """Apply a server-use preset to the visible fields and config.

        Values are persisted immediately so other launch paths observe the same
        settings. The user can still fine-tune any field before starting.
        """
        self.context_var.set(values["context_size"])
        self.batch_var.set(values["batch_size"])
        self.parallel_var.set(values["parallel"])
        self.flash_attn_var.set(values["flash_attn"])
        self.cache_type_k_var.set(values["cache_type_k"])
        self.cache_type_v_var.set(values["cache_type_v"])

        for key, value in values.items():
            self._config.set(f"server.{key}", value)

        # Bind the translation worker count to the slot count this preset asks
        # for. The two settings live in different tabs and used to drift: the
        # default of 3 workers against a 1-slot server queues two of every three
        # requests, which reads as parallel in the log and is serial in fact.
        # A preset is a statement about how the server will be run, so it is the
        # right place to bring the client's side of that pairing along.
        self._config.set("ui.max_workers", values["parallel"])

        self.log(
            "INFO",
            f"已套用{name}: ctx={values['context_size']}, "
            f"slots={values['parallel']}, "
            f"KV={values['cache_type_k']}/{values['cache_type_v']}, "
            f"翻譯 workers={values['parallel']}"
        )

    # -------------------------------------------------------- Server control
    def start_server(self):
        if not self.model_var.get():
            messagebox.showerror("\u932F\u8AA4", "\u8ACB\u5148\u9078\u64C7\u4E00\u500B\u6A21\u578B\uFF01")
            return

        model_name = self.model_var.get()
        if self._server.running and self._active_model_name == model_name:
            messagebox.showwarning("\u8B66\u544A", "\u6B64\u6A21\u578B\u5DF2\u5728\u904B\u884C\u4E2D\uFF01")
            return
        # Numeric vars (spinboxes are editable) raise TclError when cleared or
        # given non-numeric text; catch once here instead of crashing the
        # Tk callback with no feedback.
        try:
            self._config.set("server.port", self.port_var.get())
            self._config.set("server.gpu_layers", self.gpu_layers_var.get())
            self._config.set("server.context_size", self.context_var.get())
            self._config.set("server.batch_size", self.batch_var.get())
            self._config.set("server.parallel", self.parallel_var.get())
            self._config.set("server.flash_attn", self.flash_attn_var.get())
            self._config.set("server.cache_type_k", self.cache_type_k_var.get())
            self._config.set("server.cache_type_v", self.cache_type_v_var.get())
            self._config.set("server.dflash_enabled", self.dflash_enabled_var.get())
            self._config.set(
                "server.dflash_model_path", self.dflash_model_path_var.get()
            )
            self._config.set("server.dflash_n_max", self.dflash_n_max_var.get())
            self._config.set("server.mmproj_path", self.mmproj_path_var.get())
        except tk.TclError as e:
            messagebox.showerror("錯誤", f"參數輸入無效，請檢查數字欄位：{e}")
            return

        self.log("INFO", f"\u555F\u52D5\u670D\u52A1\u5668: {model_name}")
        if self._do_start_server(model_name, self.port_var.get()):
            self.log("SUCCESS", f"\u670D\u52A1\u5668\u5DF2\u555F\u52D5\u5728 http://localhost:{self.port_var.get()}")
        else:
            self.log("ERROR", "\u555F\u52D5\u5931\u6557")

    def stop_server(self):
        if self._do_stop_server():
            self.log("SUCCESS", "\u670D\u52A1\u5668\u5DF2\u505C\u6B62")

    def _do_start_server(self, model_name, port):
        model_path = self._models.get_model_path(model_name)
        if not model_path or not Path(model_path).exists():
            messagebox.showerror("Error", f"Model not found: {model_path}")
            return False

        # P2-2: switching model is one action. Validate the new model first,
        # then stop the old server and release Python-side references before
        # launching it. ServerController.stop() waits for process termination,
        # so model VRAM is released before the next process starts loading.
        if self._server.running:
            previous = self._active_model_name or "目前模型"
            self.log("INFO", f"切換模型: {previous} -> {model_name}")
            if not self._do_stop_server():
                self.log("ERROR", "無法停止舊 server，已取消模型切換")
                return False
            gc.collect()
            self.log("SUCCESS", "舊 server 已停止並釋放記憶體")

        gpu = self._config.get("server.gpu_layers", 99)
        ctx = self._config.get("server.context_size", 16384)
        bsz = self._config.get("server.batch_size", 512)
        parallel = self._config.get("server.parallel", 3)
        flash_attn = self._config.get("server.flash_attn", True)
        cont_batching = self._config.get("server.cont_batching", True)
        cache_type_k = self._config.get("server.cache_type_k", "q8_0")
        cache_type_v = self._config.get("server.cache_type_v", "q8_0")
        dflash_enabled = self._config.get("server.dflash_enabled", False)
        dflash_model_path = self._config.get("server.dflash_model_path", "")
        dflash_n_max = self._config.get("server.dflash_n_max", 6)
        dflash_gpu_layers = self._config.get(
            "server.dflash_gpu_layers", "all"
        )
        dflash_device = self._config.get("server.dflash_device", "Vulkan0")
        mmproj_path = self._config.get("server.mmproj_path", "")
        # ServerController forces --flash-attn on for DFlash; surface that so
        # the checkbox being unchecked doesn't read as the server's behavior.
        if dflash_enabled and not flash_attn:
            self.log("WARNING", "DFlash 需要 flash-attn，已自動啟用 --flash-attn on")

        try:
            self._server.start(
                model_path=model_path, port=port,
                host=self._config.get("server.host", "0.0.0.0"),
                gpu_layers=gpu, context_size=ctx, batch_size=bsz,
                parallel=parallel, flash_attn=flash_attn,
                cont_batching=cont_batching,
                cache_type_k=cache_type_k, cache_type_v=cache_type_v,
                dflash_enabled=dflash_enabled,
                dflash_model_path=dflash_model_path,
                dflash_n_max=dflash_n_max,
                dflash_gpu_layers=dflash_gpu_layers,
                dflash_device=dflash_device,
                mmproj_path=mmproj_path,
            )
            # Keep the start button available while running so selecting another
            # model + pressing it performs the automatic switch above.
            self.start_button.config(state="normal", text="\U0001F504 \u5207\u63DB\u6A21\u578B")
            self.stop_button.config(state="normal")
            self.status_label.config(text="\u25CF \u904B\u884C\u4E2D", foreground="green")
            self.start_resource_monitor()
            self._active_model_name = model_name
            self._config.set("ui.last_model", model_name)
            self.model_var.set(model_name)
            self.port_var.set(port)

            # Notify parent about state change
            if self._on_server_state_changed:
                self._on_server_state_changed(True, model_name, port)

            return True
        except Exception as e:
            messagebox.showerror("Error", f"Start failed:\n{e}")
            return False

    def _do_stop_server(self):
        if not self._server.running:
            return True
        try:
            self._server.stop()
        except Exception as e:
            messagebox.showerror("Error", f"Stop failed: {e}")
            return False

        self._active_model_name = None
        self.start_button.config(state="normal", text="\u25B6\uFE0F \u555F\u52D5\u670D\u52D9\u5668")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="\u25CF \u672A\u904B\u884C", foreground="black")
        self.stop_resource_monitor()

        # Notify parent about state change
        if self._on_server_state_changed:
            self._on_server_state_changed(False, None, None)
        return True

    def release_memory(self):
        try:
            self.log("INFO", "\u6B63\u5728\u91CB\u653E\u7CFB\u7D71\u5167\u5B58...")
            gc.collect()

            killed_count = 0
            target_names = {'llama-server.exe', 'llama-cli.exe', 'whisper-cli.exe',
                            'main.exe', 'server.exe'}
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_info = proc.info
                    if proc_info['name'] and proc_info['name'].lower() in target_names:
                        if (self._server.running and self._server._process
                                and proc_info['pid'] == self._server._process.pid):
                            continue
                        proc.terminate()
                        killed_count += 1
                        self.log("INFO", f"\u5DF2\u7D42\u6B62\u9032\u7A0B: {proc_info['name']} (PID: {proc_info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            try:
                subprocess.run(
                    ["EmptyStandbyList.exe", "standbylist"],
                    capture_output=True, timeout=5)
                self.log("SUCCESS", "\u5DF2\u6E05\u7406\u7CFB\u7D71\u5F85\u6A5F\u5217\u8868")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            ram = psutil.virtual_memory()
            ram_available_gb = ram.available / (1024**3)
            ram_total_gb = ram.total / (1024**3)

            if killed_count > 0:
                self.log("SUCCESS", f"\u5167\u5B58\u91CB\u653E\u5B8C\u6210\uFF01\u7D42\u6B62\u4E86 {killed_count} \u500B\u9032\u7A0B")
                self.log("INFO", f"\u53EF\u7528 RAM: {ram_available_gb:.1f}GB / {ram_total_gb:.1f}GB")
            else:
                self.log("INFO", f"\u6C92\u6709\u767C\u73FE\u9700\u8981\u6E05\u7406\u7684\u9032\u7A0B")
                self.log("INFO", f"\u7576\u524D\u53EF\u7528 RAM: {ram_available_gb:.1f}GB / {ram_total_gb:.1f}GB")
        except Exception as e:
            self.log("ERROR", f"\u91CB\u653E\u5167\u5B58\u5931\u6557: {str(e)}")
            messagebox.showerror("\u932F\u8AA4", f"\u91CB\u653E\u5167\u5B58\u5931\u6557:\n{str(e)}")

    # -------------------------------------------------------- Server log
    def _on_server_log(self, line: str):
        self.log("INFO", line)

    # -------------------------------------------------------- Resource monitor
    def start_resource_monitor(self):
        # A model switch stops and restarts monitoring immediately. Generation
        # tokens make the previous thread exit even if _monitor_running becomes
        # True again before its sleep finishes.
        self._monitor_generation += 1
        generation = self._monitor_generation
        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self.monitor_resources, args=(generation,), daemon=True
        )
        self._monitor_thread.start()

    def stop_resource_monitor(self):
        self._monitor_running = False
        self._monitor_generation += 1

    def monitor_resources(self, generation=None):
        if generation is None:
            generation = self._monitor_generation
        while (self._monitor_running
               and generation == self._monitor_generation):
            try:
                ram = psutil.virtual_memory()
                ram_used = ram.used / (1024**3)
                ram_total = ram.total / (1024**3)
                ram_percent = ram.percent

                gpu_status = "\u672A\u4F7F\u7528"
                vram_status = "N/A"
                process_info = ""

                if self._server.running and self._server._process:
                    try:
                        proc = psutil.Process(self._server._process.pid)
                        cpu_percent = proc.cpu_percent(interval=0.1)
                        mem_info = proc.memory_info()
                        proc_mem_mb = mem_info.rss / (1024**2)
                        try:
                            num_threads = proc.num_threads()
                        except Exception:
                            num_threads = 0
                        process_info = (f"\u670D\u52A1\u5668: CPU {cpu_percent:.1f}% | "
                                        f"\u5167\u5B58 {proc_mem_mb:.0f}MB | \u7EBF\u7A0B {num_threads}")
                        gpu_status = "\u904B\u884C\u4E2D"
                        vram_status = "\u5DF2\u8F09\u5165"
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                if process_info:
                    status_text = (f"RAM: {ram_used:.1f}GB / {ram_total:.1f}GB ({ram_percent}%) | "
                                   f"GPU: {gpu_status} | {process_info}")
                else:
                    status_text = (f"RAM: {ram_used:.1f}GB / {ram_total:.1f}GB ({ram_percent}%) | "
                                   f"GPU: {gpu_status} | VRAM: {vram_status}")

                self.winfo_toplevel().after(0, lambda text=status_text: self.resource_label.config(text=text))
            except Exception as e:
                print(f"Resource monitor error: {e}")
                self.winfo_toplevel().after(0, lambda: self.resource_label.config(
                    text=f"\u7CFB\u7D71\u76E3\u63A7\u4E2D... (\u932F\u8AA4: {str(e)[:30]})"))

            time.sleep(2)
