"""First-run setup dialog for the subtitle/transcription mainline."""
from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from tool_manager import TOOL_NAMES, ToolManager


class SetupWizard(tk.Toplevel):
    """Collect the paths needed for a first subtitle output.

    This first version deliberately prepares existing tools only.  Downloads,
    checksums and model catalogues belong to the ToolManager installer phase and
    can be added without changing this dialog's saved configuration contract.
    """

    _TOOL_LABELS = {
        "llama-server": "llama-server",
        "whisper-cli": "whisper-cli",
        "ffmpeg": "FFmpeg",
        "ffprobe": "ffprobe",
    }

    def __init__(self, parent, config_manager, tool_manager: ToolManager,
                 on_complete=None):
        super().__init__(parent)
        self.title("首次設定精靈")
        self.transient(parent)
        self.resizable(True, False)
        self._config = config_manager
        self._tools = tool_manager
        self._on_complete = on_complete
        self._vars = {
            "usage": tk.StringVar(value="transcribe"),
            "model_dir": tk.StringVar(value=config_manager.get("paths.model_dir", "")),
            "whisper_model_dir": tk.StringVar(
                value=config_manager.get("whisper.model_dir", "")),
        }
        self._tool_vars = {
            name: tk.StringVar(value=tool_manager.configured_path(name))
            for name in TOOL_NAMES
        }
        self._status = tk.StringVar(value="先選擇用途，再檢查現有工具。")
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build(self):
        body = ttk.Frame(self, padding=18)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)

        ttk.Label(body, text="準備字幕流程", font=("TkDefaultFont", 14, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        ttk.Label(body, text="設定會保存到使用者資料目錄；工具可稍後再改。",
                  wraplength=560).grid(row=1, column=0, columnspan=3,
                                       sticky="w", pady=(0, 12))

        ttk.Label(body, text="用途").grid(row=2, column=0, sticky="nw", pady=4)
        usage = ttk.Frame(body)
        usage.grid(row=2, column=1, columnspan=2, sticky="w", pady=4)
        for value, label in (
                ("transcribe", "只做轉錄"),
                ("local", "本地字幕翻譯"),
                ("remote", "遠端 API")):
            ttk.Radiobutton(usage, text=label, value=value,
                            variable=self._vars["usage"]).pack(side="left", padx=(0, 12))

        self._directory_field(body, 3, "文字模型目錄", "model_dir")
        self._directory_field(body, 4, "Whisper 模型目錄", "whisper_model_dir")

        row = 5
        for name in TOOL_NAMES:
            self._tool_field(body, row, name)
            row += 1

        ttk.Label(body, textvariable=self._status, wraplength=560).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(8, 4))
        buttons = ttk.Frame(body)
        buttons.grid(row=row + 1, column=0, columnspan=3, sticky="e", pady=(8, 0))
        ttk.Button(buttons, text="稍後設定", command=self.destroy).pack(side="left", padx=4)
        ttk.Button(buttons, text="檢查工具", command=self._check).pack(side="left", padx=4)
        ttk.Button(buttons, text="完成", command=self._finish).pack(side="left")

    def _directory_field(self, parent, row, label, key):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self._vars[key]).grid(
            row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="瀏覽…",
                   command=lambda: self._browse_directory(key)).grid(
            row=row, column=2, padx=(8, 0), pady=4)

    def _tool_field(self, parent, row, name):
        ttk.Label(parent, text=self._TOOL_LABELS[name]).grid(
            row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self._tool_vars[name]).grid(
            row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="瀏覽…",
                   command=lambda: self._browse_tool(name)).grid(
            row=row, column=2, padx=(8, 0), pady=4)

    def _browse_directory(self, key):
        path = filedialog.askdirectory(title="選擇目錄")
        if path:
            self._vars[key].set(path)

    def _browse_tool(self, name):
        filetypes = ([ ("Executable", "*.exe"), ("All files", "*.*") ]
                     if os.name == "nt" else [("Executable", "*"), ("All files", "*.*")])
        path = filedialog.askopenfilename(
            title=f"選擇 {self._TOOL_LABELS[name]}", filetypes=filetypes)
        if path:
            self._tool_vars[name].set(path)

    def _required_tools(self):
        usage = self._vars["usage"].get()
        if usage == "transcribe":
            return ("whisper-cli", "ffmpeg", "ffprobe")
        if usage == "local":
            return ("llama-server",)
        return ()

    def _check(self):
        missing = []
        for name in self._required_tools():
            value = self._tool_vars[name].get().strip()
            status = self._tools.check(name, value or None, probe=True)
            if not status.available:
                missing.append(status.message)
        if missing:
            self._status.set("需要處理：" + "；".join(missing))
            return False
        self._status.set("所選用途的工具可用。模型檔案仍需自行準備或稍後下載。")
        return True

    def _finish(self):
        if not self._check():
            messagebox.showwarning("設定未完成", self._status.get(), parent=self)
            return

        model_dir = self._vars["model_dir"].get().strip()
        whisper_dir = self._vars["whisper_model_dir"].get().strip()
        if model_dir:
            self._config.set("paths.model_dir", str(Path(model_dir).expanduser()))
        if whisper_dir:
            self._config.set("whisper.model_dir", str(Path(whisper_dir).expanduser()))
        for name, variable in self._tool_vars.items():
            self._tools.set_path(name, variable.get().strip())

        values = {
            "usage": self._vars["usage"].get(),
            "model_dir": model_dir,
            "whisper_model_dir": whisper_dir,
            **{name: variable.get().strip()
               for name, variable in self._tool_vars.items()},
        }
        if self._on_complete:
            self._on_complete(values)
        self.destroy()
