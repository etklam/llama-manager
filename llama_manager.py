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

from tkinterdnd2 import TkinterDnD

from subtitle_tab import SubtitleTranslationTab
from config_manager import ConfigManager
from server_controller import ServerController
from model_registry import ModelRegistry

class LlamaManager:
    def __init__(self, root):
        self.root = root
        self.root.title("llama.cpp Manager")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # 配置路徑
        self.base_dir = Path(r"D:\AI\llama\llama.cpp")
        self.hip_dir = self.base_dir / "llama-hip"
        self.server_exe = self.hip_dir / "llama-server.exe"

        # 配置管理
        self.config_manager = ConfigManager(str(Path(__file__).parent / "config.json"))
        self.config_manager.load()

        # ServerController — deep module managing subprocess lifecycle
        self.server = ServerController(self.server_exe, self._on_server_log)

        # ModelRegistry — deep module managing model discovery
        self.models = ModelRegistry(self.hip_dir, self.config_manager)

        # 資源監控
        self.monitor_thread = None
        self.monitor_running = False

        # 創建 UI
        self.create_ui()

        # 自動掃描模型
        self.scan_models()

    def scan_models(self):
        """掃描 llama-hip 目錄中的 .gguf 文件"""
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

    def create_ui(self):
        """Create main interface with tabbed layout."""
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Server tab
        self.server_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.server_tab, text="Server")
        self._create_server_tab()

        # Subtitle Translation tab
        self.subtitle_tab = SubtitleTranslationTab(
            self.notebook,
            get_config_callback=self.get_translation_config,
            get_model_callback=self.get_current_model,
            config_manager=self.config_manager
        )
        self.notebook.add(self.subtitle_tab, text="Subtitle Translation")

        # Refresh subtitle tab when switching to it
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

    def get_translation_config(self):
        """Return config dict for subtitle translation."""
        port = self.port_var.get()
        return {
            'api_url': f'http://localhost:{port}/v1',
            'model': self.get_current_model(),
            'max_tokens': 4096,
            'temperature': 0.2,
            'batch_size': 20,
        }

    def get_current_model(self):
        """Get the currently selected/loaded model name."""
        return self.model_var.get()

    def _on_tab_changed(self, event):
        """Refresh subtitle tab when switching to it."""
        selected = self.notebook.select()
        tab_index = self.notebook.index(selected)
        if tab_index == 1:  # Subtitle Translation tab
            self.subtitle_tab.refresh_model()

    def _create_server_tab(self):
        """Create server management UI in the Server tab."""
        main_frame = self.server_tab

        # 標題
        title_label = ttk.Label(
            main_frame,
            text="🚀 llama.cpp Manager",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 10))

        # 模型選擇區
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

        # 模型信息
        self.model_info_label = ttk.Label(model_frame, text="")
        self.model_info_label.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))

        # 服務器設置區
        server_frame = ttk.LabelFrame(main_frame, text="⚙️ 服務器設置", padding="10")
        server_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 創建參數輸入框
        params_grid = ttk.Frame(server_frame)
        params_grid.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Port
        ttk.Label(params_grid, text="端口:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.port_var = tk.IntVar(value=self.config_manager.get("server.port", 8080))
        ttk.Entry(params_grid, textvariable=self.port_var, width=10).grid(row=0, column=1, padx=(0, 20))

        # GPU Layers
        ttk.Label(params_grid, text="GPU 層數:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.gpu_layers_var = tk.IntVar(value=self.config_manager.get("server.gpu_layers", 99))
        ttk.Scale(
            params_grid,
            from_=0,
            to=99,
            variable=self.gpu_layers_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda v: self.gpu_layers_label.config(text=f"{int(float(v))}")
        ).grid(row=0, column=3, padx=(0, 5))
        self.gpu_layers_label = ttk.Label(params_grid, text="99")
        self.gpu_layers_label.grid(row=0, column=4, padx=(0, 20))

        # Context Size
        ttk.Label(params_grid, text="上下文大小:").grid(row=0, column=5, sticky=tk.W, padx=(0, 5))
        self.context_var = tk.IntVar(value=self.config_manager.get("server.context_size", 131072))
        context_combo = ttk.Combobox(
            params_grid,
            textvariable=self.context_var,
            values=[512, 1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072],
            width=10,
            state="readonly"
        )
        context_combo.grid(row=0, column=6, padx=(0, 20))

        # Batch Size
        ttk.Label(params_grid, text="批次大小:").grid(row=0, column=7, sticky=tk.W, padx=(0, 5))
        self.batch_var = tk.IntVar(value=self.config_manager.get("server.batch_size", 256))
        ttk.Entry(params_grid, textvariable=self.batch_var, width=10).grid(row=0, column=8)

        # 控制按鈕區
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, pady=(0, 10))

        self.start_button = ttk.Button(
            control_frame,
            text="▶️ 啟動服務器",
            command=self.start_server,
            width=20
        )
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(
            control_frame,
            text="⏹️ 停止服務器",
            command=self.stop_server,
            width=20,
            state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=5)

        self.release_button = ttk.Button(
            control_frame,
            text="🧹 釋放內存",
            command=self.release_memory,
            width=15,
            state="normal"
        )
        self.release_button.grid(row=0, column=2, padx=5)

        # 狀態顯示
        self.status_label = ttk.Label(
            control_frame,
            text="● 未運行",
            font=("Arial", 10)
        )
        self.status_label.grid(row=0, column=2, padx=20)

        # 日誌顯示區
        log_frame = ttk.LabelFrame(main_frame, text="📋 運行日誌", padding="10")
        log_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            height=15,
            font=("Consolas", 9)
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置日誌標籤顏色
        self.log_text.tag_config("INFO", foreground="blue")
        self.log_text.tag_config("ERROR", foreground="red")
        self.log_text.tag_config("SUCCESS", foreground="green")
        self.log_text.tag_config("WARNING", foreground="orange")

        # 資源監控
        self.resource_label = ttk.Label(
            main_frame,
            text="GPU: N/A | VRAM: N/A | RAM: N/A",
            font=("Consolas", 9)
        )
        self.resource_label.grid(row=5, column=0, pady=(10, 0), sticky=tk.W)

        # 刷新模型列表並恢復上次選擇
        self.refresh_model_list()

        # 恢復上次選擇的模型
        last_model = self.config_manager.get("ui.last_model", "")
        if last_model and last_model in self.model_combo['values']:
            self.model_var.set(last_model)
            self.on_model_select(None)

        # 綁定模型選擇事件
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_select)

    def refresh_model_list(self):
        """刷新模型列表"""
        model_list = self.models.list_models()
        model_names = [m.get("name", "Unknown") for m in model_list]
        self.model_combo['values'] = model_names

        if model_names and not self.model_var.get():
            self.model_combo.current(0)
            self.on_model_select(None)

    def on_model_select(self, event):
        """模型選擇事件處理"""
        model_name = self.model_var.get()

        # 記住選擇的模型
        if model_name:
            self.config_manager.set("ui.last_model", model_name)

        for model in self.models.list_models():
            if model.get("name") == model_name:
                info = f"大小: {model.get('size', 'N/A')} | 格式: {model.get('format', 'N/A')}"
                self.model_info_label.config(text=info)
                return

        self.model_info_label.config(text="")

    def add_model(self):
        """添加模型"""
        file_path = filedialog.askopenfilename(
            title="選擇模型文件",
            filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")],
            initialdir=str(self.hip_dir)
        )

        if file_path:
            model_info = self.models.add_model(file_path)
            self.refresh_model_list()
            self.log("SUCCESS", f"已添加模型: {model_info['name']}")

    def start_server(self):
        """啟動服務器"""
        if not self.model_var.get():
            messagebox.showerror("錯誤", "請先選擇一個模型！")
            return

        if self.server.running:
            messagebox.showwarning("警告", "服務器已在運行中！")
            return

        # 獲取模型路徑
        model_name = self.model_var.get()
        model_path = self.models.get_model_path(model_name)

        if not model_path or not Path(model_path).exists():
            messagebox.showerror("錯誤", f"找不到模型文件: {model_path}")
            return

        # 保存配置
        self.config_manager.set("server.port", self.port_var.get())
        self.config_manager.set("server.gpu_layers", self.gpu_layers_var.get())
        self.config_manager.set("server.context_size", self.context_var.get())
        self.config_manager.set("server.batch_size", self.batch_var.get())

        # 啟動服務器
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

            # 啟動資源監控
            self.start_resource_monitor()

            self.log("SUCCESS", f"服務器已啟動在 http://localhost:{self.port_var.get()}")

        except Exception as e:
            self.log("ERROR", f"啟動失敗: {str(e)}")
            messagebox.showerror("錯誤", f"啟動服務器失敗:\n{str(e)}")

    def stop_server(self):
        """停止服務器"""
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
        """釋放系統內存 (VRAM 和 RAM)"""
        try:
            self.log("INFO", "正在釋放系統內存...")

            # 強制垃圾回收
            gc.collect()

            # 終止所有 llama 相關進程
            killed_count = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    if proc_info['name']:
                        proc_name = proc_info['name'].lower()
                        # 檢查是否是 llama 相關進程
                        if 'llama' in proc_name or proc_name.endswith('.exe'):
                            # 跳過當前服務器進程
                            if self.server.running and self.server._process and proc_info['pid'] == self.server._process.pid:
                                continue

                            # 終止孤立進程
                            proc.terminate()
                            killed_count += 1
                            self.log("INFO", f"已終止進程: {proc_info['name']} (PID: {proc_info['pid']})")

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            # 強制清理系統緩存 (Windows)
            try:
                # 使用 EmptyStandbyList.exe 清理內存 (如果存在)
                subprocess.run(
                    ["EmptyStandbyList.exe", "standbylist"],
                    capture_output=True,
                    timeout=5
                )
                self.log("SUCCESS", "已清理系統待機列表")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # 顯示當前內存狀態
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
        """Callback from ServerController for each server stdout line."""
        self.log("INFO", line)

    def start_resource_monitor(self):
        """啟動資源監控"""
        self.monitor_running = True
        self.monitor_thread = threading.Thread(target=self.monitor_resources, daemon=True)
        self.monitor_thread.start()

    def stop_resource_monitor(self):
        """停止資源監控"""
        self.monitor_running = False

    def monitor_resources(self):
        """監控系統資源"""
        while self.monitor_running:
            try:
                # RAM 使用
                ram = psutil.virtual_memory()
                ram_used = ram.used / (1024**3)
                ram_total = ram.total / (1024**3)
                ram_percent = ram.percent

                # 檢查 llama-server 進程
                gpu_status = "未使用"
                vram_status = "N/A"
                process_info = ""

                if self.server.running and self.server._process:
                    try:
                        # 獲取進程信息
                        proc = psutil.Process(self.server._process.pid)
                        cpu_percent = proc.cpu_percent(interval=0.1)
                        mem_info = proc.memory_info()
                        proc_mem_mb = mem_info.rss / (1024**2)

                        # 計算進程數
                        try:
                            num_threads = proc.num_threads()
                        except:
                            num_threads = 0

                        process_info = f"服務器: CPU {cpu_percent:.1f}% | 內存 {proc_mem_mb:.0f}MB | 線程 {num_threads}"
                        gpu_status = "運行中"
                        vram_status = "已載入"

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # 更新 UI
                if process_info:
                    status_text = f"RAM: {ram_used:.1f}GB / {ram_total:.1f}GB ({ram_percent}%) | GPU: {gpu_status} | {process_info}"
                else:
                    status_text = f"RAM: {ram_used:.1f}GB / {ram_total:.1f}GB ({ram_percent}%) | GPU: {gpu_status} | VRAM: {vram_status}"

                self.root.after(0, lambda text=status_text: self.resource_label.config(text=text))

            except Exception as e:
                print(f"Resource monitor error: {e}")
                # 顯示簡化版本
                self.root.after(0, lambda: self.resource_label.config(
                    text=f"系統監控中... (錯誤: {str(e)[:30]})"
                ))

            time.sleep(2)

    def log(self, level, message):
        """添加日誌"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}\n"

        self.root.after(0, lambda: self._insert_log(log_message, level))

    def _insert_log(self, message, level):
        """插入日誌到文本框"""
        self.log_text.insert(tk.END, message, level)
        self.log_text.see(tk.END)

        # 限制日誌行數
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 1000:
            self.log_text.delete(1.0, f"{lines-1000}.0")

def main():
    root = TkinterDnD.Tk()
    app = LlamaManager(root)
    root.mainloop()

if __name__ == "__main__":
    main()
