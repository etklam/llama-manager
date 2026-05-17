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
import json
import os
import sys
import time
import psutil
from pathlib import Path
from datetime import datetime

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

        # 配置文件
        self.config_file = Path(__file__).parent / "config.json"
        self.models_file = Path(__file__).parent / "models.json"

        # 服務器進程
        self.server_process = None
        self.server_running = False
        self.log_buffer = []

        # 加載配置
        self.config = self.load_config()
        self.models = self.load_models()

        # 創建 UI
        self.create_ui()

        # 啟動監控線程
        self.monitor_thread = None
        self.monitor_running = False

        # 自動掃描模型
        self.scan_models()

    def load_config(self):
        """加載配置文件"""
        default_config = {
            "server": {
                "port": 8080,
                "host": "0.0.0.0",
                "gpu_layers": 99,
                "context_size": 4096,
                "batch_size": 512,
                "threads": -1
            },
            "ui": {
                "theme": "default",
                "auto_scroll": True
            }
        }

        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")

        return default_config

    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")

    def load_models(self):
        """加載模型列表"""
        if self.models_file.exists():
            try:
                with open(self.models_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading models: {e}")

        return {"models": []}

    def save_models(self):
        """保存模型列表"""
        try:
            with open(self.models_file, 'w', encoding='utf-8') as f:
                json.dump(self.models, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving models: {e}")

    def scan_models(self):
        """掃描 llama-hip 目錄中的 .gguf 文件"""
        if not self.hip_dir.exists():
            self.log("WARNING", f"目錄不存在: {self.hip_dir}")
            return

        gguf_files = list(self.hip_dir.glob("*.gguf"))
        existing_paths = {str(gguf_file) for gguf_file in gguf_files}

        # 獲取當前模型列表
        current_models = self.models.get("models", [])
        models_to_keep = []
        removed_count = 0
        added_count = 0

        # 檢查現有模型是否還存在
        for model in current_models:
            model_path = model.get("path", "")
            if model_path and Path(model_path).exists():
                models_to_keep.append(model)
            else:
                removed_count += 1
                self.log("INFO", f"移除不存在的模型: {model.get('name', 'Unknown')}")

        # 添加新發現的模型
        for gguf_file in gguf_files:
            if str(gguf_file) not in {m.get("path", "") for m in current_models}:
                size_gb = gguf_file.stat().st_size / (1024**3)
                model_info = {
                    "name": gguf_file.stem,
                    "path": str(gguf_file),
                    "size": f"{size_gb:.2f}GB",
                    "format": self.detect_format(gguf_file.name)
                }
                models_to_keep.append(model_info)
                added_count += 1
                self.log("SUCCESS", f"發現新模型: {model_info['name']} ({model_info['size']})")

        # 更新模型列表
        self.models["models"] = models_to_keep
        self.save_models()
        self.refresh_model_list()

        # 顯示掃描結果
        total_count = len(models_to_keep)
        if removed_count > 0 or added_count > 0:
            self.log("INFO", f"掃描完成: 共 {total_count} 個模型 (+{added_count}, -{removed_count})")
        else:
            self.log("INFO", f"掃描完成: 共 {total_count} 個模型 (無變更)")

    def detect_format(self, filename):
        """從文件名檢測量化格式"""
        formats = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0",
                  "IQ4_NL", "IQ4_XS", "Q3_K_M", "Q2_K"]
        for fmt in formats:
            if fmt.lower() in filename.lower():
                return fmt
        return "Unknown"

    def create_ui(self):
        """創建主界面"""
        # 創建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置網格權重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

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
        self.port_var = tk.IntVar(value=self.config["server"]["port"])
        ttk.Entry(params_grid, textvariable=self.port_var, width=10).grid(row=0, column=1, padx=(0, 20))

        # GPU Layers
        ttk.Label(params_grid, text="GPU 層數:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.gpu_layers_var = tk.IntVar(value=self.config["server"]["gpu_layers"])
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
        self.context_var = tk.IntVar(value=self.config["server"]["context_size"])
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
        self.batch_var = tk.IntVar(value=self.config["server"]["batch_size"])
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

        # 刷新模型列表
        self.refresh_model_list()

        # 綁定模型選擇事件
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_select)

    def refresh_model_list(self):
        """刷新模型列表"""
        models = self.models.get("models", [])
        model_names = [m.get("name", "Unknown") for m in models]
        self.model_combo['values'] = model_names

        if model_names and not self.model_var.get():
            self.model_combo.current(0)
            self.on_model_select(None)

    def on_model_select(self, event):
        """模型選擇事件處理"""
        model_name = self.model_var.get()
        models = self.models.get("models", [])

        for model in models:
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
            gguf_file = Path(file_path)
            size_gb = gguf_file.stat().st_size / (1024**3)

            model_info = {
                "name": gguf_file.stem,
                "path": str(gguf_file),
                "size": f"{size_gb:.2f}GB",
                "format": self.detect_format(gguf_file.name)
            }

            self.models.setdefault("models", []).append(model_info)
            self.save_models()
            self.refresh_model_list()

            self.log("SUCCESS", f"已添加模型: {model_info['name']}")

    def start_server(self):
        """啟動服務器"""
        if not self.model_var.get():
            messagebox.showerror("錯誤", "請先選擇一個模型！")
            return

        if self.server_running:
            messagebox.showwarning("警告", "服務器已在運行中！")
            return

        # 獲取模型路徑
        model_name = self.model_var.get()
        model_path = None
        for model in self.models.get("models", []):
            if model.get("name") == model_name:
                model_path = model.get("path")
                break

        if not model_path or not Path(model_path).exists():
            messagebox.showerror("錯誤", f"找不到模型文件: {model_path}")
            return

        # 構建命令
        cmd = [
            str(self.server_exe),
            "-m", model_path,
            "--port", str(self.port_var.get()),
            "--host", self.config["server"]["host"],
            "-ngl", str(self.gpu_layers_var.get()),
            "-c", str(self.context_var.get()),
            "-b", str(self.batch_var.get())
        ]

        # 保存配置
        self.config["server"]["port"] = self.port_var.get()
        self.config["server"]["gpu_layers"] = self.gpu_layers_var.get()
        self.config["server"]["context_size"] = self.context_var.get()
        self.config["server"]["batch_size"] = self.batch_var.get()
        self.save_config()

        # 啟動服務器
        try:
            self.log("INFO", f"啟動服務器: {model_name}")
            self.log("INFO", f"命令: {' '.join(cmd)}")

            # 創建環境變量以設置編碼
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace',  # 替換無法解碼的字節
                env=env
            )

            self.server_running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_label.config(text="● 運行中", foreground="green")

            # 啟動日誌監控線程
            self.log_thread = threading.Thread(target=self.monitor_server_logs, daemon=True)
            self.log_thread.start()

            # 啟動資源監控
            self.start_resource_monitor()

            self.log("SUCCESS", f"服務器已啟動在 http://localhost:{self.port_var.get()}")

        except Exception as e:
            self.log("ERROR", f"啟動失敗: {str(e)}")
            messagebox.showerror("錯誤", f"啟動服務器失敗:\n{str(e)}")

    def stop_server(self):
        """停止服務器"""
        if not self.server_running:
            return

        try:
            if self.server_process:
                self.server_process.terminate()

                # 等待進程結束
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.server_process.kill()
                    self.server_process.wait()

            self.server_running = False
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
            import gc
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
                            # 檢查是否是我們的進程
                            if self.server_process and proc_info['pid'] == self.server_process.pid:
                                continue  # 跳過當前服務器進程

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

    def monitor_server_logs(self):
        """監控服務器日誌"""
        if not self.server_process:
            return

        try:
            for line in iter(self.server_process.stdout.readline, ''):
                if not line:
                    break

                line = line.strip()
                if line:
                    self.log("INFO", line)

        except Exception as e:
            if self.server_running:
                self.log("ERROR", f"日誌監控錯誤: {str(e)}")

        # 服務器進程結束
        self.root.after(0, self.on_server_stopped)

    def on_server_stopped(self):
        """服務器停止回調"""
        if self.server_running:
            self.server_running = False
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.status_label.config(text="● 未運行", foreground="black")
            self.stop_resource_monitor()
            self.log("WARNING", "服務器進程已意外停止")

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

                if self.server_process and self.server_running:
                    try:
                        # 獲取進程信息
                        proc = psutil.Process(self.server_process.pid)
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
    root = tk.Tk()
    app = LlamaManager(root)
    root.mainloop()

if __name__ == "__main__":
    main()
