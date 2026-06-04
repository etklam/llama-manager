"""
Subtitle Translation Tab - tkinter GUI for batch SRT/TXT translation.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tkinterdnd2 import DND_FILES

from utils.srt_parser import parse_srt_from_file, generate_srt_from_list
from translation.local_llm_translator import LocalLLMTranslator

from constants import SUPPORTED_SUBTITLE, TARGET_LANGUAGES, SOURCE_LANGUAGES
from ui_helpers import LogMixin, parse_dropped_paths
from config_helpers import build_translation_config


class SubtitleTranslationTab(LogMixin, ttk.Frame):
    """GUI tab for batch subtitle translation."""

    def __init__(self, parent, get_port_callback, get_model_callback, config_manager=None):
        super().__init__(parent)
        self._get_port = get_port_callback
        self._get_model = get_model_callback
        self._config_manager = config_manager
        self._translator: Optional[LocalLLMTranslator] = None
        self._file_list: list = []
        self._translating = False
        self._stop_requested = False

        self._init_translator()
        self._create_ui()

    def _build_translation_config(self) -> dict:
        return build_translation_config(
            self._config_manager, self._get_port(), self._get_model()
        )

    def _init_translator(self):
        """Initialize the translator with current config."""
        config = self._build_translation_config()
        self._translator = LocalLLMTranslator(config)

    def refresh_model(self):
        """Refresh from current server model."""
        self._init_translator()
        self._update_model_display()

    def load_file(self, filepath: str):
        """Load a single file into the file list."""
        if filepath not in self._file_list and self._is_supported_file(filepath):
            self._file_list.append(filepath)
            self._file_listbox.insert(tk.END, Path(filepath).name)

    # --- UI Creation ---

    def _create_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)

        # Row 0: File Selection
        self._create_file_section()

        # Row 1: Language & Model
        self._create_language_section()

        # Row 2: Options
        self._create_options_section()

        # Row 3: Controls
        self._create_control_section()

        # Row 4: Progress
        self._create_progress_section()

        # Row 5: Log
        self._create_log_section()

    def _create_file_section(self):
        frame = ttk.LabelFrame(self, text="File Selection", padding="5")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        frame.columnconfigure(0, weight=1)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=0, column=0, sticky=tk.W)

        ttk.Button(btn_frame, text="Choose Files",
                   command=self._choose_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear",
                   command=self._clear_files).pack(side=tk.LEFT, padx=2)

        self._replace_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btn_frame, text="Replace original",
                        variable=self._replace_var).pack(side=tk.LEFT, padx=10)

        # File listbox with scrollbar
        list_frame = ttk.Frame(frame)
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        list_frame.columnconfigure(0, weight=1)

        self._file_listbox = tk.Listbox(list_frame, height=4,
                                         selectmode=tk.EXTENDED)
        self._file_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Drag-and-drop support
        self._file_listbox.drop_target_register(DND_FILES)
        self._file_listbox.dnd_bind('<<Drop>>', self._on_files_dropped)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                  command=self._file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._file_listbox.config(yscrollcommand=scrollbar.set)

    def _create_language_section(self):
        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        # Source language
        ttk.Label(frame, text="Source:").pack(side=tk.LEFT, padx=(0, 2))
        last_source = self._config_manager.get("ui.last_source_lang", "auto") if self._config_manager else "auto"
        self._source_var = tk.StringVar(value=last_source)
        self._source_combo = ttk.Combobox(frame, textvariable=self._source_var,
                                           state="readonly", width=15)
        self._source_combo.pack(side=tk.LEFT, padx=(0, 10))

        # Arrow
        ttk.Label(frame, text="->").pack(side=tk.LEFT, padx=5)

        # Target language
        ttk.Label(frame, text="Target:").pack(side=tk.LEFT, padx=(0, 2))
        last_target = self._config_manager.get("ui.last_target_lang", "zh-cn") if self._config_manager else "zh-cn"
        self._target_var = tk.StringVar(value=last_target)
        self._target_combo = ttk.Combobox(frame, textvariable=self._target_var,
                                           state="readonly", width=15)
        self._target_combo.pack(side=tk.LEFT, padx=(0, 10))

        # Model display
        ttk.Label(frame, text="   Model:").pack(side=tk.LEFT, padx=(10, 2))
        self._model_label = ttk.Label(frame, text="(unknown)",
                                       foreground="blue")
        self._model_label.pack(side=tk.LEFT)

        self._populate_languages()
        self._update_model_display()

    def _create_options_section(self):
        self._adv_frame = ttk.LabelFrame(self, text="Advanced Options",
                                          padding="5")
        self._adv_frame.grid(row=2, column=0, sticky=(tk.W, tk.E),
                             pady=(0, 5))
        self._adv_frame.columnconfigure(1, weight=1)

        # Restore saved values or use defaults
        cm = self._config_manager
        saved_batch = cm.get("ui.batch_size", 15) if cm else 15
        saved_temp = cm.get("ui.temperature", 0.2) if cm else 0.2
        saved_tokens = cm.get("ui.max_tokens", 16384) if cm else 16384
        saved_workers = cm.get("ui.max_workers", 3) if cm else 3
        saved_fast = cm.get("ui.single_step", True) if cm else True

        # Row 0: Batch size
        ttk.Label(self._adv_frame, text="Batch Size:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5))
        self._batch_var = tk.IntVar(value=saved_batch)
        ttk.Scale(self._adv_frame, from_=1, to=50,
                  variable=self._batch_var, orient=tk.HORIZONTAL,
                  length=150).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(self._adv_frame, textvariable=self._batch_var).grid(
            row=0, column=2, padx=(5, 20))

        # Row 0: Temperature
        ttk.Label(self._adv_frame, text="Temperature:").grid(
            row=0, column=3, sticky=tk.W, padx=(0, 5))
        self._temp_var = tk.DoubleVar(value=saved_temp)
        ttk.Scale(self._adv_frame, from_=0.0, to=2.0,
                  variable=self._temp_var, orient=tk.HORIZONTAL,
                  length=150).grid(row=0, column=4, sticky=tk.W)
        ttk.Label(self._adv_frame,
                  textvariable=self._temp_var).grid(row=0, column=5, padx=5)

        # Row 1: Max tokens
        ttk.Label(self._adv_frame, text="Max Tokens:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 5))
        self._tokens_var = tk.IntVar(value=saved_tokens)
        tokens_combo = ttk.Combobox(
            self._adv_frame, textvariable=self._tokens_var,
            values=[512, 1024, 2048, 4096, 8192, 16384, 32768],
            width=10, state="readonly")
        tokens_combo.grid(row=1, column=1, sticky=tk.W)

        # Row 1: Workers
        ttk.Label(self._adv_frame, text="并发数:").grid(
            row=1, column=3, sticky=tk.W, padx=(20, 5))
        self._workers_var = tk.IntVar(value=saved_workers)
        workers_scale = ttk.Scale(self._adv_frame, from_=1, to=8,
                  variable=self._workers_var, orient=tk.HORIZONTAL,
                  length=120)
        workers_scale.grid(row=1, column=4, sticky=tk.W)
        self._workers_label = ttk.Label(self._adv_frame, text=str(saved_workers))
        self._workers_label.grid(row=1, column=5, padx=5)
        workers_scale.configure(command=lambda v: self._workers_label.config(
            text=str(int(float(v)))))

        # Row 2: Fast mode checkbox
        self._fast_mode_var = tk.BooleanVar(value=saved_fast)
        ttk.Checkbutton(self._adv_frame, text="快速模式 (跳过直译, 仅意译)",
                        variable=self._fast_mode_var).grid(
            row=2, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))

        self._adv_frame.grid_remove()  # Hidden by default

    def _create_control_section(self):
        frame = ttk.Frame(self)
        frame.grid(row=3, column=0, pady=(0, 5), sticky=tk.W)

        self._toggle_adv_btn = ttk.Button(frame, text="Show Advanced",
                                          command=self._toggle_advanced)
        self._toggle_adv_btn.pack(side=tk.LEFT, padx=2)

        self._start_btn = ttk.Button(frame, text="Start Translation",
                                     command=self._start_translation)
        self._start_btn.pack(side=tk.LEFT, padx=10)

        self._stop_btn = ttk.Button(frame, text="Stop",
                                    command=self._stop_translation,
                                    state="disabled")
        self._stop_btn.pack(side=tk.LEFT, padx=2)

    def _create_progress_section(self):
        frame = ttk.Frame(self)
        frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        frame.columnconfigure(0, weight=1)

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            frame, variable=self._progress_var, maximum=100)
        self._progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self._progress_label = ttk.Label(frame, text="Ready")
        self._progress_label.grid(row=1, column=0, sticky=tk.W)

        self._init_log_widget(self, row=5, column=0)

    # --- Language population ---

    def _populate_languages(self):
        langs = SOURCE_LANGUAGES
        codes = list(langs.keys())
        names = [f"{code} - {langs[code]}" for code in codes]
        self._source_combo['values'] = names

        # Restore saved source language
        saved_source = self._source_var.get()
        source_found = False
        for i, name in enumerate(names):
            if name.startswith(saved_source):
                self._source_combo.current(i)
                source_found = True
                break
        if not source_found:
            self._source_combo.current(0)

        langs = TARGET_LANGUAGES
        codes = list(langs.keys())
        names = [f"{code} - {langs[code]}" for code in codes]
        self._target_combo['values'] = names

        # Restore saved target language
        saved_target = self._target_var.get()
        target_found = False
        for i, name in enumerate(names):
            if name.startswith(saved_target):
                self._target_combo.current(i)
                target_found = True
                break
        if not target_found:
            for i, name in enumerate(names):
                if name.startswith('zh-cn'):
                    self._target_combo.current(i)
                    break

        # Bind save on language change
        self._source_combo.bind('<<ComboboxSelected>>', self._on_language_changed)
        self._target_combo.bind('<<ComboboxSelected>>', self._on_language_changed)

    def _update_model_display(self):
        model = self._get_model()
        if model:
            self._model_label.config(text=model)
        else:
            self._model_label.config(text="(no model loaded)")

    def _on_language_changed(self, event=None):
        """Save language selections when user changes them."""
        if self._config_manager:
            source_code = self._get_source_code()
            target_code = self._get_target_code()
            if source_code:
                self._config_manager.set("ui.last_source_lang", source_code)
            if target_code:
                self._config_manager.set("ui.last_target_lang", target_code)

    def _get_target_code(self):
        val = self._target_var.get()
        return val.split(' - ')[0] if ' - ' in val else val

    def _get_source_code(self):
        val = self._source_var.get()
        return val.split(' - ')[0] if ' - ' in val else val

    # --- Actions ---

    def _choose_files(self):
        files = filedialog.askopenfilenames(
            title="Select SRT/TXT files",
            filetypes=[("Subtitle files", "*.srt;*.txt"),
                       ("All files", "*.*")])
        for f in files:
            if f not in self._file_list and self._is_supported_file(f):
                self._file_list.append(f)
                self._file_listbox.insert(tk.END, Path(f).name)
        self._log("INFO", f"Added {len(files)} file(s)")

    def _clear_files(self):
        self._file_list.clear()
        self._file_listbox.delete(0, tk.END)

    def _on_files_dropped(self, event):
        """Handle drag-and-drop of files onto the listbox."""
        raw_data = event.data
        # tkinterdnd2 on Windows wraps paths in {} and separates with space
        # Parse out individual file paths
        paths = parse_dropped_paths(raw_data)
        added = 0
        for path in paths:
            path = path.strip()
            if not path:
                continue
            # On Windows, paths may have surrounding braces removed
            if path not in self._file_list and self._is_supported_file(path):
                self._file_list.append(path)
                self._file_listbox.insert(tk.END, Path(path).name)
                added += 1
        self._log("INFO", f"Added {added} file(s) via drag-and-drop")

    def _toggle_advanced(self):
        if self._adv_frame.winfo_ismapped():
            self._adv_frame.grid_remove()
            self._toggle_adv_btn.config(text="Show Advanced")
        else:
            self._adv_frame.grid()
            self._toggle_adv_btn.config(text="Hide Advanced")

    def _start_translation(self):
        if not self._file_list:
            messagebox.showwarning("Warning", "Please select files first!")
            return

        if self._translating:
            return

        self._translating = True
        self._stop_requested = False
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._file_listbox.config(state="disabled")
        self._progress_var.set(0)
        self._clear_log()

        # Update translator config
        port = self._get_port()
        config = build_translation_config(
            self._config_manager, port, self._get_model()
        )
        config['batch_size'] = self._batch_var.get()
        config['temperature'] = self._temp_var.get()
        config['max_tokens'] = self._tokens_var.get()
        config['max_workers'] = int(float(self._workers_var.get()))
        config['single_step'] = self._fast_mode_var.get()
        self._translator = LocalLLMTranslator(config)

        if self._config_manager:
            self._config_manager.set("ui.batch_size", config['batch_size'])
            self._config_manager.set("ui.temperature", config['temperature'])
            self._config_manager.set("ui.max_tokens", config['max_tokens'])
            self._config_manager.set("ui.max_workers", config['max_workers'])
            self._config_manager.set("ui.single_step", config['single_step'])

        thread = threading.Thread(target=self._run_translation, daemon=True)
        thread.start()

    def _stop_translation(self):
        self._stop_requested = True
        self._log("WARNING", "Stopping after current file...")

    def _run_translation(self):
        total = len(self._file_list)
        target_lang = self._get_target_code()
        replace = self._replace_var.get()

        for i, filepath in enumerate(self._file_list):
            if self._stop_requested:
                self._log("WARNING", "Translation stopped by user")
                break

            try:
                self._translate_file(filepath, target_lang, replace_original=replace)
            except Exception as e:
                self._log("ERROR", f"Failed: {Path(filepath).name} - {e}")

            pct = ((i + 1) / total) * 100
            self.winfo_toplevel().after(0, lambda p=pct: self._progress_var.set(p))

        self.winfo_toplevel().after(0, self._on_translation_done)

    def _on_translation_done(self):
        self._translating = False
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._file_listbox.config(state="normal")
        self._progress_label.config(text="Completed")

        if self._stop_requested:
            self._log("INFO", "Translation stopped")
        else:
            self._log("SUCCESS", "All files translated!")

    # --- Helper methods ---

    def _is_supported_file(self, filepath: str) -> bool:
        """Check if file has supported extension."""
        ext = Path(filepath).suffix.lower()
        return ext in SUPPORTED_SUBTITLE

    def _get_output_path(self, input_path: str, target_lang: str,
                         replace_original: bool = False) -> str:
        """Generate output path for translated file."""
        if replace_original:
            return input_path
        p = Path(input_path)
        stem = p.stem
        lang_name = TARGET_LANGUAGES.get(target_lang, target_lang)
        return str(p.parent / f"{stem}_{lang_name}{p.suffix}")

    def _translate_file(self, input_path: str, target_lang: str,
                        output_path: Optional[str] = None,
                        replace_original: bool = False):
        """Translate a single SRT or TXT file."""
        filepath = Path(input_path)
        ext = filepath.suffix.lower()

        if output_path is None:
            output_path = self._get_output_path(input_path, target_lang,
                                               replace_original)

        self._log("INFO", f"Translating: {filepath.name}")
        self._on_progress(filepath.name, 0, 1, "starting")

        if ext == '.srt':
            subtitles = parse_srt_from_file(str(filepath))
            total_lines = len(subtitles)
            self._log("INFO", f"Parsed {total_lines} subtitle lines, "
                              f"translating line by line...")

            def file_progress(current, total, status):
                self._on_progress(filepath.name, current, total, status)

            def log_callback(level, message):
                self._log(level, message)

            translated = self._translator.translate_srt(
                subtitles, target_lang,
                progress_callback=file_progress,
                log_callback=log_callback
            )
            srt_content = generate_srt_from_list(translated)
            Path(output_path).write_text(srt_content, encoding='utf-8')
        elif ext == '.txt':
            text = filepath.read_text(encoding='utf-8')
            self._log("INFO", f"Text file, {len(text)} chars")
            translated = self._translator.translate(text, target_lang)
            Path(output_path).write_text(translated, encoding='utf-8')

        self._on_progress(filepath.name, 1, 1, "completed")
        self._log("SUCCESS", f"Saved: {output_path}")

    def _on_progress(self, file_name, current, total, status):
        """Handle progress updates from translator."""
        self.winfo_toplevel().after(0, lambda: self._progress_label.config(
            text=f"{status}: {file_name} ({current}/{total})"))
