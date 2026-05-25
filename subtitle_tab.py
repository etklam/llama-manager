"""
Subtitle Translation Tab - tkinter GUI for batch SRT/TXT translation.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tkinterdnd2 import DND_FILES

from utils.srt_parser import parse_srt_from_file, generate_srt_from_list
from translation.local_llm_translator import LocalLLMTranslator


# Supported file extensions
SUPPORTED_EXTENSIONS = {'.srt', '.txt'}

# Hardcoded common languages
TARGET_LANGUAGES = {
    'zh-cn': 'Simplified Chinese',
    'zh-tw': 'Traditional Chinese',
    'en': 'English',
    'ja': 'Japanese',
    'ko': 'Korean',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'it': 'Italian',
    'nl': 'Dutch',
}

SOURCE_LANGUAGES = {
    'auto': 'Auto Detect',
    **TARGET_LANGUAGES,
}


class SubtitleTranslationTab(ttk.Frame):
    """GUI tab for batch subtitle translation."""

    def __init__(self, parent, get_config_callback, get_model_callback):
        super().__init__(parent)
        self._get_config = get_config_callback
        self._get_model = get_model_callback
        self._translator: Optional[LocalLLMTranslator] = None
        self._file_list: list = []
        self._translating = False
        self._stop_requested = False

        self._init_translator()
        self._create_ui()

    def _init_translator(self):
        """Initialize the translator with current config."""
        config = self._get_config()
        model = self._get_model()
        config['model'] = model
        self._translator = LocalLLMTranslator(config)

    def refresh_model(self):
        """Refresh from current server model."""
        self._init_translator()
        self._update_model_display()

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
        self._source_var = tk.StringVar(value="auto")
        self._source_combo = ttk.Combobox(frame, textvariable=self._source_var,
                                           state="readonly", width=15)
        self._source_combo.pack(side=tk.LEFT, padx=(0, 10))

        # Arrow
        ttk.Label(frame, text="->").pack(side=tk.LEFT, padx=5)

        # Target language
        ttk.Label(frame, text="Target:").pack(side=tk.LEFT, padx=(0, 2))
        self._target_var = tk.StringVar(value="zh-cn")
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

        # Row 0: Batch size
        ttk.Label(self._adv_frame, text="Batch Size:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5))
        self._batch_var = tk.IntVar(value=5)
        ttk.Scale(self._adv_frame, from_=1, to=50,
                  variable=self._batch_var, orient=tk.HORIZONTAL,
                  length=150).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(self._adv_frame, textvariable=self._batch_var).grid(
            row=0, column=2, padx=(5, 20))

        # Row 0: Temperature
        ttk.Label(self._adv_frame, text="Temperature:").grid(
            row=0, column=3, sticky=tk.W, padx=(0, 5))
        self._temp_var = tk.DoubleVar(value=0.2)
        ttk.Scale(self._adv_frame, from_=0.0, to=2.0,
                  variable=self._temp_var, orient=tk.HORIZONTAL,
                  length=150).grid(row=0, column=4, sticky=tk.W)
        ttk.Label(self._adv_frame,
                  textvariable=self._temp_var).grid(row=0, column=5, padx=5)

        # Row 1: Max tokens
        ttk.Label(self._adv_frame, text="Max Tokens:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 5))
        self._tokens_var = tk.IntVar(value=16384)
        tokens_combo = ttk.Combobox(
            self._adv_frame, textvariable=self._tokens_var,
            values=[512, 1024, 2048, 4096, 8192, 16384, 32768],
            width=10, state="readonly")
        tokens_combo.grid(row=1, column=1, sticky=tk.W)

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

    def _create_log_section(self):
        frame = ttk.LabelFrame(self, text="Log", padding="5")
        frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self._log_text = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD, height=10,
            font=("Consolas", 9), state="disabled")
        self._log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure log tags
        self._log_text.tag_config("INFO", foreground="blue")
        self._log_text.tag_config("ERROR", foreground="red")
        self._log_text.tag_config("SUCCESS", foreground="green")
        self._log_text.tag_config("WARNING", foreground="orange")

    # --- Language population ---

    def _populate_languages(self):
        langs = SOURCE_LANGUAGES
        codes = list(langs.keys())
        names = [f"{code} - {langs[code]}" for code in codes]
        self._source_combo['values'] = names
        self._source_combo.current(0)

        langs = TARGET_LANGUAGES
        codes = list(langs.keys())
        names = [f"{code} - {langs[code]}" for code in codes]
        self._target_combo['values'] = names
        # Default to zh-cn
        for i, name in enumerate(names):
            if name.startswith('zh-cn'):
                self._target_combo.current(i)
                break

    def _update_model_display(self):
        model = self._get_model()
        if model:
            self._model_label.config(text=model)
        else:
            self._model_label.config(text="(no model loaded)")

    def _get_target_code(self):
        val = self._target_var.get()
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
        paths = self._parse_dropped_paths(raw_data)
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

    @staticmethod
    def _parse_dropped_paths(raw_data):
        """Parse file paths from tkinterdnd2 drop data on Windows."""
        import re
        # On Windows, paths are like: {C:/path/to/file1.srt} {C:/path/to/file2.txt}
        paths = re.findall(r'\{([^}]+)\}', raw_data)
        if not paths:
            # Try without braces (Linux/mac style)
            paths = raw_data.split()
        return paths

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
        config = self._get_config()
        config['model'] = self._get_model()
        config['batch_size'] = self._batch_var.get()
        config['temperature'] = self._temp_var.get()
        config['max_tokens'] = self._tokens_var.get()
        self._translator = LocalLLMTranslator(config)

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

    # --- Logging ---

    def _log(self, level, message):
        self.winfo_toplevel().after(0, lambda: self._insert_log(level, message))

    def _insert_log(self, level, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] [{level}] {message}\n"
        self._log_text.config(state="normal")
        self._log_text.insert(tk.END, line, level)
        self._log_text.see(tk.END)
        self._log_text.config(state="disabled")

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete(1.0, tk.END)
        self._log_text.config(state="disabled")

    # --- Callbacks from logic ---

    def _on_log(self, level, message):
        self._log(level, message)

    # --- Helper methods (absorbed from SubtitleTranslationLogic) ---

    def _is_supported_file(self, filepath: str) -> bool:
        """Check if file has supported extension."""
        ext = Path(filepath).suffix.lower()
        return ext in SUPPORTED_EXTENSIONS

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
