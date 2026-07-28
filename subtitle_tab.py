"""
Subtitle Translation Tab - tkinter GUI for batch SRT/TXT translation.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tkinterdnd2 import DND_FILES

from utils.srt_parser import parse_srt_from_file, generate_srt_from_list, output_path_for
from translation.local_llm_translator import LocalLLMTranslator
from translation.server_probe import (
    clamp_workers, probe_server, unreachable_message,
)

from constants import SUPPORTED_SUBTITLE, TARGET_LANGUAGES, SOURCE_LANGUAGES
from ui_helpers import (
    CHANNEL_TRANSLATE, LogMixin, populate_language_combo, extract_combo_code,
)
from config_helpers import build_translation_config
from file_listbox import FileListbox


class SubtitleTranslationTab(LogMixin, ttk.Frame):
    """GUI tab for batch subtitle translation."""

    log_channel = CHANNEL_TRANSLATE

    def __init__(self, parent, get_port_callback, get_model_callback, config_manager=None):
        super().__init__(parent)
        self._get_port = get_port_callback
        self._get_model = get_model_callback
        self._config_manager = config_manager
        self._translator: Optional[LocalLLMTranslator] = None
        self._file_list: list = []
        self._translating = False
        self._stop_requested = False
        # Progress state read by _on_progress (set per-run in _run_translation).
        self._current_file_idx = 0
        self._total_files = 1
        self._file_start_time = time.time()

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
        self._sync_workers_from_config()

    def _sync_workers_from_config(self):
        """Pull ui.max_workers back into the slider.

        A Server-tab preset writes the slot count it configured into
        ui.max_workers, but this tab only read that key when it was built, so the
        slider kept showing a stale number until restart. This runs on tab
        switch, which is the point where the user would look at it.
        """
        if not self._config_manager:
            return
        saved = self._config_manager.get("ui.max_workers", None)
        if not isinstance(saved, int) or saved < 1:
            return
        self._workers_var.set(saved)
        self._workers_label.config(text=str(saved))

    def load_file(self, filepath: str):
        """Load a single file into the file list."""
        self._file_listbox_widget.add(filepath)

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

        # Row 4: Progress & Log
        self._create_progress_section()

    def _create_file_section(self):
        frame = ttk.LabelFrame(self, text="File Selection", padding="5")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        frame.columnconfigure(0, weight=1)

        self._file_listbox_widget = FileListbox(
            frame, valid_extensions=SUPPORTED_SUBTITLE,
            filetypes_label="Subtitle files",
            filetypes_exts=[".srt", ".txt"],
            on_change=self._sync_file_list,
        )
        self._file_listbox_widget.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self._file_listbox = self._file_listbox_widget.listbox

        # ponytail: extra control rides along the widget's existing btn row.
        self._replace_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self._file_listbox_widget.button_row,
                        text="Replace original",
                        variable=self._replace_var).pack(side=tk.LEFT, padx=10)

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
        batch_scale = ttk.Scale(self._adv_frame, from_=1, to=50,
                                variable=self._batch_var, orient=tk.HORIZONTAL,
                                length=150)
        batch_scale.grid(row=0, column=1, sticky=tk.W)
        # A Scale writes doubles into its variable even when it is an IntVar, so
        # a bound textvariable renders "23.548387096774196". Format for display
        # and leave the var alone — IntVar.get() already truncates for us.
        self._batch_label = ttk.Label(self._adv_frame, text=str(saved_batch))
        self._batch_label.grid(row=0, column=2, padx=(5, 20))
        batch_scale.configure(command=lambda v: self._batch_label.config(
            text=str(int(float(v)))))

        # Row 0: Temperature
        ttk.Label(self._adv_frame, text="Temperature:").grid(
            row=0, column=3, sticky=tk.W, padx=(0, 5))
        self._temp_var = tk.DoubleVar(value=saved_temp)
        temp_scale = ttk.Scale(self._adv_frame, from_=0.0, to=2.0,
                               variable=self._temp_var, orient=tk.HORIZONTAL,
                               length=150)
        temp_scale.grid(row=0, column=4, sticky=tk.W)
        # Round for display only: bound straight to the DoubleVar, a drag shows
        # the raw float ("0.36129032258064515") and swamps the row.
        self._temp_label = ttk.Label(self._adv_frame, text=f"{saved_temp:.2f}")
        self._temp_label.grid(row=0, column=5, padx=5)
        temp_scale.configure(command=lambda v: self._temp_label.config(
            text=f"{float(v):.2f}"))

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
        last_source = self._source_var.get()
        populate_language_combo(self._source_combo, SOURCE_LANGUAGES, last_source)

        last_target = self._target_var.get()
        populate_language_combo(self._target_combo, TARGET_LANGUAGES, last_target)

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
        return extract_combo_code(self._target_var.get())

    def _get_source_code(self):
        return extract_combo_code(self._source_var.get())

    # --- Actions ---

    def _sync_file_list(self):
        """Mirror the widget's file list so _start_translation reads it."""
        self._file_list = self._file_listbox_widget.files

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

        # Read every Tk variable here, on the UI thread, and hand the plain dict
        # to the worker: the preflight below runs off-thread, and Tk variables
        # are not safe to touch from there.
        config = build_translation_config(
            self._config_manager, self._get_port(), self._get_model()
        )
        config['batch_size'] = self._batch_var.get()
        config['temperature'] = self._temp_var.get()
        config['max_tokens'] = self._tokens_var.get()
        config['single_step'] = self._fast_mode_var.get()
        requested_workers = int(float(self._workers_var.get()))
        config['max_workers'] = requested_workers

        self._translating = True
        self._stop_requested = False
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._file_listbox.config(state="disabled")
        self._progress_var.set(0)
        self._clear_log()

        if self._config_manager:
            self._config_manager.set("ui.batch_size", config['batch_size'])
            self._config_manager.set("ui.temperature", config['temperature'])
            self._config_manager.set("ui.max_tokens", config['max_tokens'])
            # Persist what the user asked for, not the value the clamp settles
            # on: the clamp reflects the server that happens to be running now,
            # and saving it would silently ratchet the slider down after one run
            # against a single-slot server.
            self._config_manager.set("ui.max_workers", requested_workers)
            self._config_manager.set("ui.single_step", config['single_step'])

        thread = threading.Thread(
            target=self._preflight_and_translate, args=(config,), daemon=True)
        thread.start()

    def _preflight_and_translate(self, config: dict):
        """Probe the server, then run the batch. Runs on the worker thread.

        The probe is here rather than in _start_translation because it is a
        network call: against a server that is down it costs about a second, and
        on the UI thread that reads as the window locking up on the click.
        """
        # Without this check, an unreachable server is discovered one batch at a
        # time: each batch spends three tenacity attempts with exponential
        # backoff before failing, so a long SRT takes minutes to report what was
        # knowable before the first request.
        info = probe_server(config['api_url'])
        if not info.reachable:
            message = unreachable_message(config['api_url'], info)
            self._log("ERROR", message)
            self.winfo_toplevel().after(0, lambda: (
                messagebox.showerror("Server not reachable", message),
                self._on_translation_aborted(),
            ))
            return

        # Align client concurrency with the server's real slot count. Each worker
        # holds one request open and llama-server runs at most total_slots of
        # them at once, so workers beyond that simply queue: throughput equal to
        # one worker, while the log claims parallelism.
        workers, note = clamp_workers(config['max_workers'], info)
        config['max_workers'] = workers
        if note:
            self._log("WARNING", note)

        self._translator = LocalLLMTranslator(config)
        self._run_translation()

    def _on_translation_aborted(self):
        """Return the controls to idle after a run that never started."""
        self._translating = False
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._file_listbox.config(state="normal")
        self._progress_label.config(text="Ready")

    def _stop_translation(self):
        self._stop_requested = True
        self._log("WARNING", "Stopping after current file...")

    def _run_translation(self):
        total = len(self._file_list)
        target_lang = self._get_target_code()
        replace = self._replace_var.get()

        # File-level context read by _on_progress to compute overall percentage
        # (file index + within-file line fraction) and per-file ETA.
        self._total_files = total
        self._file_start_time = time.time()

        for i, filepath in enumerate(self._file_list):
            if self._stop_requested:
                self._log("WARNING", "Translation stopped by user")
                break

            self._current_file_idx = i
            self._file_start_time = time.time()

            try:
                self._translate_file(filepath, target_lang, replace_original=replace)
            except Exception as e:
                self._log("ERROR", f"Failed: {Path(filepath).name} - {e}")

            # Snap the bar to the whole-file boundary once the file finishes;
            # within-file advancement is driven by _on_progress at line level.
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

    def _translate_file(self, input_path: str, target_lang: str,
                        output_path: Optional[str] = None,
                        replace_original: bool = False):
        """Translate a single SRT or TXT file."""
        filepath = Path(input_path)
        ext = filepath.suffix.lower()

        if output_path is None:
            output_path = output_path_for(input_path, target_lang, replace_original)

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
        """Handle progress updates from the translator.

        Advances the progress bar at line granularity: overall percentage is
        (completed files + current file's line fraction) / total files, so a
        single large SRT no longer freezes the bar. Also derives lines/s and an
        ETA for the current file from the wall-clock elapsed since it started.
        """
        total = max(1, total)
        file_frac = min(current, total) / total

        file_idx = getattr(self, "_current_file_idx", 0)
        total_files = max(1, getattr(self, "_total_files", 1))
        overall_pct = ((file_idx + file_frac) / total_files) * 100

        elapsed = time.time() - getattr(self, "_file_start_time", time.time())
        detail = f"{status}" if status else ""
        if current > 0 and elapsed > 0:
            rate = current / elapsed
            remaining = (total - current) / rate if rate > 0 else 0
            eta = self._format_eta(remaining)
            detail = f"{rate:.1f} lines/s · ETA {eta}"

        label = f"{file_name} ({current}/{total})"
        if detail:
            label = f"{label} · {detail}"

        self.winfo_toplevel().after(0, lambda p=overall_pct, t=label: (
            self._progress_var.set(p),
            self._progress_label.config(text=t),
        ))

    @staticmethod
    def _format_eta(seconds: float) -> str:
        """Format a seconds count as mm:ss (or h:mm:ss when over an hour)."""
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
