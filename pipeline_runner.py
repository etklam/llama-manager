"""PipelineRunner - Orchestrates Whisper speech-to-text followed by subtitle translation."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

from whisper_controller import WhisperController
from translation.local_llm_translator import LocalLLMTranslator
from utils.srt_parser import parse_srt_from_file, generate_srt_from_list
from config_manager import ConfigManager


SUPPORTED_MEDIA = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma',
                   '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.webm', '.ts'}


class PipelineRunner:
    """
    Orchestrates the Whisper -> Translate pipeline.

    This module is pure logic (no tkinter dependency). It processes a list of
    files sequentially:
      - For media files (audio/video): runs Whisper to produce an SRT, then
        translates it.
      - For SRT/TXT files: translates directly.

    The runner calls back to the owner for logging, progress updates, and
    completion notification.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        get_port: Callable[[], int],
        get_current_model: Callable[[], str],
        resolve_whisper_model_path: Callable[[str, str], str],
        get_whisper_models: Callable[[], List[Dict]],
        on_log: Callable[[str], None],
        on_progress: Callable[[str], None],
        on_done: Callable[[bool], None],
    ):
        self._config_manager = config_manager
        self._get_port = get_port
        self._get_current_model = get_current_model
        self._resolve_whisper_model_path = resolve_whisper_model_path
        self._get_whisper_models = get_whisper_models
        self._on_log = on_log
        self._on_progress = on_progress
        self._on_done = on_done

        self._running = False
        self._stop_requested = False

    @property
    def running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._stop_requested = True

    def run(
        self,
        files: List[str],
        target_lang: str,
        language: str,
        replace_original: bool,
        whisper_cli_path: Path,
        whisper_model_name: str,
        whisper_model_dir: str,
    ) -> None:
        """
        Process files sequentially in the calling thread.

        For each file:
          - .srt / .txt: translate directly.
          - media extensions: run Whisper, then translate the resulting SRT.

        Calls on_done(stopped: bool) when finished.
        """
        self._running = True
        self._stop_requested = False

        model_path = self._resolve_whisper_model_path(whisper_model_dir, whisper_model_name)

        try:
            for i, filepath in enumerate(files):
                if self._stop_requested:
                    break
                try:
                    ext = Path(filepath).suffix.lower()
                    if ext in ('.srt', '.txt'):
                        self._on_progress(
                            f"[{i+1}/{len(files)}] Translating: {Path(filepath).name}"
                        )
                        self._translate_file(filepath, target_lang, replace_original)
                    else:
                        self._on_progress(
                            f"[{i+1}/{len(files)}] Whisper: {Path(filepath).name}"
                        )
                        srt_path = self._run_whisper(
                            filepath, whisper_cli_path, model_path, language
                        )
                        if not srt_path or self._stop_requested:
                            continue
                        self._on_progress(
                            f"[{i+1}/{len(files)}] Translating: {Path(srt_path).name}"
                        )
                        self._translate_file(srt_path, target_lang, replace_original)
                except Exception as e:
                    self._on_log(f"Error: {Path(filepath).name} - {e}")
        finally:
            self._running = False
            self._on_done(stopped=self._stop_requested)

    # ------------------------------------------------------------------
    # Whisper step
    # ------------------------------------------------------------------

    def _run_whisper(
        self,
        filepath: str,
        cli_path: Path,
        model_path: str,
        language: str,
    ) -> Optional[str]:
        """
        Run WhisperController synchronously (blocking).

        Returns the output SRT path on success, or None if stopped/failed.
        """
        completed = threading.Event()
        result = {'srt': None, 'error': None}

        def on_ok(srt_path):
            result['srt'] = srt_path
            completed.set()

        def on_err(msg):
            result['error'] = msg
            completed.set()

        controller = WhisperController(
            cli_path=cli_path,
            on_log=lambda m: self._on_log(f"  {m}"),
            on_progress=lambda m: self._on_log(f"  {m}"),
            on_complete=on_ok,
            on_error=on_err,
        )

        threads = self._config_manager.get("whisper.threads", 8)
        controller.start(filepath, model_path, language, threads)

        while not completed.is_set():
            completed.wait(timeout=0.2)
            if self._stop_requested:
                controller.stop()
                return None

        if result['error']:
            raise RuntimeError(result['error'])
        return result['srt']

    # ------------------------------------------------------------------
    # Translation step
    # ------------------------------------------------------------------

    def _translate_file(
        self,
        srt_path: str,
        target_lang: str,
        replace_original: bool,
    ) -> None:
        """Parse an SRT file, translate it, and write the result."""
        subtitles = parse_srt_from_file(srt_path)
        if not subtitles:
            return

        port = self._get_port()
        config = {
            'api_url': f'http://localhost:{port}/v1',
            'model': self._get_current_model(),
            'max_tokens': self._config_manager.get("ui.max_tokens", 16384),
            'temperature': self._config_manager.get("ui.temperature", 0.2),
            'batch_size': self._config_manager.get("ui.batch_size", 15),
            'max_workers': self._config_manager.get("ui.max_workers", 3),
            'single_step': self._config_manager.get("ui.single_step", True),
        }

        translator = LocalLLMTranslator(config)
        translated = translator.translate_srt(
            subtitles, target_lang,
            progress_callback=lambda c, t, s: self._on_log(f"  {s}: {c}/{t}"),
            log_callback=lambda lv, m: self._on_log(f"  [{lv}] {m}"),
        )

        srt_content = generate_srt_from_list(translated)
        if replace_original:
            out_path = srt_path
        else:
            p = Path(srt_path)
            out_path = str(p.parent / f"{p.stem}_translated{p.suffix}")
        Path(out_path).write_text(srt_content, encoding='utf-8')
        self._on_progress(f"Saved: {Path(out_path).name}")
