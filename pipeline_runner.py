"""PipelineRunner - Orchestrates Whisper speech-to-text followed by subtitle translation."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

from whisper_transcription import (
    CancellationToken,
    Cancelled,
    Completed,
    Failed,
    TranscriptionEvent,
    TranscriptionRequest,
    WhisperTranscriber,
)
from translation.local_llm_translator import LocalLLMTranslator
from translation.server_probe import (
    ServerInfo,
    clamp_workers,
    probe_server,
    unreachable_message,
)
from utils.srt_parser import parse_srt_from_file, generate_srt_from_list, output_path_for
from config_manager import ConfigManager

from constants import SUPPORTED_MEDIA
from config_helpers import api_url_for_port, build_translation_config


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
        transcriber: Optional[WhisperTranscriber] = None,
    ):
        self._config_manager = config_manager
        self._get_port = get_port
        self._get_current_model = get_current_model
        self._resolve_whisper_model_path = resolve_whisper_model_path
        self._get_whisper_models = get_whisper_models
        self._on_log = on_log
        self._on_progress = on_progress
        self._on_done = on_done
        self._transcriber = transcriber or WhisperTranscriber()

        self._running = False
        self._stop_requested = False
        self._active_cancellation: Optional[CancellationToken] = None

        # Translator reused across files in a batch. Rebuilding it per file also
        # rebuilt the OpenAI client and its connection pool, so every file paid
        # a fresh TCP + HTTP handshake for work that targets the same local
        # server. Keyed on the config so a settings change still takes effect.
        self._translator: Optional[LocalLLMTranslator] = None
        self._translator_config: Optional[Dict] = None

        # What the preflight probe found, set once per run() and used to size the
        # worker pool. None means "not probed yet", which leaves the configured
        # worker count untouched.
        self._server_info: Optional[ServerInfo] = None
        # Logged once rather than per file: the clamp is identical for every file
        # in a run, and the pipeline log is already dense.
        self._last_clamp_note: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._stop_requested = True
        if self._active_cancellation is not None:
            self._active_cancellation.cancel()

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
        self._active_cancellation = CancellationToken()
        self._server_info = None
        self._last_clamp_note = None
        if self._stop_requested:
            self._active_cancellation.cancel()

        model_path = self._resolve_whisper_model_path(whisper_model_dir, whisper_model_name)

        try:
            # Probe once per run, before any transcription. The caller checks
            # that the server process is up, but a process that is still loading
            # a model answers nothing: without this, the failure only surfaces
            # after Whisper has finished, as a retry storm during translation.
            # The slot count also sizes the translator's worker pool.
            api_url = api_url_for_port(self._get_port())
            info = probe_server(api_url)
            if not info.reachable:
                message = unreachable_message(api_url, info)
                self._on_progress(f"Error: {message}")
                self._on_log(message)
                # Report this as a stop, not a clean finish: on_done only carries
                # a bool, and stopped=False makes the caller announce success for
                # a run that translated nothing.
                self._stop_requested = True
                return
            self._server_info = info

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
                        outcome = self._run_whisper(
                            filepath, whisper_cli_path, model_path, language
                        )
                        if isinstance(outcome, Cancelled):
                            self._stop_requested = True
                            break
                        if isinstance(outcome, Failed):
                            self._on_progress(
                                f"Error: {Path(filepath).name} - "
                                f"{outcome.stage}: {outcome.message}"
                            )
                            self._on_log(
                                f"Whisper failed ({outcome.stage}): {outcome.message}"
                            )
                            continue
                        if not isinstance(outcome, Completed) or self._stop_requested:
                            continue
                        srt_path = str(outcome.srt_path)
                        self._on_progress(
                            f"[{i+1}/{len(files)}] Translating: {Path(srt_path).name}"
                        )
                        self._translate_file(srt_path, target_lang, replace_original)
                except Exception as e:
                    self._on_progress(f"Error: {Path(filepath).name} - {e}")
                    self._on_log(f"Error: {Path(filepath).name} - {e}")
        finally:
            stopped = self._stop_requested
            self._active_cancellation = None
            self._running = False
            self._on_done(stopped=stopped)
            self._stop_requested = False

    # ------------------------------------------------------------------
    # Whisper step
    # ------------------------------------------------------------------

    def _run_whisper(
        self,
        filepath: str,
        cli_path: Path,
        model_path: str,
        language: str,
    ):
        threads = self._config_manager.get("whisper.threads", 8)
        chunk_enabled = bool(self._config_manager.get("whisper.chunk_long_audio", False))
        cancellation = self._active_cancellation or CancellationToken()
        return self._transcriber.transcribe(
            TranscriptionRequest(
                source=Path(filepath),
                cli_path=Path(cli_path),
                model_path=Path(model_path),
                language=language,
                threads=threads,
                chunk_long_audio=chunk_enabled,
            ),
            cancellation=cancellation,
            emit=self._on_whisper_event,
        )

    def _on_whisper_event(self, event: TranscriptionEvent) -> None:
        if event.kind == "diagnostic" and event.message:
            self._on_log(f"  {event.message}")
        elif event.kind == "progress" and event.total:
            self._on_progress(
                f"Whisper {event.stage}: {event.current}/{event.total}"
            )
        elif event.kind == "started":
            self._on_progress(f"Whisper: {event.stage}")

    # ------------------------------------------------------------------
    # Translation step
    # ------------------------------------------------------------------

    def _get_translator(self, config: Dict) -> LocalLLMTranslator:
        """Return a translator for `config`, reusing the previous one if it fits.

        A fresh LocalLLMTranslator per file means a fresh OpenAI client and HTTP
        connection pool per file. Holding one instance avoids that repeated
        setup and lets the SDK reuse warm connections. It does not guarantee
        llama-server slot affinity: slot selection and prompt-cache reuse remain
        server scheduling decisions.

        The config is still rebuilt per file because port and model can change
        between files, so we only reuse when every value the translator reads at
        construction time is unchanged.
        """
        if self._translator is None or config != self._translator_config:
            self._translator = LocalLLMTranslator(config)
            self._translator_config = dict(config)
        return self._translator

    def _translate_file(
        self,
        srt_path: str,
        target_lang: str,
        replace_original: bool,
    ) -> None:
        """Parse an SRT file, translate it, and write the result."""
        subtitles = parse_srt_from_file(srt_path)
        if not subtitles:
            self._on_progress(f"Empty SRT, skipping: {Path(srt_path).name}")
            return

        config = build_translation_config(
            self._config_manager, self._get_port(), self._get_current_model()
        )

        if not config.get('model'):
            raise RuntimeError("No model loaded - select a model on the Server tab")

        # Align workers with the slot count the probe reported. Workers past that
        # only queue on the server, so the extra concurrency buys nothing while
        # the log reports it as parallel work.
        if self._server_info is not None:
            workers, note = clamp_workers(config['max_workers'], self._server_info)
            config['max_workers'] = workers
            if note and note != self._last_clamp_note:
                self._on_log(note)
                self._last_clamp_note = note

        self._on_progress(f"Translating {len(subtitles)} lines: {Path(srt_path).name}")

        translator = self._get_translator(config)
        translated = translator.translate_srt(
            subtitles, target_lang,
            progress_callback=lambda c, t, s: self._on_progress(
                f"Translating {Path(srt_path).name}: {c}/{t} {s}"
            ),
            log_callback=lambda lv, m: self._on_log(f"  [{lv}] {m}"),
        )

        srt_content = generate_srt_from_list(translated)
        out_path = output_path_for(srt_path, target_lang, replace_original)
        Path(out_path).write_text(srt_content, encoding='utf-8')
        self._on_progress(f"Saved: {Path(out_path).name}")
