"""PipelineRunner - Orchestrates Whisper speech-to-text followed by subtitle translation."""
from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Callable, Dict, List, Optional

from whisper_transcription import (
    TranscriptionRequest,
    WhisperTranscriber,
)
from transcription_runner import TranscriptionPresenter, TranscriptionRunner
from translation.local_llm_translator import LocalLLMTranslator
from translation.preflight import PreflightPlan, plan_for_target, run_preflight
from utils.srt_parser import parse_srt_from_file, generate_srt_from_list, output_path_for
from config_manager import ConfigManager

from constants import SUPPORTED_MEDIA
from config_helpers import build_translation_config
from llm_target import LLMTarget, resolve_llm_target
from whisper_policy import chunking_enabled

SERVER_READY_TIMEOUT_SECONDS = 120.0
SERVER_READY_POLL_SECONDS = 1.0


class _PipelinePresenter(TranscriptionPresenter):
    """Routes the run loop's presentation to the pipeline's text channels."""

    def __init__(self, on_log: Callable[[str], None],
                 on_progress: Callable[[str], None]):
        self._on_log = on_log
        self._on_progress = on_progress

    def log(self, level, message):
        # Errors also take over the status line so a failed file is visible
        # at a glance, matching the pipeline's own Error: lines.
        if level == "ERROR":
            self._on_progress(f"Error: {message}")
        self._on_log(message)

    def status(self, message):
        self._on_progress(message)


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
        on_file_completed: Optional[Callable[[str], None]] = None,
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
        self._on_file_completed = on_file_completed or (lambda filepath: None)
        self.failed_files = 0
        self.startup_error = None
        self._stop_event = threading.Event()
        self._transcriber = transcriber or WhisperTranscriber()

        # The whisper step goes through the shared transcription run loop,
        # which owns the stop flag and cancellation token for its part of the
        # run; this class keeps its own flag for the pipeline loop itself.
        self._whisper_runner = TranscriptionRunner(
            self._transcriber,
            _PipelinePresenter(self._on_log, self._on_progress),
        )

        self._running = False
        self._stop_requested = False

        # Translator reused across files in a batch. Rebuilding it per file also
        # rebuilt the OpenAI client and its connection pool, so every file paid
        # a fresh TCP + HTTP handshake for work that targets the same local
        # server. Keyed on the config so a settings change still takes effect.
        self._translator: Optional[LocalLLMTranslator] = None
        self._translator_config: Optional[Dict] = None

        # The preflight plan for this run, set once in run() and used to size
        # the worker pool per file. None means "not preflighted yet", which
        # leaves the configured worker count untouched.
        self._preflight_plan: Optional[PreflightPlan] = None
        self._context_mode_snapshot = 'none'
        self._translation_config_snapshot: Optional[Dict] = None
        self._target_snapshot: Optional[LLMTarget] = None

    @property
    def running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._stop_requested = True
        self._stop_event.set()
        self._whisper_runner.stop()

    def _wait_for_server(self, api_url, requested_workers):
        """Wait for cold-start transport/503 failures without blocking the UI."""
        deadline = time.monotonic() + SERVER_READY_TIMEOUT_SECONDS
        while not self._stop_requested:
            plan = run_preflight(api_url, requested_workers)
            if plan.reachable or not plan.info.retryable:
                return plan
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return plan
            self._on_progress(
                f"Waiting for llama-server to be ready ({remaining:.0f}s remaining)...")
            if self._stop_event.wait(min(SERVER_READY_POLL_SECONDS, remaining)):
                break
        return None

    def _wait_for_ready(self, target, requested_workers):
        """Preflight the backend the target names, waiting only when waiting helps.

        A local llama-server may still be loading a model, so retrying the
        probe pays off; a remote API has no cold start to wait through, so its
        plan (a configuration check) is returned immediately.
        """
        if target.mode == 'remote':
            return plan_for_target(target, requested_workers)
        return self._wait_for_server(target.api_url, requested_workers)

    def run(
        self,
        files: List[str],
        target_lang: str,
        language: str,
        replace_original: bool,
        whisper_cli_path: Path,
        whisper_model_name: str,
        whisper_model_dir: str,
        context_mode: Optional[str] = None,
    ) -> None:
        """
        Process files sequentially in the calling thread.

        For each file:
          - .srt / .txt: translate directly.
          - media extensions: run Whisper, then translate the resulting SRT.

        Calls on_done(stopped: bool) when finished.
        """
        self._running = True
        self.failed_files = 0
        self.startup_error = None
        self._preflight_plan = None
        self._context_mode_snapshot = (
            context_mode if context_mode in ('none', 'story')
            else self._config_manager.get('ui.context_mode', 'none')
        )
        # A stop requested before run() is honored by the run loop: its first
        # transcribe_file creates a fresh token and cancels it immediately.

        try:
            model_path = self._resolve_whisper_model_path(whisper_model_dir, whisper_model_name)
            # Preflight once per run, before any transcription. The caller
            # checks that the server process is up (local mode), but a process
            # that is still loading a model answers nothing: without this, the
            # failure only surfaces after Whisper has finished, as a retry
            # storm during translation. The resolved target decides which
            # preflight runs — the /props probe and its slot count for a local
            # llama-server, a configuration check for a remote profile.
            target = resolve_llm_target(
                self._config_manager, self._get_port, self._get_current_model
            )
            requested_workers = target.max_workers
            plan = self._wait_for_ready(target, requested_workers)
            if plan is None or self._stop_requested:
                return
            if not plan.reachable:
                message = plan.note or "no response"
                self.startup_error = message
                # PipelineCard publishes progress messages to its log, so sending
                # this failure through both callbacks would duplicate the line.
                self._on_progress(f"Error: {message}")
                return
            self._preflight_plan = plan
            self._target_snapshot = target
            self._translation_config_snapshot = build_translation_config(
                self._config_manager, target
            )
            self._translation_config_snapshot['max_workers'] = plan.workers
            self._translation_config_snapshot['context_size'] = plan.info.context_size
            self._translation_config_snapshot['context_mode'] = self._context_mode_snapshot
            # The plan formats its note once; log it here once, not per file.
            if plan.note:
                self._on_log(plan.note)

            for i, filepath in enumerate(files):
                if self._stop_requested:
                    break
                try:
                    ext = Path(filepath).suffix.lower()
                    if ext in ('.srt', '.txt'):
                        self._on_progress(
                            f"[{i+1}/{len(files)}] Translating: {Path(filepath).name}"
                        )
                        if self._translate_file(
                            filepath, target_lang, replace_original
                        ):
                            self._on_file_completed(filepath)
                    else:
                        self._on_progress(
                            f"[{i+1}/{len(files)}] Whisper: {Path(filepath).name}"
                        )
                        srt_path = self._run_whisper(
                            filepath, whisper_cli_path, model_path, language
                        )
                        if srt_path is not None and not self._stop_requested:
                            self._on_progress(
                                f"[{i+1}/{len(files)}] Translating: {Path(srt_path).name}"
                            )
                            if self._translate_file(
                                str(srt_path), target_lang, replace_original
                            ):
                                self._on_file_completed(filepath)
                        elif self._stop_requested:
                            break
                        else:
                            self.failed_files += 1
                except Exception as e:
                    if self._stop_requested:
                        break
                    self.failed_files += 1
                    self._on_progress(f"Error: {Path(filepath).name} - {e}")
                    self._on_log(f"Error: {Path(filepath).name} - {e}")
        except Exception as error:
            self.startup_error = f"{type(error).__name__}: {error}"
            self._on_progress(f"Error: {self.startup_error}")
        finally:
            stopped = self._stop_requested
            self._whisper_runner.reset()
            self._running = False
            self._on_done(stopped=stopped)
            self._stop_requested = False
            self._stop_event.clear()

    # ------------------------------------------------------------------
    # Whisper step
    # ------------------------------------------------------------------

    def _run_whisper(
        self,
        filepath: str,
        cli_path: Path,
        model_path: str,
        language: str,
    ) -> Optional[Path]:
        """Run the whisper step for one media file through the shared loop.

        Event routing, outcome handling, and cancellation all live in the
        TranscriptionRunner; this returns the Committed SRT path, or None
        when the file failed or the run was cancelled.
        """
        return self._whisper_runner.transcribe_file(
            filepath,
            lambda f: self._build_whisper_request(
                f, cli_path, model_path, language),
        ).srt_path

    def _build_whisper_request(
        self,
        filepath: str,
        cli_path: Path,
        model_path: str,
        language: str,
    ) -> TranscriptionRequest:
        threads = self._config_manager.get("whisper.threads", 8)
        chunk_enabled = chunking_enabled(self._config_manager)
        return TranscriptionRequest(
            source=Path(filepath),
            cli_path=Path(cli_path),
            model_path=Path(model_path),
            language=language,
            threads=threads,
            chunk_long_audio=chunk_enabled,
        )

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

        A run snapshots translation settings after preflight, so all files use
        the same mode, model, capacity, and worker plan. A later run may carry a
        different snapshot; reuse only occurs when every construction value is
        unchanged.
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
    ) -> bool:
        """Parse, translate, and save a subtitle file when it completes."""
        snapshot = getattr(self, '_translation_config_snapshot', None)
        target = getattr(self, '_target_snapshot', None)
        config = dict(snapshot) if snapshot is not None else build_translation_config(
            self._config_manager,
            target or resolve_llm_target(
                self._config_manager, self._get_port, self._get_current_model
            ),
        )

        if not config.get('model'):
            if target is not None and target.mode == 'remote':
                raise RuntimeError(
                    "No model configured - set one in the active remote profile")
            raise RuntimeError("No model loaded - select a model on the Server tab")

        # Align workers with the plan's decision. Workers past the server's slot
        # count only queue, so the extra concurrency buys nothing while the log
        # reports it as parallel work.
        if self._preflight_plan is not None:
            config['max_workers'] = self._preflight_plan.workers
            config['context_size'] = self._preflight_plan.info.context_size
        config['context_mode'] = getattr(
            self, '_context_mode_snapshot', config.get('context_mode', 'none')
        )

        if Path(srt_path).suffix.lower() == '.txt' and config['context_mode'] == 'story':
            self._on_progress("全文理解翻譯只適用於 SRT；TXT 使用標準翻譯")
            config['context_mode'] = 'none'

        subtitles = parse_srt_from_file(srt_path)
        if not subtitles:
            self._on_progress(f"Empty SRT, skipping: {Path(srt_path).name}")
            return False

        self._on_progress(f"Translating {len(subtitles)} lines: {Path(srt_path).name}")

        translator = self._get_translator(config)
        translated = translator.translate_srt(
            subtitles, target_lang,
            progress_callback=lambda c, t, s: self._on_progress(
                (f"Translating {Path(srt_path).name}: {c}/{t} {s}"
                 if c >= 0 else f"{Path(srt_path).name}: {s}")
            ),
            log_callback=lambda lv, m: self._on_log(f"  [{lv}] {m}"),
            cancel_callback=lambda: getattr(self, '_stop_requested', False),
        )

        if getattr(self, '_stop_requested', False):
            return False

        srt_content = generate_srt_from_list(translated)
        out_path = output_path_for(srt_path, target_lang, replace_original)
        Path(out_path).write_text(srt_content, encoding='utf-8')
        self._on_progress(f"Saved: {Path(out_path).name}")
        return True
