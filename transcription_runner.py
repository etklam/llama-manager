"""One transcription run-loop for every consumer.

The Whisper tab and the pipeline runner used to hand-roll the same per-file
Transcription loop: build a request, run it through WhisperTranscriber with an
event handler, and map the Transcription Outcome onto UI state by hand — and
both dropped the committing/completed events, so the moment the Committed SRT
lands was invisible. This module is that loop, once.

It owns the run's stop flag and cancellation token, routes the engine's event
stream onto a TranscriptionPresenter (the whole UI, injected at the seam), and
handles every Transcription Outcome exactly once: Cancelled stops the run,
Failed is logged and skipped, Completed reports the Committed SRT. No Tkinter
here; the presenter marshals to the UI thread itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from whisper_transcription import (
    CancellationToken,
    Cancelled,
    Completed,
    Failed,
    TranscriptionEvent,
    TranscriptionRequest,
    WhisperTranscriber,
)


class TranscriptionPresenter:
    """Seam between the run loop and its shell.

    The loop calls these from the worker thread; the implementation marshals
    widget updates to the UI thread (after(0, ...)) and owns every widget it
    touches. Defaults are no-ops so a shell implements only what it displays.
    """

    def log(self, level: str, message: str) -> None:
        """A log line: level in {"INFO", "WARNING", "ERROR", "SUCCESS"}."""

    def status(self, message: str) -> None:
        """A one-line current status (stage names, per-file progress)."""

    def progress(self, percent: float) -> None:
        """Batch progress as a 0-100 percentage."""

    def srt_generated(self, filepath: str, srt_path: Path) -> None:
        """A Committed SRT landed for `filepath`."""

    def finished(self, stopped: bool) -> None:
        """The batch finished; `stopped` is True when it was interrupted."""


@dataclass(frozen=True)
class TranscriptionFileResult:
    """One file's run: the Committed SRT path, or why there is none."""

    srt_path: Optional[Path] = None
    cancelled: bool = False


@dataclass(frozen=True)
class TranscriptionBatchReport:
    """Aggregate of a run_files batch."""

    total: int
    completed: int
    failed: int
    stopped: bool


class TranscriptionRunner:
    """Run-loop for Transcription runs: one file at a time or a whole batch.

    Owns the stop flag and cancellation token of a run. `stop()` cancels the
    active Transcription and makes the loop exit after the current file; a
    Cancelled outcome is the engine's side of the same coin and stops the run
    too. Failures are logged and skipped. A successful run returns the
    Committed SRT path so the caller can sequence its next step (translate,
    update a file list).
    """

    def __init__(
        self,
        transcriber: WhisperTranscriber,
        presenter: TranscriptionPresenter,
    ):
        self._transcriber = transcriber
        self._presenter = presenter
        self._running = False
        self._stop_requested = False
        self._cancellation: Optional[CancellationToken] = None

    @property
    def running(self) -> bool:
        """True while run_files is iterating."""
        return self._running

    @property
    def stopped(self) -> bool:
        """True when stop() was requested (or the run was cancelled)."""
        return self._stop_requested

    def stop(self) -> None:
        """Request a stop: cancel the active Transcription, if any.

        The loop exits after the current file completes or is cancelled.
        """
        self._stop_requested = True
        if self._cancellation is not None:
            self._cancellation.cancel()

    def reset(self) -> None:
        """Discard run state after a run that used transcribe_file directly.

        run_files resets itself; callers of transcribe_file (the pipeline
        runner) must reset at the end of their own run, or a cancelled run
        would stop the next one.
        """
        self._running = False
        self._stop_requested = False
        self._cancellation = None

    def transcribe_file(
        self,
        filepath: str,
        build_request: Callable[[str], TranscriptionRequest],
    ) -> TranscriptionFileResult:
        """Run one Transcription and handle its outcome.

        Routes engine events to the presenter and logs the outcome once.
        Returns the Committed SRT path on success (or None). `cancelled` is
        True only when the user stopped the run — the caller should break its
        own loop.
        """
        if self._cancellation is None:
            self._cancellation = CancellationToken()
        if self._stop_requested:
            self._cancellation.cancel()

        name = Path(filepath).name
        try:
            request = build_request(filepath)
        except Exception as exc:
            self._presenter.log("ERROR", f"Failed: {name} - {exc}")
            return TranscriptionFileResult()

        outcome = self._transcriber.transcribe(
            request, cancellation=self._cancellation, emit=self._on_event)

        if isinstance(outcome, Cancelled):
            self._presenter.log("WARNING", f"Transcription cancelled: {name}")
            return TranscriptionFileResult(cancelled=True)
        if isinstance(outcome, Failed):
            self._presenter.log(
                "ERROR", f"Failed: {name} - {outcome.stage}: {outcome.message}")
            return TranscriptionFileResult()
        self._presenter.log("SUCCESS", f"Generated: {outcome.srt_path}")
        return TranscriptionFileResult(srt_path=outcome.srt_path)

    def run_files(
        self,
        files: List[str],
        build_request: Callable[[str], TranscriptionRequest],
    ) -> TranscriptionBatchReport:
        """Run a batch of files, stopping after the current one on stop()."""
        self._running = True
        total = len(files)
        completed = 0
        failed = 0
        try:
            for i, filepath in enumerate(files):
                if self._stop_requested:
                    self._presenter.log("WARNING", "Transcription stopped by user")
                    break
                self._presenter.status(
                    f"Processing: {Path(filepath).name} ({i + 1}/{total})")
                result = self.transcribe_file(filepath, build_request)
                if result.cancelled:
                    self._stop_requested = True
                    break
                if result.srt_path is not None:
                    completed += 1
                    self._presenter.srt_generated(filepath, result.srt_path)
                else:
                    failed += 1
                self._presenter.progress(((i + 1) / total) * 100)
        finally:
            stopped = self._stop_requested
            self.reset()
            self._presenter.finished(stopped=stopped)
            return TranscriptionBatchReport(
                total=total, completed=completed, failed=failed, stopped=stopped)

    def _on_event(self, event: TranscriptionEvent) -> None:
        """Map the engine's event stream onto the presenter, once.

        diagnostic → log line; started/progress/completed → status. The
        committing stage's completed event is the moment the Committed SRT
        lands, so it gets a status of its own instead of being dropped like
        the old per-consumer handlers did.
        """
        if event.kind == "diagnostic" and event.message:
            self._presenter.log("INFO", event.message)
        elif event.kind == "progress" and event.total:
            self._presenter.status(
                f"{event.stage}: {event.current}/{event.total}")
        elif event.kind == "started":
            self._presenter.status(event.stage.capitalize())
        elif event.kind == "completed":
            if event.stage == "committing":
                self._presenter.status("SRT committed")
            else:
                self._presenter.status(f"{event.stage.capitalize()} complete")
