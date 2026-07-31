"""Tests for TranscriptionRunner: the shared transcription run-loop.

Replaces the tests of the superseded BatchRunner shell. The interface is the
test surface: outcome handling (Cancelled/Failed/Completed), event routing
(including the committing/completed stages), and the batch loop with its stop
semantics all live here, exercised through a fake transcriber and a mock
presenter — no Tkinter anywhere.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from transcription_runner import TranscriptionBatchReport, TranscriptionRunner
from whisper_transcription import (
    Cancelled,
    Completed,
    Failed,
    TranscriptionEvent,
    TranscriptionRequest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def transcriber():
    return MagicMock()


@pytest.fixture
def presenter():
    return MagicMock()


@pytest.fixture
def runner(transcriber, presenter):
    return TranscriptionRunner(transcriber, presenter)


def _request(filepath):
    return TranscriptionRequest(
        source=Path(filepath),
        cli_path=Path("whisper-cli.exe"),
        model_path=Path("model.bin"),
    )


def _completed(filepath):
    return Completed(Path(filepath).with_suffix(".srt"))


def _log_calls(presenter, level):
    return [c.args[1] for c in presenter.log.call_args_list
            if len(c.args) >= 2 and c.args[0] == level]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_not_running_initially(self, runner):
        assert runner.running is False

    def test_not_stopped_initially(self, runner):
        assert runner.stopped is False

    def test_reset_clears_run_state(self, runner):
        runner.stop()
        runner.reset()
        assert runner.stopped is False
        assert runner.running is False


# ---------------------------------------------------------------------------
# transcribe_file: one file, request building, outcome handling
# ---------------------------------------------------------------------------

class TestTranscribeFile:

    def test_builds_request_and_returns_committed_srt(self, runner, transcriber, presenter):
        transcriber.transcribe.return_value = _completed("/x/video.mp4")

        result = runner.transcribe_file("/x/video.mp4", _request)

        assert result.srt_path == Path("/x/video.srt")
        assert result.cancelled is False
        request = transcriber.transcribe.call_args.args[0]
        assert request.source == Path("/x/video.mp4")
        assert _log_calls(presenter, "SUCCESS") == [
            f"Generated: {Path('/x/video.srt')}"]

    def test_cancelled_outcome_reports_cancelled(self, runner, transcriber, presenter):
        transcriber.transcribe.return_value = Cancelled()

        result = runner.transcribe_file("/x/video.mp4", _request)

        assert result.srt_path is None
        assert result.cancelled is True
        assert _log_calls(presenter, "WARNING") == ["Transcription cancelled: video.mp4"]

    def test_failed_outcome_logs_stage_and_message(self, runner, transcriber, presenter):
        transcriber.transcribe.return_value = Failed("transcribing", "boom")

        result = runner.transcribe_file("/x/video.mp4", _request)

        assert result.srt_path is None
        assert result.cancelled is False
        assert _log_calls(presenter, "ERROR") == [
            "Failed: video.mp4 - transcribing: boom"]

    def test_exception_in_build_request_is_logged_and_skipped(self, runner, presenter):
        def bad_request(filepath):
            raise RuntimeError("no model")

        result = runner.transcribe_file("/x/video.mp4", bad_request)

        assert result.srt_path is None
        assert result.cancelled is False
        assert _log_calls(presenter, "ERROR") == ["Failed: video.mp4 - no model"]

    def test_stop_before_transcribe_cancels_the_fresh_token(self, runner, transcriber):
        runner.stop()
        seen = {}

        def spy(request, cancellation, emit):
            seen['cancelled'] = cancellation.cancelled
            return Cancelled()

        transcriber.transcribe.side_effect = spy

        runner.transcribe_file("/x/video.mp4", _request)

        assert seen['cancelled'] is True


# ---------------------------------------------------------------------------
# Event routing: stage -> status, kind -> log/status, commit moment surfaces
# ---------------------------------------------------------------------------

class TestEventRouting:

    def _run_with_events(self, runner, transcriber, events):
        def spy(request, cancellation, emit):
            for event in events:
                emit(event)
            return _completed("/x/video.mp4")

        transcriber.transcribe.side_effect = spy
        return runner.transcribe_file("/x/video.mp4", _request)

    def test_diagnostic_events_become_log_lines(self, runner, transcriber, presenter):
        self._run_with_events(runner, transcriber, [
            TranscriptionEvent("transcribing", "diagnostic", message="hello"),
        ])
        assert _log_calls(presenter, "INFO") == ["hello"]

    def test_started_events_become_stage_status(self, runner, transcriber, presenter):
        self._run_with_events(runner, transcriber, [
            TranscriptionEvent("preparing", "started"),
        ])
        presenter.status.assert_any_call("Preparing")

    def test_progress_events_become_fraction_status(self, runner, transcriber, presenter):
        self._run_with_events(runner, transcriber, [
            TranscriptionEvent("chunking", "progress", current=1, total=3),
        ])
        presenter.status.assert_any_call("chunking: 1/3")

    def test_completed_events_no_longer_dropped(self, runner, transcriber, presenter):
        self._run_with_events(runner, transcriber, [
            TranscriptionEvent("transcribing", "completed"),
        ])
        presenter.status.assert_any_call("Transcribing complete")

    def test_committing_completed_surfaces_the_committed_srt(self, runner, transcriber, presenter):
        """The moment the Committed SRT lands must be visible, not dropped."""
        self._run_with_events(runner, transcriber, [
            TranscriptionEvent("committing", "started"),
            TranscriptionEvent("committing", "completed"),
        ])
        presenter.status.assert_any_call("Committing")
        presenter.status.assert_any_call("SRT committed")

    def test_full_engine_stream_maps_cleanly(self, runner, transcriber, presenter):
        self._run_with_events(runner, transcriber, [
            TranscriptionEvent("preparing", "started"),
            TranscriptionEvent("preparing", "completed"),
            TranscriptionEvent("transcribing", "started"),
            TranscriptionEvent("transcribing", "completed"),
            TranscriptionEvent("committing", "started"),
            TranscriptionEvent("committing", "completed"),
        ])
        statuses = [c.args[0] for c in presenter.status.call_args_list]
        assert statuses == [
            "Preparing", "Preparing complete",
            "Transcribing", "Transcribing complete",
            "Committing", "SRT committed",
        ]


# ---------------------------------------------------------------------------
# run_files: the batch loop
# ---------------------------------------------------------------------------

class TestRunFiles:

    def test_transcribes_once_per_file(self, runner, transcriber, presenter):
        transcriber.transcribe.side_effect = [
            _completed("/x/a.mp4"), _completed("/x/b.mp4"), _completed("/x/c.mp4"),
        ]

        report = runner.run_files(["/x/a.mp4", "/x/b.mp4", "/x/c.mp4"], _request)

        assert transcriber.transcribe.call_count == 3
        assert report == TranscriptionBatchReport(
            total=3, completed=3, failed=0, stopped=False)

    def test_reports_failures_without_stopping(self, runner, transcriber):
        transcriber.transcribe.side_effect = [
            _completed("/x/a.mp4"),
            Failed("transcribing", "boom"),
            _completed("/x/c.mp4"),
        ]

        report = runner.run_files(["/x/a.mp4", "/x/b.mp4", "/x/c.mp4"], _request)

        assert transcriber.transcribe.call_count == 3
        assert (report.completed, report.failed, report.stopped) == (2, 1, False)

    def test_per_file_status_and_batch_progress(self, runner, transcriber, presenter):
        transcriber.transcribe.side_effect = [
            _completed("/x/a.mp4"), _completed("/x/b.mp4"),
        ]

        runner.run_files(["/x/a.mp4", "/x/b.mp4"], _request)

        presenter.status.assert_any_call("Processing: a.mp4 (1/2)")
        presenter.status.assert_any_call("Processing: b.mp4 (2/2)")
        presenter.progress.assert_has_calls([call(50.0), call(100.0)])

    def test_srt_generated_reported_per_completed_file(self, runner, transcriber, presenter):
        transcriber.transcribe.side_effect = [
            _completed("/x/a.mp4"), Failed("transcribing", "boom"),
        ]

        runner.run_files(["/x/a.mp4", "/x/b.mp4"], _request)

        presenter.srt_generated.assert_called_once_with("/x/a.mp4", Path("/x/a.srt"))

    def test_running_flag_set_during_iteration(self, runner, transcriber):
        seen = []

        def spy(request, cancellation, emit):
            seen.append(runner.running)
            return _completed(request.source)

        transcriber.transcribe.side_effect = spy
        runner.run_files(["/x/a.mp4"], _request)

        assert seen == [True]
        assert runner.running is False

    def test_finished_reports_clean_run(self, runner, transcriber, presenter):
        transcriber.transcribe.side_effect = [_completed("/x/a.mp4")]

        runner.run_files(["/x/a.mp4"], _request)

        presenter.finished.assert_called_once_with(stopped=False)

    def test_stop_requested_before_run_stops_without_transcribing(self, runner, transcriber, presenter):
        runner.stop()

        report = runner.run_files(["/x/a.mp4", "/x/b.mp4"], _request)

        assert transcriber.transcribe.call_count == 0
        assert report.stopped is True
        presenter.finished.assert_called_once_with(stopped=True)
        assert _log_calls(presenter, "WARNING") == ["Transcription stopped by user"]


# ---------------------------------------------------------------------------
# Stop semantics
# ---------------------------------------------------------------------------

class TestStop:

    def test_stop_sets_flag(self, runner):
        runner.stop()
        assert runner.stopped is True

    def test_stop_cancels_active_transcription(self, runner, transcriber):
        seen = {}

        def spy(request, cancellation, emit):
            runner.stop()
            seen['cancelled'] = cancellation.cancelled
            return Cancelled()

        transcriber.transcribe.side_effect = spy

        report = runner.run_files(["/x/a.mp4"], _request)

        assert seen['cancelled'] is True
        assert report.stopped is True

    def test_stop_exits_loop_after_current_file(self, runner, transcriber, presenter):
        def spy(request, cancellation, emit):
            runner.stop()
            return _completed(request.source)

        transcriber.transcribe.side_effect = spy

        report = runner.run_files(
            ["/x/a.mp4", "/x/b.mp4", "/x/c.mp4"], _request)

        # Only the first file is transcribed; the loop then sees the stop flag.
        assert transcriber.transcribe.call_count == 1
        assert report.stopped is True
        presenter.finished.assert_called_once_with(stopped=True)

    def test_cancelled_outcome_stops_the_run(self, runner, transcriber, presenter):
        transcriber.transcribe.side_effect = [
            Cancelled(), _completed("/x/b.mp4"),
        ]

        report = runner.run_files(["/x/a.mp4", "/x/b.mp4"], _request)

        assert transcriber.transcribe.call_count == 1
        assert report.stopped is True
        assert _log_calls(presenter, "WARNING") == ["Transcription cancelled: a.mp4"]

    def test_stopped_run_does_not_leak_into_next_run(self, runner, transcriber):
        transcriber.transcribe.side_effect = [
            _completed("/x/a.mp4"), _completed("/x/b.mp4"),
        ]

        runner.stop()
        runner.run_files(["/x/a.mp4"], _request)  # stops immediately
        report = runner.run_files(["/x/a.mp4", "/x/b.mp4"], _request)

        # The second run processes both files normally: the stop did not leak.
        assert transcriber.transcribe.call_count == 2
        assert (report.completed, report.failed, report.stopped) == (2, 0, False)
