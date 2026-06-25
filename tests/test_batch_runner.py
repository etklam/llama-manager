"""Tests for BatchRunner: the shared batch run-loop shell."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, call

import pytest

from ui_helpers import BatchRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def widgets():
    """Mock tkinter start/stop buttons + schedule fn."""
    return {
        'start_btn': MagicMock(),
        'stop_btn': MagicMock(),
        'log': MagicMock(),
        'schedule': MagicMock(),   # replaces tk after(0, fn)
    }


@pytest.fixture
def runner(widgets):
    return BatchRunner(
        start_btn=widgets['start_btn'],
        stop_btn=widgets['stop_btn'],
        log_fn=widgets['log'],
        schedule_fn=widgets['schedule'],
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_not_running_initially(self, runner):
        assert runner.is_running is False

    def test_stop_flag_false_initially(self, runner):
        assert runner._stop_requested is False


# ---------------------------------------------------------------------------
# run() iterates files, calls per_file_fn once per file
# ---------------------------------------------------------------------------

class TestRunIteratesFiles:

    def test_calls_per_file_fn_once_per_file(self, runner):
        per_file = MagicMock()
        runner.run(['a.srt', 'b.srt', 'c.srt'], per_file)
        assert per_file.call_count == 3
        per_file.assert_has_calls([call('a.srt'), call('b.srt'), call('c.srt')])

    def test_run_sets_running_flag(self, runner):
        seen = []
        def per_file(f):
            seen.append(runner.is_running)
        runner.run(['a'], per_file)
        assert seen == [True]

    def test_run_clears_running_after(self, runner):
        runner.run(['a'], lambda f: None)
        assert runner.is_running is False


# ---------------------------------------------------------------------------
# Button + schedule wiring on run lifecycle
# ---------------------------------------------------------------------------

class TestLifecycleWiring:

    def test_disables_start_enables_stop_at_start(self, runner, widgets):
        runner.run(['a'], lambda f: None)
        widgets['start_btn'].config.assert_any_call(state="disabled")
        widgets['stop_btn'].config.assert_any_call(state="normal")

    def test_re_enables_start_disables_stop_at_end(self, runner, widgets):
        runner.run(['a'], lambda f: None)
        # final state
        start_calls = widgets['start_btn'].config.call_args_list
        stop_calls = widgets['stop_btn'].config.call_args_list
        assert start_calls[-1] == call(state="normal")
        assert stop_calls[-1] == call(state="disabled")


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------

class TestStop:

    def test_stop_sets_flag(self, runner):
        runner.stop()
        assert runner._stop_requested is True

    def test_stop_exits_loop_after_current_file(self, runner):
        """When stop() is called mid-file, loop exits after that file."""
        processed = []

        def per_file(f):
            processed.append(f)
            if f == 'b':
                runner.stop()

        runner.run(['a', 'b', 'c', 'd'], per_file)
        # c and d must NOT be processed
        assert processed == ['a', 'b']


# ---------------------------------------------------------------------------
# Per-file exceptions don't crash the loop
# ---------------------------------------------------------------------------

class TestExceptionHandling:

    def test_exception_in_one_file_does_not_stop_loop(self, runner, widgets):
        per_file = MagicMock(side_effect=[RuntimeError("boom"), "ok"])
        runner.run(['bad', 'good'], per_file)
        assert per_file.call_count == 2

    def test_exception_is_logged(self, runner, widgets):
        per_file = MagicMock(side_effect=RuntimeError("boom"))
        runner.run(['bad'], per_file)
        # log_fn called at least once with ERROR level
        error_calls = [c for c in widgets['log'].call_args_list
                       if len(c.args) >= 2 and c.args[0] == "ERROR"]
        assert len(error_calls) >= 1


# ---------------------------------------------------------------------------
# on_progress callback fired per file
# ---------------------------------------------------------------------------

class TestProgressCallback:

    def test_on_progress_called_per_file_with_index(self, runner):
        progress = MagicMock()
        runner.run(['a', 'b'], lambda f: None, on_progress=progress)
        assert progress.call_count == 2
        # MagicMocks add a __bool__ check from `if on_progress:` — filter it.
        real_calls = [c for c in progress.call_args_list
                      if c.args and isinstance(c.args[0], int)]
        assert real_calls == [call(0, 'a'), call(1, 'b')]

    def test_on_progress_optional(self, runner):
        # No on_progress supplied: loop still completes
        per_file = MagicMock()
        runner.run(['a', 'b'], per_file)
        assert per_file.call_count == 2


# ---------------------------------------------------------------------------
# on_done callback semantics
# ---------------------------------------------------------------------------

class TestOnDone:

    def test_on_done_called_with_stopped_false_on_normal_completion(self, runner, widgets):
        done = MagicMock()
        runner.run(['a'], lambda f: None, on_done=done)
        # on_done is marshalled via schedule_fn; that wrapper is invoked once
        # and, when run, calls done(stopped=False).
        assert widgets['schedule'].call_count == 1
        scheduled_fn = widgets['schedule'].call_args.args[0]
        scheduled_fn()
        done.assert_called_once_with(stopped=False)

    def test_on_done_called_with_stopped_true_when_stopped(self, runner, widgets):
        done = MagicMock()
        def per_file(f):
            runner.stop()
        runner.run(['a', 'b'], per_file, on_done=done)
        scheduled_fn = widgets['schedule'].call_args.args[0]
        scheduled_fn()
        done.assert_called_once_with(stopped=True)


# ---------------------------------------------------------------------------
# schedule_fn used to marshal back to UI thread
# ---------------------------------------------------------------------------

class TestScheduleUsage:

    def test_schedule_invoked_for_done(self, runner, widgets):
        runner.run(['a'], lambda f: None, on_done=lambda **kw: None)
        assert widgets['schedule'].call_count == 1
