"""Replay cold server startup through the real probe and pipeline lifecycle."""
from pathlib import Path
from unittest.mock import Mock, patch
import threading

import httpx

from pipeline_runner import PipelineRunner


def make_runner():
    config = Mock()
    config.get.side_effect = lambda key, default=None: default
    runner = PipelineRunner(
        config, lambda: 8080, lambda: 'model', lambda directory, name: name,
        lambda: [], Mock(), Mock(), Mock(),
    )
    runner._run_whisper = Mock(return_value=Path('clip.srt'))
    runner._translate_file = Mock(return_value=True)
    return runner


def run(runner):
    runner.run(['clip.mp4'], 'zh-tw', 'ja', False, Path('whisper-cli.exe'),
               'kotoba-whisper-v2.0', 'models')


def test_first_start_waits_for_loading_server_then_runs_without_second_click():
    runner = make_runner()
    with patch('translation.server_probe.httpx.get', side_effect=[
        httpx.Response(503), httpx.Response(200, json={'total_slots': 1}),
    ]), patch('pipeline_runner.SERVER_READY_POLL_SECONDS', 0, create=True):
        run(runner)
    runner._run_whisper.assert_called_once()
    runner._translate_file.assert_called_once()
    runner._on_done.assert_called_once_with(stopped=False)


def test_connection_not_yet_listening_recovers_on_first_run():
    runner = make_runner()
    with patch('translation.server_probe.httpx.get', side_effect=[
        httpx.ConnectError('not listening yet'),
        httpx.Response(200, json={'total_slots': 1}),
    ]), patch('pipeline_runner.SERVER_READY_POLL_SECONDS', 0, create=True):
        run(runner)
    runner._run_whisper.assert_called_once()


def test_non_retryable_response_keeps_reason_and_does_not_retry():
    runner = make_runner()
    with patch('translation.server_probe.httpx.get', return_value=httpx.Response(401)) as get:
        run(runner)
    get.assert_called_once()
    runner._run_whisper.assert_not_called()
    assert 'HTTP 401' in runner.startup_error
    runner._on_done.assert_called_once_with(stopped=False)


def test_wait_timeout_is_an_error_not_a_user_stop():
    runner = make_runner()
    with patch('translation.server_probe.httpx.get', return_value=httpx.Response(503)), \
         patch('pipeline_runner.time.monotonic', side_effect=[0, 121]):
        run(runner)
    runner._run_whisper.assert_not_called()
    assert 'HTTP 503' in runner.startup_error
    assert not runner.running
    runner._on_done.assert_called_once_with(stopped=False)


def test_stop_interrupts_wait_and_does_not_stop_the_next_run():
    runner = make_runner()
    waiting = threading.Event()
    runner._on_progress.side_effect = lambda message: waiting.set()
    with patch('translation.server_probe.httpx.get', return_value=httpx.Response(503)):
        worker = threading.Thread(target=run, args=(runner,))
        worker.start()
        try:
            assert waiting.wait(2)
        finally:
            runner.stop()
            worker.join(2)
        assert not worker.is_alive()
    runner._on_done.assert_called_once_with(stopped=True)
    runner._run_whisper.assert_not_called()
    with patch('translation.server_probe.httpx.get',
               return_value=httpx.Response(200, json={'total_slots': 1})):
        run(runner)
    runner._run_whisper.assert_called_once()
    assert runner.startup_error is None


def test_first_run_setup_exception_resets_controls_with_a_reason():
    runner = make_runner()
    runner._resolve_whisper_model_path = Mock(side_effect=ValueError('invalid model'))
    run(runner)
    assert not runner.running
    assert 'invalid model' in runner.startup_error
    runner._on_done.assert_called_once_with(stopped=False)
