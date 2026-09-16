"""Replay cold server startup through the real probe and pipeline lifecycle."""
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
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


# ---------------------------------------------------------------------------
# Remote LLM mode: no llama-server involved
# ---------------------------------------------------------------------------

REMOTE_PROFILE = {
    'id': 'p1', 'name': 'OpenRouter',
    'base_url': 'https://openrouter.ai/api/v1',
    'model': 'provider/model', 'api_key_env': 'OPENROUTER_API_KEY',
    'max_workers': 2, 'proxy': '',
}


def make_remote_runner(tmp_path, monkeypatch, profile, *, with_key=True):
    """A runner whose config selects a remote profile as the LLM backend."""
    from config_manager import ConfigManager
    config = ConfigManager(str(tmp_path / 'remote_cfg.json'))
    config.load()
    config.set('llm.mode', 'remote')
    pid = profile.get('id') or 'p1'
    config.set('llm.profiles', [dict(profile, id=pid)])
    config.set('llm.active_profile_id', pid)
    if with_key:
        monkeypatch.setenv(profile['api_key_env'], 'sk-env')
    else:
        monkeypatch.delenv(profile['api_key_env'], raising=False)
    return PipelineRunner(
        config, lambda: 8080, lambda: 'local-model', lambda directory, name: name,
        lambda: [], Mock(), Mock(), Mock(),
    )


def test_remote_run_skips_llama_server_probe_and_runs(tmp_path, monkeypatch):
    runner = make_remote_runner(tmp_path, monkeypatch, REMOTE_PROFILE)
    runner._run_whisper = Mock(return_value=Path('clip.srt'))
    runner._translate_file = Mock(return_value=True)

    with patch('translation.server_probe.httpx.get') as probe:
        run(runner)

    probe.assert_not_called()  # remote mode must never call llama-server /props
    runner._run_whisper.assert_called_once()
    runner._translate_file.assert_called_once()
    assert runner.startup_error is None


def test_remote_run_uses_the_profile_endpoint_for_translation(tmp_path, monkeypatch):
    from utils.srt_parser import Cue
    cue = Cue(line=1, start_time=0, end_time=1000, text='Hello')
    runner = make_remote_runner(tmp_path, monkeypatch, REMOTE_PROFILE)
    runner._run_whisper = Mock(return_value=Path('clip.srt'))

    with patch('translation.server_probe.httpx.get'), \
         patch('pipeline_runner.parse_srt_from_file', return_value=[cue]), \
         patch('pipeline_runner.generate_srt_from_list', return_value='srt'), \
         patch('pathlib.Path.write_text', MagicMock()), \
         patch('pipeline_runner.LocalLLMTranslator') as translator_cls:
        translator_cls.return_value.translate_srt.return_value = [cue]
        run(runner)

    config = translator_cls.call_args[0][0]
    assert config['api_url'] == 'https://openrouter.ai/api/v1'
    assert config['model'] == 'provider/model'
    assert config['api_key'] == 'sk-env'
    assert config['max_workers'] == 2  # profile count, never slot-clamped


def test_remote_run_with_missing_key_fails_naming_the_profile(tmp_path, monkeypatch):
    runner = make_remote_runner(tmp_path, monkeypatch, REMOTE_PROFILE,
                                with_key=False)
    runner._run_whisper = Mock(return_value=Path('clip.srt'))

    with patch('translation.server_probe.httpx.get') as probe:
        run(runner)

    probe.assert_not_called()
    runner._run_whisper.assert_not_called()
    assert 'OpenRouter' in runner.startup_error
    assert 'OPENROUTER_API_KEY' in runner.startup_error
