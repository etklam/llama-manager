"""Tests for PipelineRunner module."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

import pytest

from pipeline_runner import PipelineRunner, SUPPORTED_MEDIA
from whisper_transcription import Cancelled, Completed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_manager(tmp_path):
    """Create a ConfigManager backed by a temp file."""
    from config_manager import ConfigManager
    config_file = tmp_path / "test_config.json"
    cm = ConfigManager(str(config_file))
    cm.load()
    return cm


@pytest.fixture
def callbacks():
    """Create mock callbacks for log, progress, and completion."""
    return {
        'on_log': Mock(),
        'on_progress': Mock(),
        'on_done': Mock(),
    }


@pytest.fixture
def runner(config_manager, callbacks):
    """Create a PipelineRunner with mock callbacks."""
    return PipelineRunner(
        config_manager=config_manager,
        get_port=lambda: 8080,
        get_current_model=lambda: 'test-model',
        resolve_whisper_model_path=lambda model_dir, name: str(Path(model_dir) / name) if model_dir else name,
        get_whisper_models=lambda: [{'name': 'tiny', 'path': '/models/tiny.bin'}],
        on_log=callbacks['on_log'],
        on_progress=callbacks['on_progress'],
        on_done=callbacks['on_done'],
    )


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class TestConstructor:

    def test_not_running_initially(self, runner):
        assert runner.running is False

    def test_stop_requested_is_false_initially(self, runner):
        assert runner._stop_requested is False


# ---------------------------------------------------------------------------
# SRT/TXT files: translate directly
# ---------------------------------------------------------------------------

class TestTranslateSrtDirectly:

    @patch('pipeline_runner.generate_srt_from_list', return_value="1\n00:00:00,000 --> 00:00:01,000\nHello\n")
    @patch('pipeline_runner.LocalLLMTranslator')
    @patch('pipeline_runner.parse_srt_from_file')
    @patch('builtins.open', MagicMock())
    @patch('pathlib.Path.write_text', MagicMock())
    def test_srt_file_translates_directly(
        self, mock_parse, mock_translator_cls, mock_generate, runner, callbacks
    ):
        mock_parse.return_value = [
            {'line': 1, 'text': 'Hello', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator = MagicMock()
        mock_translator.translate_srt.return_value = [
            {'line': 1, 'text': 'Hola', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator_cls.return_value = mock_translator

        runner.run(
            files=['/test/video.srt'],
            target_lang='zh-cn',
            language='en',
            replace_original=False,
            whisper_cli_path=Path('whisper-cli.exe'),
            whisper_model_name='tiny',
            whisper_model_dir='/models',
        )

        # Should NOT call whisper, only translate
        mock_parse.assert_called_once_with('/test/video.srt')
        mock_translator.translate_srt.assert_called_once()
        mock_generate.assert_called_once()
        callbacks['on_done'].assert_called_once()

    @patch('pipeline_runner.generate_srt_from_list', return_value="1\n00:00:00,000 --> 00:00:01,000\nHello\n")
    @patch('pipeline_runner.LocalLLMTranslator')
    @patch('pipeline_runner.parse_srt_from_file')
    @patch('pathlib.Path.write_text', MagicMock())
    def test_txt_file_translates_directly(
        self, mock_parse, mock_translator_cls, mock_generate, runner, callbacks
    ):
        mock_parse.return_value = [
            {'line': 1, 'text': 'Hello', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator = MagicMock()
        mock_translator.translate_srt.return_value = [
            {'line': 1, 'text': 'Hola', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator_cls.return_value = mock_translator

        runner.run(
            files=['/test/subs.txt'],
            target_lang='zh-cn',
            language='en',
            replace_original=False,
            whisper_cli_path=Path('whisper-cli.exe'),
            whisper_model_name='tiny',
            whisper_model_dir='/models',
        )

        mock_parse.assert_called_once_with('/test/subs.txt')
        callbacks['on_done'].assert_called_once()


# ---------------------------------------------------------------------------
# Media files: whisper then translate
# ---------------------------------------------------------------------------

class TestMediaFileWhisperThenTranslate:

    @patch('pipeline_runner.generate_srt_from_list', return_value="srt output")
    @patch('pipeline_runner.LocalLLMTranslator')
    @patch('pipeline_runner.parse_srt_from_file')
    @patch('pathlib.Path.write_text', MagicMock())
    def test_media_file_runs_whisper_then_translates(
        self, mock_parse, mock_translator_cls, mock_generate, runner, callbacks
    ):
        # Mock _run_whisper to return an SRT path immediately
        runner._run_whisper = Mock(
            return_value=Completed(Path('/test/video.srt'))
        )

        # Set up translation
        mock_parse.return_value = [
            {'line': 1, 'text': 'Hello', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator = MagicMock()
        mock_translator.translate_srt.return_value = [
            {'line': 1, 'text': 'Translated', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator_cls.return_value = mock_translator

        runner.run(
            files=['/test/video.mp4'],
            target_lang='zh-cn',
            language='en',
            replace_original=False,
            whisper_cli_path=Path('whisper-cli.exe'),
            whisper_model_name='tiny',
            whisper_model_dir='/models',
        )

        runner._run_whisper.assert_called_once()
        mock_translator.translate_srt.assert_called_once()
        callbacks['on_done'].assert_called_once()


# ---------------------------------------------------------------------------
# Multiple files processed sequentially
# ---------------------------------------------------------------------------

class TestMultipleFilesSequential:

    @patch('pipeline_runner.generate_srt_from_list', return_value="srt output")
    @patch('pipeline_runner.LocalLLMTranslator')
    @patch('pipeline_runner.parse_srt_from_file')
    @patch('pathlib.Path.write_text', MagicMock())
    def test_processes_files_sequentially(
        self, mock_parse, mock_translator_cls, mock_generate, runner, callbacks
    ):
        mock_parse.return_value = [
            {'line': 1, 'text': 'Hello', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator = MagicMock()
        mock_translator.translate_srt.return_value = [
            {'line': 1, 'text': 'Translated', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator_cls.return_value = mock_translator

        runner.run(
            files=['/test/a.srt', '/test/b.srt'],
            target_lang='zh-cn',
            language='en',
            replace_original=False,
            whisper_cli_path=Path('whisper-cli.exe'),
            whisper_model_name='tiny',
            whisper_model_dir='/models',
        )

        assert mock_translator.translate_srt.call_count == 2
        callbacks['on_done'].assert_called_once()


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------

class TestStop:

    def test_stop_sets_flag(self, runner):
        runner.stop()
        assert runner._stop_requested is True

    def test_stop_before_run_is_not_discarded(self, runner, callbacks):
        runner.stop()

        runner.run(
            files=['/test/video.mp4'],
            target_lang='zh-cn',
            language='en',
            replace_original=False,
            whisper_cli_path=Path('whisper-cli.exe'),
            whisper_model_name='tiny',
            whisper_model_dir='',
        )

        callbacks['on_done'].assert_called_once_with(stopped=True)

    def test_stop_cancels_active_transcription(self, config_manager, callbacks):
        holder = {}

        class CancellingTranscriber:
            def transcribe(self, request, cancellation, emit):
                holder['runner'].stop()
                assert cancellation.cancelled is True
                return Cancelled()

        runner = PipelineRunner(
            config_manager=config_manager,
            get_port=lambda: 8080,
            get_current_model=lambda: 'test-model',
            resolve_whisper_model_path=lambda d, n: n,
            get_whisper_models=lambda: [],
            on_log=callbacks['on_log'],
            on_progress=callbacks['on_progress'],
            on_done=callbacks['on_done'],
            transcriber=CancellingTranscriber(),
        )
        holder['runner'] = runner

        runner.run(
            files=['/test/video.mp4'],
            target_lang='zh-cn',
            language='en',
            replace_original=False,
            whisper_cli_path=Path('whisper-cli.exe'),
            whisper_model_name='tiny',
            whisper_model_dir='',
        )

        callbacks['on_done'].assert_called_once_with(stopped=True)

    @patch('pipeline_runner.generate_srt_from_list', return_value="srt output")
    @patch('pipeline_runner.LocalLLMTranslator')
    @patch('pipeline_runner.parse_srt_from_file')
    @patch('pathlib.Path.write_text', MagicMock())
    def test_stop_stops_after_current_file(
        self, mock_parse, mock_translator_cls, mock_generate, runner, callbacks
    ):
        """When stop is requested, pipeline should stop after the current file."""
        mock_parse.return_value = [
            {'line': 1, 'text': 'Hello', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator = MagicMock()
        translated_result = [
            {'line': 1, 'text': 'Translated', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]

        call_count = 0

        def translate_with_stop(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            runner.stop()
            return translated_result

        mock_translator.translate_srt.side_effect = translate_with_stop
        mock_translator_cls.return_value = mock_translator

        runner.run(
            files=['/test/a.srt', '/test/b.srt', '/test/c.srt'],
            target_lang='zh-cn',
            language='en',
            replace_original=False,
            whisper_cli_path=Path('whisper-cli.exe'),
            whisper_model_name='tiny',
            whisper_model_dir='/models',
        )

        # Should only have processed 1 file (then stop was set)
        assert call_count == 1
        callbacks['on_done'].assert_called_once()


# ---------------------------------------------------------------------------
# Replace original
# ---------------------------------------------------------------------------

class TestReplaceOriginal:

    @patch('pipeline_runner.generate_srt_from_list', return_value="srt content")
    @patch('pipeline_runner.LocalLLMTranslator')
    @patch('pipeline_runner.parse_srt_from_file')
    def test_replace_original_writes_to_same_path(
        self, mock_parse, mock_translator_cls, mock_generate, runner, callbacks, tmp_path
    ):
        srt_file = str(tmp_path / "video.srt")
        Path(srt_file).write_text("original", encoding='utf-8')

        mock_parse.return_value = [
            {'line': 1, 'text': 'Hello', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator = MagicMock()
        mock_translator.translate_srt.return_value = [
            {'line': 1, 'text': 'Translated', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator_cls.return_value = mock_translator

        runner.run(
            files=[srt_file],
            target_lang='zh-cn',
            language='en',
            replace_original=True,
            whisper_cli_path=Path('whisper-cli.exe'),
            whisper_model_name='tiny',
            whisper_model_dir='/models',
        )

        # When replace_original is True, output path == input path
        written = Path(srt_file).read_text(encoding='utf-8')
        assert written == "srt content"

    @patch('pipeline_runner.generate_srt_from_list', return_value="srt content")
    @patch('pipeline_runner.LocalLLMTranslator')
    @patch('pipeline_runner.parse_srt_from_file')
    def test_no_replace_writes_to_translated_path(
        self, mock_parse, mock_translator_cls, mock_generate, runner, callbacks, tmp_path
    ):
        srt_file = str(tmp_path / "video.srt")
        Path(srt_file).write_text("original", encoding='utf-8')

        mock_parse.return_value = [
            {'line': 1, 'text': 'Hello', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator = MagicMock()
        mock_translator.translate_srt.return_value = [
            {'line': 1, 'text': 'Translated', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator_cls.return_value = mock_translator

        runner.run(
            files=[srt_file],
            target_lang='zh-cn',
            language='en',
            replace_original=False,
            whisper_cli_path=Path('whisper-cli.exe'),
            whisper_model_name='tiny',
            whisper_model_dir='/models',
        )

        expected_path = tmp_path / "video_Simplified Chinese.srt"  # ponytail: was _translated, unified via output_path_for
        assert expected_path.exists()
        assert expected_path.read_text(encoding='utf-8') == "srt content"
        # Original should be unchanged
        assert Path(srt_file).read_text(encoding='utf-8') == "original"


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TestCallbacks:

    @patch('pipeline_runner.generate_srt_from_list', return_value="srt output")
    @patch('pipeline_runner.LocalLLMTranslator')
    @patch('pipeline_runner.parse_srt_from_file')
    @patch('pathlib.Path.write_text', MagicMock())
    def test_on_progress_called_per_file(
        self, mock_parse, mock_translator_cls, mock_generate, runner, callbacks
    ):
        mock_parse.return_value = [
            {'line': 1, 'text': 'Hello', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator = MagicMock()
        mock_translator.translate_srt.return_value = [
            {'line': 1, 'text': 'Translated', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator_cls.return_value = mock_translator

        runner.run(
            files=['/test/a.srt'],
            target_lang='zh-cn',
            language='en',
            replace_original=False,
            whisper_cli_path=Path('whisper-cli.exe'),
            whisper_model_name='tiny',
            whisper_model_dir='/models',
        )

        # on_progress should be called at least once per file
        assert callbacks['on_progress'].call_count >= 1

    @patch('pipeline_runner.generate_srt_from_list', return_value="srt output")
    @patch('pipeline_runner.LocalLLMTranslator')
    @patch('pipeline_runner.parse_srt_from_file')
    @patch('pathlib.Path.write_text', MagicMock())
    def test_on_done_called_when_complete(
        self, mock_parse, mock_translator_cls, mock_generate, runner, callbacks
    ):
        mock_parse.return_value = [
            {'line': 1, 'text': 'Hello', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator = MagicMock()
        mock_translator.translate_srt.return_value = [
            {'line': 1, 'text': 'Translated', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator_cls.return_value = mock_translator

        runner.run(
            files=['/test/a.srt'],
            target_lang='zh-cn',
            language='en',
            replace_original=False,
            whisper_cli_path=Path('whisper-cli.exe'),
            whisper_model_name='tiny',
            whisper_model_dir='/models',
        )

        callbacks['on_done'].assert_called_once_with(stopped=False)

    @patch('pipeline_runner.generate_srt_from_list', return_value="srt output")
    @patch('pipeline_runner.LocalLLMTranslator')
    @patch('pipeline_runner.parse_srt_from_file')
    @patch('pathlib.Path.write_text', MagicMock())
    def test_on_done_called_with_stopped_when_stop_requested(
        self, mock_parse, mock_translator_cls, mock_generate, runner, callbacks
    ):
        mock_parse.return_value = [
            {'line': 1, 'text': 'Hello', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator = MagicMock()
        mock_translator.translate_srt.return_value = [
            {'line': 1, 'text': 'Translated', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator_cls.return_value = mock_translator

        # Simulate stop being requested during translation of the first file
        original_translate = mock_translator.translate_srt

        def translate_then_stop(*args, **kwargs):
            runner._stop_requested = True
            return original_translate(*args, **kwargs)

        mock_translator.translate_srt.side_effect = translate_then_stop

        runner.run(
            files=['/test/a.srt', '/test/b.srt'],
            target_lang='zh-cn',
            language='en',
            replace_original=False,
            whisper_cli_path=Path('whisper-cli.exe'),
            whisper_model_name='tiny',
            whisper_model_dir='/models',
        )

        callbacks['on_done'].assert_called_once_with(stopped=True)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    @patch('pipeline_runner.generate_srt_from_list', return_value="srt output")
    @patch('pipeline_runner.LocalLLMTranslator')
    @patch('pipeline_runner.parse_srt_from_file')
    @patch('pathlib.Path.write_text', MagicMock())
    def test_error_in_file_does_not_stop_pipeline(
        self, mock_parse, mock_translator_cls, mock_generate, runner, callbacks
    ):
        """Errors in individual files should be logged but not crash the pipeline."""
        # First file raises, second file succeeds
        mock_parse.side_effect = [
            RuntimeError("parse error"),
            [{'line': 1, 'text': 'Hello', 'time': '00:00:00,000 --> 00:00:01,000'}],
        ]
        mock_translator = MagicMock()
        mock_translator.translate_srt.return_value = [
            {'line': 1, 'text': 'Translated', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator_cls.return_value = mock_translator

        runner.run(
            files=['/test/bad.srt', '/test/good.srt'],
            target_lang='zh-cn',
            language='en',
            replace_original=False,
            whisper_cli_path=Path('whisper-cli.exe'),
            whisper_model_name='tiny',
            whisper_model_dir='/models',
        )

        # Should still process second file
        assert mock_translator.translate_srt.call_count == 1
        callbacks['on_done'].assert_called_once()


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------

class TestModelPathResolution:

    def test_supported_media_contains_common_formats(self):
        assert '.mp4' in SUPPORTED_MEDIA
        assert '.mkv' in SUPPORTED_MEDIA
        assert '.wav' in SUPPORTED_MEDIA
        assert '.mp3' in SUPPORTED_MEDIA
        assert '.flac' in SUPPORTED_MEDIA

    def test_srt_not_in_supported_media(self):
        assert '.srt' not in SUPPORTED_MEDIA
        assert '.txt' not in SUPPORTED_MEDIA


# ---------------------------------------------------------------------------
# Translation config from config_manager
# ---------------------------------------------------------------------------

class TestTranslationConfig:

    @patch('pipeline_runner.generate_srt_from_list', return_value="srt output")
    @patch('pipeline_runner.LocalLLMTranslator')
    @patch('pipeline_runner.parse_srt_from_file')
    @patch('pathlib.Path.write_text', MagicMock())
    def test_uses_config_manager_translation_settings(
        self, mock_parse, mock_translator_cls, mock_generate, config_manager, callbacks
    ):
        config_manager.set('ui.batch_size', 25)
        config_manager.set('ui.temperature', 0.5)

        mock_parse.return_value = [
            {'line': 1, 'text': 'Hello', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator = MagicMock()
        mock_translator.translate_srt.return_value = [
            {'line': 1, 'text': 'Translated', 'time': '00:00:00,000 --> 00:00:01,000'}
        ]
        mock_translator_cls.return_value = mock_translator

        runner = PipelineRunner(
            config_manager=config_manager,
            get_port=lambda: 8080,
            get_current_model=lambda: 'test-model',
            resolve_whisper_model_path=lambda d, n: n,
            get_whisper_models=lambda: [],
            on_log=callbacks['on_log'],
            on_progress=callbacks['on_progress'],
            on_done=callbacks['on_done'],
        )

        runner.run(
            files=['/test/a.srt'],
            target_lang='zh-cn',
            language='en',
            replace_original=False,
            whisper_cli_path=Path('whisper-cli.exe'),
            whisper_model_name='tiny',
            whisper_model_dir='',
        )

        # Check that LocalLLMTranslator was created with the merged config
        translator_call_config = mock_translator_cls.call_args[0][0]
        assert translator_call_config['batch_size'] == 25
        assert translator_call_config['temperature'] == 0.5
