"""Tests for WhisperController module."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import threading
import time
import tempfile
import os
import subprocess

from whisper_controller import WhisperController


@pytest.fixture
def temp_input_file():
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    yield temp_path
    try:
        os.unlink(temp_path)
    except:
        pass


@pytest.fixture
def temp_mp3_file():
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        temp_path = f.name
    yield temp_path
    try:
        os.unlink(temp_path)
    except:
        pass


@pytest.fixture
def temp_model_file():
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        temp_path = f.name
    yield temp_path
    try:
        os.unlink(temp_path)
    except:
        pass


def make_controller():
    cli_path = Path("whisper-cli.exe")
    on_log = Mock()
    on_progress = Mock()
    on_complete = Mock()
    on_error = Mock()
    controller = WhisperController(cli_path, on_log, on_progress, on_complete, on_error)
    return controller, on_log, on_progress, on_complete, on_error


class TestInitialState:
    def test_not_running(self):
        controller, _, _, _, _ = make_controller()
        assert controller.running is False

    def test_no_current_file(self):
        controller, _, _, _, _ = make_controller()
        assert controller.current_file is None


class TestStart:
    @patch('whisper_controller.subprocess.Popen')
    def test_start_launches_whisper_cli_with_correct_arguments(self, mock_popen, temp_input_file, temp_model_file):
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller, _, _, _, _ = make_controller()

        controller.start(
            input_file=temp_input_file,
            model_path=temp_model_file,
            language='en',
            threads=4
        )

        expected_cmd = [
            str(Path("whisper-cli.exe")),
            "-m", temp_model_file,
            "-f", temp_input_file,
            "--output-srt",
            "--output-file", str(Path(temp_input_file).with_suffix('')),
            "-t", "4",
            "--language", "en"
        ]
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == expected_cmd
        assert call_args[1]['stdout'] == subprocess.PIPE
        assert call_args[1]['stderr'] == subprocess.STDOUT
        assert call_args[1]['text'] is True
        assert call_args[1]['bufsize'] == 1

    @patch('whisper_controller.subprocess.Popen')
    def test_running_is_true_after_start(self, mock_popen, temp_input_file, temp_model_file):
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller, _, _, _, _ = make_controller()

        controller.start(
            input_file=temp_input_file,
            model_path=temp_model_file,
            language='en',
            threads=4
        )

        assert controller.running is True

    def test_start_raises_if_input_file_doesnt_exist(self, temp_model_file):
        controller, _, _, _, _ = make_controller()

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            controller.start(
                input_file="D:\\nonexistent\\audio.wav",
                model_path=temp_model_file,
                language='en',
                threads=4
            )

    @patch('whisper_controller.subprocess.Popen')
    def test_start_raises_if_already_running(self, mock_popen, temp_input_file, temp_model_file):
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller, _, _, _, _ = make_controller()

        controller.start(
            input_file=temp_input_file,
            model_path=temp_model_file,
            language='en',
            threads=4
        )

        with pytest.raises(RuntimeError, match="Whisper is already running"):
            controller.start(
                input_file=temp_input_file,
                model_path=temp_model_file,
                language='en',
                threads=4
            )

    @patch('whisper_controller.subprocess.Popen')
    def test_start_sets_current_file(self, mock_popen, temp_input_file, temp_model_file):
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller, _, _, _, _ = make_controller()

        controller.start(
            input_file=temp_input_file,
            model_path=temp_model_file,
            language='en',
            threads=4
        )

        assert controller.current_file == temp_input_file

    @patch('whisper_controller.subprocess.Popen')
    def test_start_launches_ffmpeg_for_non_wav_files(self, mock_popen, temp_mp3_file, temp_model_file):
        ffmpeg_process = Mock()
        ffmpeg_process.returncode = 0
        ffmpeg_process.communicate.return_value = ('', '')

        whisper_process = Mock()
        whisper_process.stdout.readline.return_value = ''
        whisper_process.poll.return_value = None

        def popen_side_effect(cmd, **kwargs):
            if 'ffmpeg' in cmd[0]:
                return ffmpeg_process
            return whisper_process

        mock_popen.side_effect = popen_side_effect

        controller, _, _, _, _ = make_controller()

        controller.start(
            input_file=temp_mp3_file,
            model_path=temp_model_file,
            language='en',
            threads=4
        )

        calls = mock_popen.call_args_list
        assert len(calls) == 2

        ffmpeg_cmd = calls[0][0][0]
        assert 'ffmpeg' in ffmpeg_cmd[0]
        assert '-i' in ffmpeg_cmd
        assert temp_mp3_file in ffmpeg_cmd
        assert '-ac' in ffmpeg_cmd
        assert '1' in ffmpeg_cmd
        assert '-ar' in ffmpeg_cmd
        assert '16000' in ffmpeg_cmd

    @patch('whisper_controller.subprocess.Popen')
    def test_start_uses_output_dir_when_provided(self, mock_popen, temp_input_file, temp_model_file):
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller, _, _, _, _ = make_controller()

        output_dir = "C:\\output"
        controller.start(
            input_file=temp_input_file,
            model_path=temp_model_file,
            language='en',
            threads=4,
            output_dir=output_dir
        )

        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        output_file_idx = cmd.index('--output-file')
        output_path = cmd[output_file_idx + 1]
        assert output_path.startswith(output_dir)


class TestStop:
    @patch('whisper_controller.subprocess.Popen')
    def test_stop_terminates_process(self, mock_popen, temp_input_file, temp_model_file):
        mock_process = Mock()
        mock_process.wait.return_value = None
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller, _, _, _, _ = make_controller()

        controller.start(
            input_file=temp_input_file,
            model_path=temp_model_file,
            language='en',
            threads=4
        )

        controller.stop()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)

    @patch('whisper_controller.subprocess.Popen')
    def test_running_is_false_after_stop(self, mock_popen, temp_input_file, temp_model_file):
        mock_process = Mock()
        mock_process.wait.return_value = None
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller, _, _, _, _ = make_controller()

        controller.start(
            input_file=temp_input_file,
            model_path=temp_model_file,
            language='en',
            threads=4
        )

        controller.stop()

        assert controller.running is False

    @patch('whisper_controller.subprocess.Popen')
    def test_stop_kills_if_terminate_times_out(self, mock_popen, temp_input_file, temp_model_file):
        mock_process = Mock()
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired('cmd', 5),
            None
        ]
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller, _, _, _, _ = make_controller()

        controller.start(
            input_file=temp_input_file,
            model_path=temp_model_file,
            language='en',
            threads=4
        )

        controller.stop()

        mock_process.terminate.assert_called_once()
        assert mock_process.wait.call_count == 2
        mock_process.kill.assert_called_once()

    def test_stop_is_safe_when_not_running(self):
        controller, _, _, _, _ = make_controller()
        controller.stop()
        assert controller.running is False

    @patch('whisper_controller.subprocess.Popen')
    @patch('os.path.exists', return_value=True)
    @patch('os.remove')
    def test_stop_cleans_up_temp_wav_file(self, mock_remove, mock_exists, mock_popen, temp_mp3_file, temp_model_file):
        ffmpeg_process = Mock()
        ffmpeg_process.returncode = 0
        ffmpeg_process.communicate.return_value = ('', '')

        whisper_process = Mock()
        whisper_process.wait.return_value = None
        whisper_process.stdout.readline.return_value = ''
        whisper_process.poll.return_value = None

        def popen_side_effect(cmd, **kwargs):
            if 'ffmpeg' in cmd[0]:
                return ffmpeg_process
            return whisper_process

        mock_popen.side_effect = popen_side_effect

        controller, _, _, _, _ = make_controller()

        controller.start(
            input_file=temp_mp3_file,
            model_path=temp_model_file,
            language='en',
            threads=4
        )

        temp_wav = str(Path(temp_mp3_file).with_suffix('')) + '_16k.wav'

        controller.stop()

        mock_remove.assert_called_with(temp_wav)


class TestLogCallback:
    @patch('whisper_controller.subprocess.Popen')
    def test_log_callback_receives_stdout_lines(self, mock_popen, temp_input_file, temp_model_file):
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "Transcribing audio...\n",
            "Processing segment 1\n",
            "Done.\n",
            ''
        ]
        mock_process.poll.return_value = 0
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        controller, on_log, _, on_complete, _ = make_controller()

        controller.start(
            input_file=temp_input_file,
            model_path=temp_model_file,
            language='en',
            threads=4
        )

        time.sleep(0.3)

        assert on_log.call_count >= 3

        calls = [str(c) for c in on_log.call_args_list]
        assert any("Transcribing audio" in c for c in calls)
        assert any("Processing segment 1" in c for c in calls)
        assert any("Done." in c for c in calls)


class TestNeedsConversion:
    def test_wav_returns_false(self):
        assert WhisperController.needs_conversion("test.wav") is False

    def test_WAV_returns_false(self):
        assert WhisperController.needs_conversion("test.WAV") is False

    def test_mp3_returns_true(self):
        assert WhisperController.needs_conversion("test.mp3") is True

    def test_mp4_returns_true(self):
        assert WhisperController.needs_conversion("test.mp4") is True

    def test_flac_returns_true(self):
        assert WhisperController.needs_conversion("test.flac") is True

    def test_txt_returns_true(self):
        assert WhisperController.needs_conversion("test.txt") is True

    def test_mkv_returns_true(self):
        assert WhisperController.needs_conversion("video.mkv") is True

    def test_ogg_returns_true(self):
        assert WhisperController.needs_conversion("audio.ogg") is True
