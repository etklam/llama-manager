"""Tests for ServerController module."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import threading
import time
import tempfile
import os

from server_controller import ServerController


@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    try:
        os.unlink(temp_path)
    except:
        pass


class TestServerControllerInitialState:
    """Test the initial state of ServerController."""

    def test_controller_starts_in_not_running_state(self):
        """ServerController should start in not-running state."""
        server_exe = Path("llama-server.exe")
        log_callback = Mock()

        controller = ServerController(server_exe, log_callback)

        assert controller.running is False
        assert controller.port is None


class TestServerStart:
    """Test server start functionality."""

    @patch('server_controller.subprocess.Popen')
    def test_start_launches_subprocess_with_correct_arguments(self, mock_popen, temp_model_file):
        """start() should launch subprocess with correct arguments."""
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''  # End the log thread
        mock_process.poll.return_value = None  # Process is still running
        mock_popen.return_value = mock_process

        server_exe = Path("llama-server.exe")
        log_callback = Mock()
        controller = ServerController(server_exe, log_callback)

        controller.start(
            model_path=temp_model_file,
            port=8080,
            host="0.0.0.0",
            gpu_layers=99,
            context_size=131072,
            batch_size=256
        )

        # Verify Popen was called with correct command
        expected_cmd = [
            str(server_exe),
            "-m", temp_model_file,
            "--port", "8080",
            "--host", "0.0.0.0",
            "-ngl", "99",
            "-c", "131072",
            "-b", "256"
        ]
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == expected_cmd

        # Verify stdout/stderr are piped
        import subprocess
        assert call_args[1]['stdout'] == subprocess.PIPE
        assert call_args[1]['stderr'] == subprocess.STDOUT
        assert call_args[1]['text'] is True
        assert call_args[1]['bufsize'] == 1

    @patch('server_controller.subprocess.Popen')
    def test_running_is_true_after_start(self, mock_popen, temp_model_file):
        """running property should be True after successful start."""
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''  # End the log thread
        mock_process.poll.return_value = None  # Process is still running
        mock_popen.return_value = mock_process

        server_exe = Path("llama-server.exe")
        log_callback = Mock()
        controller = ServerController(server_exe, log_callback)

        controller.start(
            model_path=temp_model_file,
            port=8080,
            host="0.0.0.0",
            gpu_layers=99,
            context_size=131072,
            batch_size=256
        )

        assert controller.running is True

    def test_start_raises_if_model_path_doesnt_exist(self):
        """start() should raise if model_path doesn't exist."""
        server_exe = Path("llama-server.exe")
        log_callback = Mock()
        controller = ServerController(server_exe, log_callback)

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            controller.start(
                model_path="D:\\nonexistent\\model.gguf",
                port=8080,
                host="0.0.0.0",
                gpu_layers=99,
                context_size=131072,
                batch_size=256
            )

    @patch('server_controller.subprocess.Popen')
    def test_start_raises_if_already_running(self, mock_popen, temp_model_file):
        """start() should raise if server is already running."""
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''  # End the log thread
        mock_process.poll.return_value = None  # Process is still running
        mock_popen.return_value = mock_process

        server_exe = Path("llama-server.exe")
        log_callback = Mock()
        controller = ServerController(server_exe, log_callback)

        # Start once
        controller.start(
            model_path=temp_model_file,
            port=8080,
            host="0.0.0.0",
            gpu_layers=99,
            context_size=131072,
            batch_size=256
        )

        # Try to start again - should raise
        with pytest.raises(RuntimeError, match="Server is already running"):
            controller.start(
                model_path=temp_model_file,
                port=8080,
                host="0.0.0.0",
                gpu_layers=99,
                context_size=131072,
                batch_size=256
            )


class TestServerStop:
    """Test server stop functionality."""

    @patch('server_controller.subprocess.Popen')
    def test_stop_terminates_the_process(self, mock_popen, temp_model_file):
        """stop() should terminate the server process."""
        mock_process = Mock()
        mock_process.wait.return_value = None
        mock_process.stdout.readline.return_value = ''  # End the log thread
        mock_process.poll.return_value = None  # Process is still running
        mock_popen.return_value = mock_process

        server_exe = Path("llama-server.exe")
        log_callback = Mock()
        controller = ServerController(server_exe, log_callback)

        controller.start(
            model_path=temp_model_file,
            port=8080,
            host="0.0.0.0",
            gpu_layers=99,
            context_size=131072,
            batch_size=256
        )

        controller.stop()

        # Verify terminate was called
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)

    @patch('server_controller.subprocess.Popen')
    def test_running_is_false_after_stop(self, mock_popen, temp_model_file):
        """running property should be False after stop."""
        mock_process = Mock()
        mock_process.wait.return_value = None
        mock_process.stdout.readline.return_value = ''  # End the log thread
        mock_process.poll.return_value = None  # Process is still running
        mock_popen.return_value = mock_process

        server_exe = Path("llama-server.exe")
        log_callback = Mock()
        controller = ServerController(server_exe, log_callback)

        controller.start(
            model_path=temp_model_file,
            port=8080,
            host="0.0.0.0",
            gpu_layers=99,
            context_size=131072,
            batch_size=256
        )

        controller.stop()

        assert controller.running is False

    @patch('server_controller.subprocess.Popen')
    def test_stop_kills_process_if_terminate_times_out(self, mock_popen, temp_model_file):
        """stop() should kill process if terminate times out."""
        import subprocess
        mock_process = Mock()
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired('cmd', 5),
            None
        ]  # First call times out
        mock_process.stdout.readline.return_value = ''  # End the log thread
        mock_process.poll.return_value = None  # Process is still running
        mock_popen.return_value = mock_process

        server_exe = Path("llama-server.exe")
        log_callback = Mock()
        controller = ServerController(server_exe, log_callback)

        controller.start(
            model_path=temp_model_file,
            port=8080,
            host="0.0.0.0",
            gpu_layers=99,
            context_size=131072,
            batch_size=256
        )

        controller.stop()

        # Verify both terminate and kill were called
        mock_process.terminate.assert_called_once()
        assert mock_process.wait.call_count == 2
        mock_process.kill.assert_called_once()

    def test_stop_is_safe_to_call_when_not_running(self):
        """stop() should be safe to call when server is not running."""
        server_exe = Path("llama-server.exe")
        log_callback = Mock()
        controller = ServerController(server_exe, log_callback)

        # Should not raise
        controller.stop()

        assert controller.running is False


class TestLogCallback:
    """Test log callback functionality."""

    @patch('server_controller.subprocess.Popen')
    def test_log_callback_receives_stdout_lines(self, mock_popen, temp_model_file):
        """Log callback should receive stdout lines from the process."""
        mock_process = Mock()
        # Simulate multiple lines of output, then end
        mock_process.stdout.readline.side_effect = [
            "Starting server...\n",
            "Loading model...\n",
            "Server ready on port 8080\n",
            ''  # End the log thread
        ]
        mock_process.poll.return_value = None  # Process is still running
        mock_popen.return_value = mock_process

        server_exe = Path("llama-server.exe")
        log_callback = Mock()
        controller = ServerController(server_exe, log_callback)

        controller.start(
            model_path=temp_model_file,
            port=8080,
            host="0.0.0.0",
            gpu_layers=99,
            context_size=131072,
            batch_size=256
        )

        # Wait a bit for the log thread to process
        time.sleep(0.2)

        # Verify log_callback was called with each line
        assert log_callback.call_count >= 3

        # Check the calls contain our lines
        calls = [str(call) for call in log_callback.call_args_list]
        assert any("Starting server" in call for call in calls)
        assert any("Loading model" in call for call in calls)
        assert any("Server ready" in call for call in calls)


class TestPortProperty:
    """Test port property."""

    @patch('server_controller.subprocess.Popen')
    def test_port_returns_configured_port_when_running(self, mock_popen, temp_model_file):
        """port property should return the configured port when running."""
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''  # End the log thread
        mock_process.poll.return_value = None  # Process is still running
        mock_popen.return_value = mock_process

        server_exe = Path("llama-server.exe")
        log_callback = Mock()
        controller = ServerController(server_exe, log_callback)

        controller.start(
            model_path=temp_model_file,
            port=9999,
            host="0.0.0.0",
            gpu_layers=99,
            context_size=131072,
            batch_size=256
        )

        assert controller.port == 9999

    def test_port_returns_none_when_not_running(self):
        """port property should return None when not running."""
        server_exe = Path("llama-server.exe")
        log_callback = Mock()
        controller = ServerController(server_exe, log_callback)

        assert controller.port is None

        # Also test after stop
        with patch('server_controller.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.wait.return_value = None
            mock_process.stdout.readline.return_value = ''
            mock_popen.return_value = mock_process

            # Create a temp file for this test
            with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
                temp_path = f.name

            try:
                controller.start(
                    model_path=temp_path,
                    port=8080,
                    host="0.0.0.0",
                    gpu_layers=99,
                    context_size=131072,
                    batch_size=256
                )

                controller.stop()

                assert controller.port is None
            finally:
                os.unlink(temp_path)
