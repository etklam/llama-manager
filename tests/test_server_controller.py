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

        # Verify Popen was called with the core arguments. Concurrency/KV
        # flags default on, so assert the prefix rather than exact equality.
        expected_prefix = [
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
        cmd = call_args[0][0]
        assert cmd[:len(expected_prefix)] == expected_prefix
        assert cmd[cmd.index("--cache-type-k") + 1] == "q8_0"
        assert cmd[cmd.index("--cache-type-v") + 1] == "q8_0"

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


class TestConcurrencyAndKVFlags:
    """Test that concurrency and KV-cache flags are emitted correctly."""

    @patch('server_controller.subprocess.Popen')
    def test_parallel_greater_than_one_adds_parallel_flag(self, mock_popen, temp_model_file):
        """parallel > 1 should add --parallel N to the command."""
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller = ServerController(Path("llama-server.exe"), Mock())
        controller.start(
            model_path=temp_model_file, port=8080, host="0.0.0.0",
            gpu_layers=99, context_size=16384, batch_size=512,
            parallel=3
        )

        cmd = mock_popen.call_args[0][0]
        assert "--parallel" in cmd
        assert cmd[cmd.index("--parallel") + 1] == "3"

    @patch('server_controller.subprocess.Popen')
    def test_parallel_one_sets_single_slot(self, mock_popen, temp_model_file):
        """parallel == 1 should override llama-server's backend default."""
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller = ServerController(Path("llama-server.exe"), Mock())
        controller.start(
            model_path=temp_model_file, port=8080, host="0.0.0.0",
            gpu_layers=99, context_size=16384, batch_size=512,
            parallel=1
        )

        cmd = mock_popen.call_args[0][0]
        assert cmd[cmd.index("--parallel") + 1] == "1"

    @patch('server_controller.subprocess.Popen')
    def test_flash_attn_and_cont_batching_flags(self, mock_popen, temp_model_file):
        """flash_attn and cont_batching should emit their flags when enabled."""
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller = ServerController(Path("llama-server.exe"), Mock())
        controller.start(
            model_path=temp_model_file, port=8080, host="0.0.0.0",
            gpu_layers=99, context_size=16384, batch_size=512,
            flash_attn=True, cont_batching=True
        )

        cmd = mock_popen.call_args[0][0]
        assert "--cont-batching" in cmd
        assert "--flash-attn" in cmd
        assert cmd[cmd.index("--flash-attn") + 1] == "on"

    @patch('server_controller.subprocess.Popen')
    def test_flags_omitted_when_disabled(self, mock_popen, temp_model_file):
        """flash_attn / cont_batching False should omit their flags."""
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller = ServerController(Path("llama-server.exe"), Mock())
        controller.start(
            model_path=temp_model_file, port=8080, host="0.0.0.0",
            gpu_layers=99, context_size=16384, batch_size=512,
            flash_attn=False, cont_batching=False
        )

        cmd = mock_popen.call_args[0][0]
        assert "--flash-attn" not in cmd
        assert "--cont-batching" not in cmd

    @patch('server_controller.subprocess.Popen')
    def test_kv_cache_type_flags(self, mock_popen, temp_model_file):
        """cache_type_k/v should emit -ctk/-ctv equivalents when set."""
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller = ServerController(Path("llama-server.exe"), Mock())
        controller.start(
            model_path=temp_model_file, port=8080, host="0.0.0.0",
            gpu_layers=99, context_size=16384, batch_size=512,
            cache_type_k="q8_0", cache_type_v="q8_0"
        )

        cmd = mock_popen.call_args[0][0]
        assert cmd[cmd.index("--cache-type-k") + 1] == "q8_0"
        assert cmd[cmd.index("--cache-type-v") + 1] == "q8_0"

    @patch('server_controller.subprocess.Popen')
    def test_kv_cache_type_omitted_when_none(self, mock_popen, temp_model_file):
        """cache_type_k/v None should omit the flags (keeps f16 default)."""
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller = ServerController(Path("llama-server.exe"), Mock())
        controller.start(
            model_path=temp_model_file, port=8080, host="0.0.0.0",
            gpu_layers=99, context_size=16384, batch_size=512,
            cache_type_k=None, cache_type_v=None
        )

        cmd = mock_popen.call_args[0][0]
        assert "--cache-type-k" not in cmd
        assert "--cache-type-v" not in cmd


class TestDFlashFlags:
    """Test DFlash and multimodal launch behavior."""

    @patch('server_controller.subprocess.Popen')
    def test_dflash_adds_speculative_flags_and_jinja(self, mock_popen, temp_model_file, tmp_path):
        draft_model = tmp_path / "dflash-kquant.gguf"
        draft_model.write_bytes(b"gguf")
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller = ServerController(Path("llama-server.exe"), Mock())
        controller.start(
            model_path=temp_model_file, port=8080, host="0.0.0.0",
            gpu_layers=99, context_size=16384, batch_size=512,
            flash_attn=False,
            dflash_enabled=True, dflash_model_path=str(draft_model),
            dflash_n_max=8, dflash_gpu_layers="all",
            dflash_device="Vulkan1",
        )

        cmd = mock_popen.call_args[0][0]
        assert cmd[cmd.index("--spec-draft-model") + 1] == str(draft_model)
        assert cmd[cmd.index("--spec-type") + 1] == "draft-dflash"
        assert cmd[cmd.index("--spec-draft-n-max") + 1] == "8"
        assert cmd[cmd.index("--spec-draft-ngl") + 1] == "all"
        assert cmd[cmd.index("--spec-draft-device") + 1] == "Vulkan1"
        assert cmd[cmd.index("--flash-attn") + 1] == "on"
        assert "--jinja" in cmd

    def test_dflash_requires_existing_draft_model(self, temp_model_file, tmp_path):
        controller = ServerController(Path("llama-server.exe"), Mock())
        missing = tmp_path / "missing-dflash.gguf"

        with pytest.raises(FileNotFoundError, match="DFlash draft model file not found"):
            controller.start(
                model_path=temp_model_file, port=8080, host="0.0.0.0",
                gpu_layers=99, context_size=16384, batch_size=512,
                dflash_enabled=True, dflash_model_path=str(missing),
            )

    @patch('server_controller.subprocess.Popen')
    def test_mmproj_is_validated_and_added_without_dflash(self, mock_popen, temp_model_file, tmp_path):
        mmproj = tmp_path / "mmproj-kquant.gguf"
        mmproj.write_bytes(b"gguf")
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ''
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        controller = ServerController(Path("llama-server.exe"), Mock())
        controller.start(
            model_path=temp_model_file, port=8080, host="0.0.0.0",
            gpu_layers=99, context_size=16384, batch_size=512,
            mmproj_path=str(mmproj),
        )

        cmd = mock_popen.call_args[0][0]
        assert cmd[cmd.index("--mmproj") + 1] == str(mmproj)
        assert "--spec-draft-model" not in cmd

    def test_mmproj_requires_existing_file(self, temp_model_file, tmp_path):
        controller = ServerController(Path("llama-server.exe"), Mock())
        missing = tmp_path / "missing-mmproj.gguf"

        with pytest.raises(FileNotFoundError, match="Multimodal projector file not found"):
            controller.start(
                model_path=temp_model_file, port=8080, host="0.0.0.0",
                gpu_layers=99, context_size=16384, batch_size=512,
                mmproj_path=str(missing),
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
