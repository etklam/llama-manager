"""ServerController - Manages llama.cpp server process lifecycle."""
import subprocess
import socket
import os
from contextlib import contextmanager
from collections import deque
from functools import wraps
import threading
import time
from pathlib import Path
from typing import Optional, Callable


class ServerBusyError(RuntimeError):
    pass


def owned_operation(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._state_lock:
            self._check_owner()
            return method(self, *args, **kwargs)
    return wrapped


class ServerController:
    """Manages the llama.cpp server subprocess lifecycle."""

    def __init__(self, server_exe: Path, log_callback: Callable[[str], None]):
        """Create controller.

        Args:
            server_exe: Path to llama-server.exe executable
            log_callback: Function to call with each line of server output
        """
        self._state_lock = threading.RLock()
        self._lease_owner = None
        self._lease_name = None
        self.last_command = None
        self.logs = deque(maxlen=4000)
        self._server_exe = server_exe
        self._log_callback = log_callback
        self._process: Optional[subprocess.Popen] = None
        self._running = False
        self._port: Optional[int] = None
        self._log_thread: Optional[threading.Thread] = None
        self._log_thread_stopped = False  # Flag to track if log thread ended naturally

    @owned_operation
    def start(self, model_path: str, port: int, host: str,
              gpu_layers: int, context_size: int, batch_size: int,
              parallel: int = 1, flash_attn: bool = True,
              cont_batching: bool = True,
              cache_type_k: Optional[str] = "q8_0",
              cache_type_v: Optional[str] = "q8_0",
              dflash_enabled: bool = False,
              dflash_model_path: str = "",
              dflash_n_max: int = 6,
              dflash_gpu_layers: str = "all",
              dflash_device: str = "Vulkan0",
              mmproj_path: str = "", device: str = "",
              extra_args: tuple = ()) -> None:
        """Start the server with given params.

        Args:
            model_path: Path to the .gguf model file
            port: Port number for the server
            host: Host address to bind to
            gpu_layers: Number of GPU layers to offload
            context_size: Context window size
            batch_size: Batch size for processing
            parallel: Number of server slots (-np). >1 enables concurrent
                request handling; KV cache is split evenly across slots.
            flash_attn: Enable FlashAttention (--flash-attn on). Saves
                attention VRAM and is required by some builds for quantized KV.
            cont_batching: Enable continuous batching (--cont-batching).
            cache_type_k: KV cache K data type; defaults to "q8_0".
                Pass None to use llama.cpp's native default.
            cache_type_v: KV cache V data type; defaults to "q8_0".
                Pass None to use llama.cpp's native default.
            dflash_enabled: Enable DFlash speculative decoding.
            dflash_model_path: Path to the DFlash draft GGUF model.
            dflash_n_max: Maximum number of drafted tokens.
            dflash_gpu_layers: Draft model GPU layer setting.
            dflash_device: Optional draft model device.
            mmproj_path: Optional multimodal projector GGUF path.

        Raises:
            FileNotFoundError: If model_path doesn't exist
            RuntimeError: If server is already running
        """
        if self._running:
            raise RuntimeError("Server is already running")

        # Check model file exists
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if dflash_enabled and not Path(dflash_model_path).is_file():
            raise FileNotFoundError(
                f"DFlash draft model file not found: {dflash_model_path}"
            )
        if mmproj_path and not Path(mmproj_path).is_file():
            raise FileNotFoundError(
                f"Multimodal projector file not found: {mmproj_path}"
            )

        # Build command
        cmd = [
            str(self._server_exe),
            "-m", model_path,
            "--port", str(port),
            "--host", host,
            "-ngl", str(gpu_layers),
            "-c", str(context_size),
            "-b", str(batch_size)
        ]

        # Concurrency: enable multiple server slots so ThreadPoolExecutor
        # workers on the client side are actually processed in parallel.
        if parallel and parallel > 0:
            cmd += ["--parallel", str(parallel)]
        if cont_batching:
            cmd += ["--cont-batching"]
        if flash_attn or dflash_enabled:
            cmd += ["--flash-attn", "on"]
        # KV cache quantization (requires flash-attn on most builds).
        if cache_type_k:
            cmd += ["--cache-type-k", cache_type_k]
        if cache_type_v:
            cmd += ["--cache-type-v", cache_type_v]
        if dflash_enabled:
            cmd += [
                "--spec-draft-model", dflash_model_path,
                "--spec-type", "draft-dflash",
                "--spec-draft-n-max", str(dflash_n_max),
                "--spec-draft-ngl", str(dflash_gpu_layers),
            ]
            if dflash_device:
                cmd += ["--spec-draft-device", dflash_device]
            cmd += ["--jinja"]
        if mmproj_path:
            cmd += ["--mmproj", mmproj_path]

        if device:
            cmd += ["--device", device]
        cmd += list(extra_args)
        self.start_command(cmd, port)

    @contextmanager
    def lease(self, name):
        owner = threading.get_ident()
        with self._state_lock:
            if self._lease_owner is not None:
                raise ServerBusyError(f"Local server busy: {self._lease_name}")
            self._lease_owner, self._lease_name = owner, name
        try:
            yield self
        finally:
            with self._state_lock:
                self._lease_owner = self._lease_name = None

    def _check_owner(self):
        if self._lease_owner is not None and self._lease_owner != threading.get_ident():
            raise ServerBusyError(f"Local server busy: {self._lease_name}")

    @owned_operation
    def start_command(self, cmd, port, isolated=False):
        if self._process is not None or self._running:
            raise RuntimeError("Server is already running")
        # Never attach to an external listener, even if it speaks our API.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        self.logs = deque(maxlen=4000)
        self._isolated_logs = isolated
        self.last_command = (tuple(cmd), port, isolated)
        # Start process
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
            env=({k: v for k, v in os.environ.items()
                  if not k.startswith(('LLAMA_ARG_', 'MTMD_', 'GGML_'))}
                 if isolated else None),
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )

        self._running = True
        self._port = port
        self._log_thread_stopped = False

        # Start log monitoring thread
        self._log_thread = threading.Thread(
            target=self._monitor_logs,
            args=(self._process, self.logs, isolated),
            daemon=True
        )
        self._log_thread.start()

        # Give the thread a moment to start
        time.sleep(0.01)

    @owned_operation
    def stop(self) -> None:
        """Stop only our tracked process; retain ownership on failure."""
        process = self._process
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if self._process is process:
            self._running = False
            self._port = None
            self._process = None

    @property
    def running(self) -> bool:
        """Is the server currently running?"""
        return self._running

    def assert_alive(self):
        with self._state_lock:
            if self._process is None or self._process.poll() is not None:
                raise RuntimeError('Managed server process is no longer alive')

    @property
    def port(self) -> Optional[int]:
        """The port the server is running on, or None."""
        return self._port

    def _monitor_logs(self, process=None, log_lines=None, isolated=None) -> None:
        """Monitor server stdout and call log callback for each line.

        Runs in a background thread.
        """
        process = process or self._process
        log_lines = self.logs if log_lines is None else log_lines
        isolated = getattr(self, '_isolated_logs', False) if isolated is None else isolated
        if not process:
            return

        try:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break

                line = line.strip()
                if line:
                    log_lines.append(line)
                    if not isolated:
                        self._log_callback(line)

        except Exception as e:
            # Log errors only if server is supposed to be running
            if self._running:
                self._log_callback(f"Error monitoring server logs: {e}")

        # Server process has ended
        # Only mark as stopped if the process actually died (poll returns exit code)
        # In tests with mocks, the process might still be "running"
        with self._state_lock:
            if self._process is process and process.poll() is not None:
                self._running = False
                self._port = None
                self._process = None
