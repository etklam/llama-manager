"""ServerController - Manages llama.cpp server process lifecycle."""
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Callable


class ServerController:
    """Manages the llama.cpp server subprocess lifecycle."""

    def __init__(self, server_exe: Path, log_callback: Callable[[str], None]):
        """Create controller.

        Args:
            server_exe: Path to llama-server.exe executable
            log_callback: Function to call with each line of server output
        """
        self._server_exe = server_exe
        self._log_callback = log_callback
        self._process: Optional[subprocess.Popen] = None
        self._running = False
        self._port: Optional[int] = None
        self._log_thread: Optional[threading.Thread] = None
        self._log_thread_stopped = False  # Flag to track if log thread ended naturally

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
              mmproj_path: str = "") -> None:
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

        # Start process
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )

        self._running = True
        self._port = port
        self._log_thread_stopped = False

        # Start log monitoring thread
        self._log_thread = threading.Thread(
            target=self._monitor_logs,
            daemon=True
        )
        self._log_thread.start()

        # Give the thread a moment to start
        time.sleep(0.01)

    def stop(self) -> None:
        """Stop the running server.

        Safe to call multiple times. Does nothing if server is not running.
        """
        if not self._running or not self._process:
            return

        try:
            # Try graceful termination first
            self._process.terminate()

            # Wait for process to end
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                self._process.kill()
                self._process.wait()

        finally:
            self._running = False
            self._port = None
            self._process = None

    @property
    def running(self) -> bool:
        """Is the server currently running?"""
        return self._running

    @property
    def port(self) -> Optional[int]:
        """The port the server is running on, or None."""
        return self._port

    def _monitor_logs(self) -> None:
        """Monitor server stdout and call log callback for each line.

        Runs in a background thread.
        """
        if not self._process:
            return

        try:
            for line in iter(self._process.stdout.readline, ''):
                if not line:
                    break

                line = line.strip()
                if line:
                    self._log_callback(line)

        except Exception as e:
            # Log errors only if server is supposed to be running
            if self._running:
                self._log_callback(f"Error monitoring server logs: {e}")

        # Server process has ended
        # Only mark as stopped if the process actually died (poll returns exit code)
        # In tests with mocks, the process might still be "running"
        if self._process and self._process.poll() is not None:
            # Process actually died
            self._running = False
            self._port = None
            self._process = None
