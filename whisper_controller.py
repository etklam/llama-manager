"""WhisperController - Manages whisper.cpp speech recognition process lifecycle."""
import os
import platform
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Callable

from constants import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS


def _detect_hwaccel_args():
    if platform.system() != 'Windows':
        return []
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-hwaccels'],
            capture_output=True, text=True, timeout=5)
        if 'd3d11va' in result.stdout:
            return ['-hwaccel', 'd3d11va', '-hwaccel_output_format', 'd3d11']
    except Exception:
        pass
    return []


class WhisperController:

    _hwaccel_args = None

    @classmethod
    def get_hwaccel_args(cls):
        if cls._hwaccel_args is None:
            cls._hwaccel_args = _detect_hwaccel_args()
        return cls._hwaccel_args

    def __init__(self, cli_path: Path, on_log: Callable[[str], None],
                 on_progress: Callable[[str], None],
                 on_complete: Callable[[str], None],
                 on_error: Callable[[str], None]):
        self._cli_path = cli_path
        self._on_log = on_log
        self._on_progress = on_progress
        self._on_complete = on_complete
        self._on_error = on_error
        self._process: Optional[subprocess.Popen] = None
        self._running = False
        self._current_file: Optional[str] = None
        self._log_thread: Optional[threading.Thread] = None
        self._temp_wav: Optional[str] = None
        self._output_srt: Optional[str] = None

    def start(self, input_file: str, model_path: str, language: str = 'auto',
              threads: int = 8, output_dir: Optional[str] = None) -> None:
        if self._running:
            raise RuntimeError("Whisper is already running")

        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        wav_file = input_file
        if self.needs_conversion(input_file):
            self._on_log(f"Converting to WAV: {input_path.name}")
            wav_file = self._convert_audio(input_file)
            self._temp_wav = wav_file
            self._on_log(f"Conversion done: {Path(wav_file).name}")

        self._on_log(f"Starting whisper-cli...")

        stem = input_path.stem
        if output_dir:
            output_base = str(Path(output_dir) / stem)
        else:
            output_base = str(input_path.with_suffix(''))

        self._output_srt = output_base + '.srt'

        cmd = [
            str(self._cli_path),
            "-m", model_path,
            "-f", wav_file,
            "--output-srt",
            "--output-file", output_base,
            "-t", str(threads),
            "--max-len", "200",
            "--entropy-thold", "2.4",
            "--logprob-thold", "-1.0",
            "--no-speech-thold", "0.6",
            "--suppress-nst",
        ]

        if language and language != 'auto':
            cmd.extend(["--language", language])

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
        self._current_file = input_file

        self._log_thread = threading.Thread(
            target=self._monitor_output,
            daemon=True
        )
        self._log_thread.start()

        time.sleep(0.01)

    def stop(self) -> None:
        if not self._running or not self._process:
            return

        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        finally:
            self._running = False
            self._current_file = None
            self._process = None

            self._cleanup_temp_wav()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def current_file(self) -> Optional[str]:
        return self._current_file

    def _convert_audio(self, input_file: str) -> str:
        ext = Path(input_file).suffix.lower()
        is_video = ext in VIDEO_EXTENSIONS

        if is_video:
            audio_file = self._extract_audio_only(input_file)
            try:
                wav_file = self._to_wav(audio_file)
            finally:
                self._remove_file(audio_file)
            return wav_file

        return self._to_wav(input_file)

    def _extract_audio_only(self, input_file: str) -> str:
        tmp_dir = tempfile.gettempdir()
        audio_path = os.path.join(tmp_dir, Path(input_file).stem + '_audio.aac')

        cmd = [
            'ffmpeg',
            '-y',
            *self.get_hwaccel_args(),
            '-i', input_file,
            '-vn',
            '-c:a', 'aac',
            '-q:a', '2',
            audio_path
        ]

        self._on_log(f"Extracting audio from video (stream copy): {Path(input_file).name}")
        self._on_log(f"ffmpeg: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding='utf-8', errors='replace')
        _, stderr = process.communicate(timeout=3600)

        if process.returncode != 0:
            err_msg = stderr.strip().split('\n')[-1] if stderr else 'unknown error'
            self._on_log(f"ffmpeg error: {err_msg}")
            raise RuntimeError(f"ffmpeg audio extraction failed: {err_msg}")

        return audio_path

    def _to_wav(self, input_file: str) -> str:
        output_path = str(Path(input_file).with_suffix('')) + '_16k.wav'

        ext = Path(input_file).suffix.lower()
        hwaccel_args = self.get_hwaccel_args() if ext in VIDEO_EXTENSIONS else []

        cmd = [
            'ffmpeg',
            '-y',
            *hwaccel_args,
            '-i', input_file,
            '-ac', '1',
            '-ar', '16000',
            '-c:a', 'pcm_s16le',
            output_path
        ]

        self._on_log(f"ffmpeg: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding='utf-8', errors='replace')
        _, stderr = process.communicate(timeout=3600)

        if process.returncode != 0:
            err_msg = stderr.strip().split('\n')[-1] if stderr else 'unknown error'
            self._on_log(f"ffmpeg error: {err_msg}")
            raise RuntimeError(f"ffmpeg conversion failed: {err_msg}")

        return output_path

    def _monitor_output(self) -> None:
        if not self._process:
            return

        try:
            for line in iter(self._process.stdout.readline, ''):
                if not line:
                    break

                line = line.strip()
                if line:
                    self._on_log(line)
        except Exception as e:
            if self._running:
                self._on_error(f"Error monitoring whisper output: {e}")

        if self._process and self._process.poll() is not None:
            exit_code = self._process.returncode
            if exit_code == 0 and self._output_srt:
                self._on_complete(self._output_srt)
            elif self._running:
                self._on_error(f"whisper-cli exited with code {exit_code}")

            self._cleanup_temp_wav()

            self._running = False
            self._current_file = None
            self._process = None

    @staticmethod
    def _remove_file(path: str) -> None:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    def _cleanup_temp_wav(self):
        if self._temp_wav and os.path.exists(self._temp_wav):
            try:
                os.remove(self._temp_wav)
                self._on_log(f"Cleaned up temp file: {Path(self._temp_wav).name}")
            except OSError:
                pass
        self._temp_wav = None

    @staticmethod
    def needs_conversion(filepath: str) -> bool:
        ext = Path(filepath).suffix.lower()
        if ext == '.wav':
            return False
        return ext in AUDIO_EXTENSIONS or ext in VIDEO_EXTENSIONS or ext not in {'.wav'}

    @staticmethod
    def resolve_model_path(model_dir, model_name, get_whisper_models_fn=None):
        if not model_dir or not model_name:
            return model_name
        if get_whisper_models_fn:
            for m in get_whisper_models_fn():
                if m.get('name') == model_name:
                    return m.get('path', model_name)
        return str(Path(model_dir) / model_name)

    @staticmethod
    def transcribe_sync(
        cli_path, filepath, model_path, language='auto', threads=8,
        on_log=None, check_stop=None, output_dir=None,
    ):
        completed = threading.Event()
        result = {'srt': None, 'error': None}

        def on_ok(srt_path):
            result['srt'] = srt_path
            completed.set()

        def on_err(msg):
            result['error'] = msg
            completed.set()

        log_fn = on_log or (lambda m: None)
        controller = WhisperController(
            cli_path=Path(cli_path),
            on_log=log_fn,
            on_progress=log_fn,
            on_complete=on_ok,
            on_error=on_err,
        )
        controller.start(filepath, model_path, language, threads, output_dir=output_dir)

        while not completed.is_set():
            completed.wait(timeout=0.2)
            if check_stop and check_stop():
                controller.stop()
                return None

        if result['error']:
            raise RuntimeError(result['error'])
        return result['srt']
