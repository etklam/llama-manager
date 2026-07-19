"""Deep synchronous lifecycle for whisper.cpp transcription."""
from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Union

from constants import VIDEO_EXTENSIONS
from utils.srt_parser import generate_srt_from_list, parse_srt_from_file


DEFAULT_CHUNK_SECONDS = 30 * 60
_CHUNK_TIME_EPSILON = 0.001


@dataclass(frozen=True)
class TranscriptionRequest:
    source: Path
    cli_path: Path
    model_path: Path
    language: str = "auto"
    threads: int = 8
    output_dir: Optional[Path] = None
    chunk_long_audio: bool = False


@dataclass(frozen=True)
class Completed:
    srt_path: Path


@dataclass(frozen=True)
class Cancelled:
    pass


@dataclass(frozen=True)
class Failed:
    stage: str
    message: str


TranscriptionOutcome = Union[Completed, Cancelled, Failed]


@dataclass(frozen=True)
class TranscriptionEvent:
    stage: str
    kind: str
    message: Optional[str] = None
    current: Optional[int] = None
    total: Optional[int] = None


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()
        self._commit_lock = threading.Lock()

    def cancel(self) -> None:
        with self._commit_lock:
            self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def _commit(self, action: Callable[[], None]) -> bool:
        """Run the publication commit only if cancellation has not won."""
        with self._commit_lock:
            if self._event.is_set():
                return False
            action()
            return True


class MediaAdapter(Protocol):
    def prepare_audio(self, source: Path, destination: Path,
                      cancellation: CancellationToken, emit: Callable) -> Path: ...

    def duration_seconds(self, source: Path, cancellation: CancellationToken,
                         emit: Callable) -> float: ...

    def render_chunk(self, source: Path, start_seconds: float,
                     duration_seconds: float, destination: Path,
                     cancellation: CancellationToken, emit: Callable) -> None: ...


class WhisperAdapter(Protocol):
    def transcribe(self, audio: Path, destination: Path,
                   request: TranscriptionRequest,
                   cancellation: CancellationToken, emit: Callable) -> None: ...


class _ProcessCancelled(Exception):
    pass


class _StageFailure(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt" and getattr(process, "pid", None):
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=5,
            )
            process.wait(timeout=5)
            return
        except (OSError, subprocess.SubprocessError):
            pass

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_process(
    command: List[str],
    cancellation: CancellationToken,
    emit: Callable[[TranscriptionEvent], None],
    stage: str,
) -> str:
    if cancellation.cancelled:
        raise _ProcessCancelled()

    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        universal_newlines=True,
        creationflags=creationflags,
    )
    lines: List[str] = []

    def read_output() -> None:
        if process.stdout is None:
            return
        for raw_line in iter(process.stdout.readline, ""):
            line = raw_line.rstrip()
            if line:
                lines.append(line)
                emit(TranscriptionEvent(stage, "diagnostic", message=line))

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    while process.poll() is None:
        if cancellation.cancelled:
            _terminate_process(process)
            reader.join(timeout=1)
            raise _ProcessCancelled()
        time.sleep(0.05)

    reader.join(timeout=1)
    if cancellation.cancelled:
        raise _ProcessCancelled()
    tool_error = next(
        (line for line in lines if line.lower().startswith("error:")), None)
    if tool_error:
        raise RuntimeError(tool_error)
    if process.returncode != 0:
        detail = lines[-1] if lines else f"process exited with code {process.returncode}"
        raise RuntimeError(detail)
    return "\n".join(lines)


def _detect_hwaccel_args() -> List[str]:
    if platform.system() != "Windows":
        return []
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=5,
            encoding="utf-8",
            errors="replace",
        )
        if "d3d11va" in result.stdout:
            return ["-hwaccel", "d3d11va", "-hwaccel_output_format", "d3d11"]
    except Exception:
        pass
    return []


class FfmpegMediaAdapter:
    _hwaccel_args: Optional[List[str]] = None

    @classmethod
    def get_hwaccel_args(cls) -> List[str]:
        if cls._hwaccel_args is None:
            cls._hwaccel_args = _detect_hwaccel_args()
        return cls._hwaccel_args

    def prepare_audio(
        self,
        source: Path,
        destination: Path,
        cancellation: CancellationToken,
        emit: Callable[[TranscriptionEvent], None],
    ) -> Path:
        if source.suffix.lower() == ".wav":
            return source

        command = ["ffmpeg", "-y"]
        if source.suffix.lower() in VIDEO_EXTENSIONS:
            command.extend(self.get_hwaccel_args())
        command.extend([
            "-i", str(source),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(destination),
        ])
        _run_process(command, cancellation, emit, "preparing")
        if not destination.exists():
            raise RuntimeError("ffmpeg produced no WAV")
        return destination

    def duration_seconds(
        self,
        source: Path,
        cancellation: CancellationToken,
        emit: Callable[[TranscriptionEvent], None],
    ) -> float:
        output = _run_process([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(source),
        ], cancellation, emit, "probing")
        try:
            return float(output.strip().splitlines()[-1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"ffprobe returned invalid duration: {output!r}") from exc

    def render_chunk(
        self,
        source: Path,
        start_seconds: float,
        duration_seconds: float,
        destination: Path,
        cancellation: CancellationToken,
        emit: Callable[[TranscriptionEvent], None],
    ) -> None:
        _run_process([
            "ffmpeg", "-y",
            "-ss", f"{start_seconds:.3f}",
            "-t", f"{duration_seconds:.3f}",
            "-i", str(source),
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(destination),
        ], cancellation, emit, "chunking")
        if not destination.exists():
            raise RuntimeError("ffmpeg produced no chunk WAV")


class WhisperCliAdapter:
    def transcribe(
        self,
        audio: Path,
        destination: Path,
        request: TranscriptionRequest,
        cancellation: CancellationToken,
        emit: Callable[[TranscriptionEvent], None],
    ) -> None:
        output_base = destination.with_suffix("")
        command = [
            str(request.cli_path),
            "-m", str(request.model_path),
            "-f", str(audio),
            "--output-srt",
            "--output-file", str(output_base),
            "-t", str(request.threads),
            "--max-len", "0",
            "--entropy-thold", "2.4",
            "--logprob-thold", "-1.0",
            "--no-speech-thold", "0.6",
            "--temperature", "0.0",
            "--no-fallback",
            "--max-context", "0",
        ]
        if request.language and request.language != "auto":
            command.extend(["--language", request.language])

        _run_process(command, cancellation, emit, "transcribing")
        if not destination.exists():
            raise RuntimeError("whisper-cli produced no SRT")


class WhisperTranscriber:
    def __init__(
        self,
        media: Optional[MediaAdapter] = None,
        whisper: Optional[WhisperAdapter] = None,
    ):
        self._media = media or FfmpegMediaAdapter()
        self._whisper = whisper or WhisperCliAdapter()

    def transcribe(
        self,
        request: TranscriptionRequest,
        *,
        cancellation: CancellationToken,
        emit: Optional[Callable[[TranscriptionEvent], None]] = None,
    ) -> TranscriptionOutcome:
        observer = emit or (lambda event: None)

        def safe_emit(event: TranscriptionEvent) -> None:
            try:
                observer(event)
            except Exception:
                pass

        emit = safe_emit
        source = Path(request.source)
        output_dir = Path(request.output_dir) if request.output_dir else source.parent
        final_srt = output_dir / f"{source.stem}.srt"
        workspace: Optional[Path] = None

        stage = "preparing"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            workspace = Path(tempfile.mkdtemp(prefix=".whisper-", dir=output_dir))
            if not source.exists():
                return Failed(stage, f"Input file not found: {source}")
            if cancellation.cancelled:
                return Cancelled()

            emit(TranscriptionEvent(stage, "started"))
            prepared = self._media.prepare_audio(
                source, workspace / "prepared.wav", cancellation, emit)
            if cancellation.cancelled:
                return Cancelled()
            emit(TranscriptionEvent(stage, "completed"))

            candidate = workspace / "result.srt"
            if request.chunk_long_audio:
                stage = "probing"
                emit(TranscriptionEvent(stage, "started"))
                duration = self._media.duration_seconds(prepared, cancellation, emit)
                if cancellation.cancelled:
                    return Cancelled()
                emit(TranscriptionEvent(stage, "completed"))
                if duration > DEFAULT_CHUNK_SECONDS + _CHUNK_TIME_EPSILON:
                    stage = "chunking"
                    emit(TranscriptionEvent(stage, "started"))
                    self._transcribe_chunks(
                        prepared, duration, workspace, candidate,
                        request, cancellation, emit)
                    if cancellation.cancelled:
                        return Cancelled()
                    emit(TranscriptionEvent(stage, "completed"))
                else:
                    self._transcribe_one(
                        prepared, candidate, request, cancellation, emit)
            else:
                self._transcribe_one(
                    prepared, candidate, request, cancellation, emit)

            if cancellation.cancelled:
                return Cancelled()
            if not candidate.exists():
                return Failed(stage, "whisper-cli produced no SRT")

            stage = "committing"
            emit(TranscriptionEvent(stage, "started"))
            committed = cancellation._commit(
                lambda: os.replace(candidate, final_srt))
            if not committed:
                return Cancelled()
            emit(TranscriptionEvent(stage, "completed"))
            return Completed(final_srt)
        except _ProcessCancelled:
            return Cancelled()
        except _StageFailure as exc:
            if cancellation.cancelled:
                return Cancelled()
            return Failed(exc.stage, str(exc))
        except Exception as exc:
            if cancellation.cancelled:
                return Cancelled()
            return Failed(stage, str(exc))
        finally:
            if workspace is not None:
                shutil.rmtree(workspace, ignore_errors=True)

    def _transcribe_one(
        self,
        audio: Path,
        candidate: Path,
        request: TranscriptionRequest,
        cancellation: CancellationToken,
        emit: Callable[[TranscriptionEvent], None],
    ) -> None:
        stage = "transcribing"
        emit(TranscriptionEvent(stage, "started"))
        try:
            self._whisper.transcribe(
                audio, candidate, request, cancellation, emit)
        except _ProcessCancelled:
            raise
        except Exception as exc:
            raise _StageFailure(stage, str(exc)) from exc
        emit(TranscriptionEvent(stage, "completed"))

    def _transcribe_chunks(
        self,
        prepared: Path,
        duration: float,
        workspace: Path,
        candidate: Path,
        request: TranscriptionRequest,
        cancellation: CancellationToken,
        emit: Callable[[TranscriptionEvent], None],
    ) -> None:
        chunk_srts: List[Path] = []
        offsets: List[float] = []
        chunk_count = max(
            1,
            math.ceil(
                (duration - _CHUNK_TIME_EPSILON) / DEFAULT_CHUNK_SECONDS
            ),
        )

        for index in range(chunk_count):
            if cancellation.cancelled:
                return
            start = index * DEFAULT_CHUNK_SECONDS
            chunk_duration = min(DEFAULT_CHUNK_SECONDS, duration - start)
            audio_path = workspace / f"chunk-{index:03d}.wav"
            srt_path = workspace / f"chunk-{index:03d}.srt"
            emit(TranscriptionEvent(
                "chunking", "progress", current=index + 1, total=chunk_count))
            try:
                self._media.render_chunk(
                    prepared, start, chunk_duration, audio_path,
                    cancellation, emit)
            except _ProcessCancelled:
                raise
            except Exception as exc:
                raise _StageFailure("chunking", str(exc)) from exc
            if cancellation.cancelled:
                return

            emit(TranscriptionEvent(
                "transcribing", "started", current=index + 1, total=chunk_count))
            try:
                self._whisper.transcribe(
                    audio_path, srt_path, request, cancellation, emit)
            except _ProcessCancelled:
                raise
            except Exception as exc:
                raise _StageFailure("transcribing", str(exc)) from exc
            if not srt_path.exists():
                raise _StageFailure(
                    "transcribing", f"chunk {index + 1} produced no SRT")
            emit(TranscriptionEvent(
                "transcribing", "completed", current=index + 1, total=chunk_count))
            chunk_srts.append(srt_path)
            offsets.append(start)

        emit(TranscriptionEvent("merging", "started"))
        try:
            merged = []
            for srt_path, offset in zip(chunk_srts, offsets):
                offset_ms = int(round(offset * 1000))
                for subtitle in parse_srt_from_file(str(srt_path)):
                    merged.append({
                        "start_time": subtitle["start_time"] + offset_ms,
                        "end_time": subtitle["end_time"] + offset_ms,
                        "text": subtitle["text"],
                    })
            candidate.write_text(generate_srt_from_list(merged), encoding="utf-8")
        except Exception as exc:
            raise _StageFailure("merging", str(exc)) from exc
        emit(TranscriptionEvent("merging", "completed"))


def resolve_model_path(model_dir, model_name, get_whisper_models_fn=None):
    if not model_dir or not model_name:
        return model_name
    if get_whisper_models_fn:
        for model in get_whisper_models_fn():
            if model.get("name") == model_name:
                return model.get("path", model_name)
    return str(Path(model_dir) / model_name)
