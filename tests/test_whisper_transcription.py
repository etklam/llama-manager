"""Behavior tests for the deep Whisper transcription module."""
from pathlib import Path
import shutil
from unittest.mock import patch
import wave

import pytest

from whisper_transcription import (
    CancellationToken,
    Cancelled,
    Completed,
    Failed,
    FfmpegMediaAdapter,
    TranscriptionRequest,
    WhisperCliAdapter,
    WhisperTranscriber,
    resolve_model_path,
)


class FakeMediaAdapter:
    def __init__(self, duration=60.0):
        self.duration = duration
        self.rendered_chunks = []
        self.duration_calls = 0

    def prepare_audio(self, source, destination, cancellation, emit):
        return Path(source)

    def duration_seconds(self, source, cancellation, emit):
        self.duration_calls += 1
        return self.duration

    def render_chunk(self, source, start_seconds, duration_seconds, destination,
                     cancellation, emit):
        self.rendered_chunks.append((start_seconds, duration_seconds))
        Path(destination).write_bytes(b"wav")


class FakeWhisperAdapter:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, destination, request, cancellation, emit):
        self.calls.append(Path(audio))
        Path(destination).write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nhello\n",
            encoding="utf-8",
        )


def test_success_commits_one_complete_srt(tmp_path):
    source = tmp_path / "speech.wav"
    source.write_bytes(b"wav")
    existing = tmp_path / "speech.srt"
    existing.write_text("old result", encoding="utf-8")

    transcriber = WhisperTranscriber(FakeMediaAdapter(), FakeWhisperAdapter())
    result = transcriber.transcribe(
        TranscriptionRequest(
            source=source,
            cli_path=Path("whisper-cli"),
            model_path=Path("model.bin"),
        ),
        cancellation=CancellationToken(),
    )

    assert isinstance(result, Completed)
    assert result.srt_path == existing
    assert "hello" in existing.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".whisper-*"))


def test_chunking_disabled_does_not_probe_duration(tmp_path):
    source = tmp_path / "speech.wav"
    source.write_bytes(b"wav")
    media = FakeMediaAdapter(duration=7200.0)

    result = WhisperTranscriber(media, FakeWhisperAdapter()).transcribe(
        TranscriptionRequest(
            source=source,
            cli_path=Path("whisper-cli"),
            model_path=Path("model.bin"),
            chunk_long_audio=False,
        ),
        cancellation=CancellationToken(),
    )

    assert isinstance(result, Completed)
    assert media.duration_calls == 0
    assert media.rendered_chunks == []


def test_short_media_with_chunking_enabled_stays_single_run(tmp_path):
    source = tmp_path / "speech.wav"
    source.write_bytes(b"wav")
    media = FakeMediaAdapter(duration=1800.0)
    whisper = FakeWhisperAdapter()

    result = WhisperTranscriber(media, whisper).transcribe(
        TranscriptionRequest(
            source=source,
            cli_path=Path("whisper-cli"),
            model_path=Path("model.bin"),
            chunk_long_audio=True,
        ),
        cancellation=CancellationToken(),
    )

    assert isinstance(result, Completed)
    assert media.duration_calls == 1
    assert media.rendered_chunks == []
    assert whisper.calls == [source]


def test_long_media_is_chunked_and_merged_behind_the_same_interface(tmp_path):
    source = tmp_path / "long.wav"
    source.write_bytes(b"wav")
    media = FakeMediaAdapter(duration=3600.0)
    whisper = FakeWhisperAdapter()

    result = WhisperTranscriber(media, whisper).transcribe(
        TranscriptionRequest(
            source=source,
            cli_path=Path("whisper-cli"),
            model_path=Path("model.bin"),
            chunk_long_audio=True,
        ),
        cancellation=CancellationToken(),
    )

    assert isinstance(result, Completed)
    assert media.rendered_chunks == [(0.0, 1800.0), (1800.0, 1800.0)]
    assert len(whisper.calls) == 2
    merged = result.srt_path.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:02,000" in merged
    assert "00:30:01,000 --> 00:30:02,000" in merged


def test_progress_is_reported_as_structured_stage_events(tmp_path):
    source = tmp_path / "speech.wav"
    source.write_bytes(b"wav")
    events = []

    result = WhisperTranscriber(
        FakeMediaAdapter(), FakeWhisperAdapter()
    ).transcribe(
        TranscriptionRequest(
            source=source,
            cli_path=Path("whisper-cli"),
            model_path=Path("model.bin"),
        ),
        cancellation=CancellationToken(),
        emit=events.append,
    )

    assert isinstance(result, Completed)
    assert [(event.stage, event.kind) for event in events] == [
        ("preparing", "started"),
        ("preparing", "completed"),
        ("transcribing", "started"),
        ("transcribing", "completed"),
        ("committing", "started"),
        ("committing", "completed"),
    ]


def test_cancellation_preserves_existing_srt_and_removes_staging(tmp_path):
    source = tmp_path / "speech.wav"
    source.write_bytes(b"wav")
    existing = tmp_path / "speech.srt"
    existing.write_text("previous complete result", encoding="utf-8")
    cancellation = CancellationToken()

    class CancellingWhisperAdapter(FakeWhisperAdapter):
        def transcribe(self, audio, destination, request, token, emit):
            Path(destination).write_text("partial", encoding="utf-8")
            token.cancel()

    result = WhisperTranscriber(
        FakeMediaAdapter(), CancellingWhisperAdapter()
    ).transcribe(
        TranscriptionRequest(
            source=source,
            cli_path=Path("whisper-cli"),
            model_path=Path("model.bin"),
        ),
        cancellation=cancellation,
    )

    assert isinstance(result, Cancelled)
    assert existing.read_text(encoding="utf-8") == "previous complete result"
    assert not list(tmp_path.glob(".whisper-*"))


def test_failure_identifies_stage_and_preserves_existing_srt(tmp_path):
    source = tmp_path / "speech.wav"
    source.write_bytes(b"wav")
    existing = tmp_path / "speech.srt"
    existing.write_text("previous complete result", encoding="utf-8")

    class FailingWhisperAdapter(FakeWhisperAdapter):
        def transcribe(self, audio, destination, request, cancellation, emit):
            raise RuntimeError("model rejected input")

    result = WhisperTranscriber(
        FakeMediaAdapter(), FailingWhisperAdapter()
    ).transcribe(
        TranscriptionRequest(
            source=source,
            cli_path=Path("whisper-cli"),
            model_path=Path("model.bin"),
        ),
        cancellation=CancellationToken(),
    )

    assert result == Failed("transcribing", "model rejected input")
    assert existing.read_text(encoding="utf-8") == "previous complete result"


def test_model_resolution_uses_registry_path_when_available():
    models = lambda: [{"name": "tiny", "path": "D:/models/tiny.bin"}]

    assert resolve_model_path("D:/models", "tiny", models) == "D:/models/tiny.bin"
    assert resolve_model_path("D:/models", "other.bin", models) == str(
        Path("D:/models") / "other.bin"
    )


@patch("whisper_transcription._run_process")
def test_whisper_adapter_owns_command_construction(mock_run, tmp_path):
    audio = tmp_path / "audio.wav"
    output = tmp_path / "result.srt"
    request = TranscriptionRequest(
        source=audio,
        cli_path=Path("tools/whisper-cli.exe"),
        model_path=Path("models/ggml.bin"),
        language="en",
        threads=4,
    )

    def create_output(command, cancellation, emit, stage):
        output.write_text("", encoding="utf-8")
        return ""

    mock_run.side_effect = create_output
    WhisperCliAdapter().transcribe(
        audio, output, request, CancellationToken(), lambda event: None)

    command = mock_run.call_args.args[0]
    assert command[:5] == [
        str(Path("tools/whisper-cli.exe")),
        "-m", str(Path("models/ggml.bin")),
        "-f", str(audio),
    ]
    assert command[command.index("--output-file") + 1] == str(output.with_suffix(""))
    assert "--temperature-nc" not in command
    assert "--no-context" not in command
    assert "--prompt-max-len" not in command
    assert "--no-fallback" in command
    assert command[command.index("--max-context") + 1] == "0"
    assert command[-2:] == ["--language", "en"]


def test_cancellation_at_commit_point_preserves_existing_srt(tmp_path):
    source = tmp_path / "speech.wav"
    source.write_bytes(b"wav")
    existing = tmp_path / "speech.srt"
    existing.write_text("previous complete result", encoding="utf-8")
    cancellation = CancellationToken()

    def cancel_before_commit(event):
        if event.stage == "committing" and event.kind == "started":
            cancellation.cancel()

    result = WhisperTranscriber(
        FakeMediaAdapter(), FakeWhisperAdapter()
    ).transcribe(
        TranscriptionRequest(
            source=source,
            cli_path=Path("whisper-cli"),
            model_path=Path("model.bin"),
        ),
        cancellation=cancellation,
        emit=cancel_before_commit,
    )

    assert isinstance(result, Cancelled)
    assert existing.read_text(encoding="utf-8") == "previous complete result"


def test_observer_failure_does_not_change_committed_outcome(tmp_path):
    source = tmp_path / "speech.wav"
    source.write_bytes(b"wav")

    def broken_observer(event):
        if event.stage == "committing" and event.kind == "completed":
            raise RuntimeError("UI closed")

    result = WhisperTranscriber(
        FakeMediaAdapter(), FakeWhisperAdapter()
    ).transcribe(
        TranscriptionRequest(
            source=source,
            cli_path=Path("whisper-cli"),
            model_path=Path("model.bin"),
        ),
        cancellation=CancellationToken(),
        emit=broken_observer,
    )

    assert isinstance(result, Completed)
    assert result.srt_path.exists()


def test_chunked_timestamps_do_not_wrap_after_24_hours(tmp_path):
    source = tmp_path / "very-long.wav"
    source.write_bytes(b"wav")
    media = FakeMediaAdapter(duration=25 * 60 * 60)

    result = WhisperTranscriber(media, FakeWhisperAdapter()).transcribe(
        TranscriptionRequest(
            source=source,
            cli_path=Path("whisper-cli"),
            model_path=Path("model.bin"),
            chunk_long_audio=True,
        ),
        cancellation=CancellationToken(),
    )

    assert isinstance(result, Completed)
    merged = result.srt_path.read_text(encoding="utf-8")
    assert "24:00:01,000 --> 24:00:02,000" in merged


def test_probe_rounding_does_not_create_zero_length_final_chunk(tmp_path):
    source = tmp_path / "long.wav"
    source.write_bytes(b"wav")
    media = FakeMediaAdapter(duration=3600.0004)

    result = WhisperTranscriber(media, FakeWhisperAdapter()).transcribe(
        TranscriptionRequest(
            source=source,
            cli_path=Path("whisper-cli"),
            model_path=Path("model.bin"),
            chunk_long_audio=True,
        ),
        cancellation=CancellationToken(),
    )

    assert isinstance(result, Completed)
    assert media.rendered_chunks == [(0.0, 1800.0), (1800.0, 1800.0)]


def test_output_directory_failure_is_returned_as_failed_outcome(tmp_path):
    source = tmp_path / "speech.wav"
    source.write_bytes(b"wav")
    not_a_directory = tmp_path / "blocked"
    not_a_directory.write_text("file", encoding="utf-8")

    result = WhisperTranscriber(
        FakeMediaAdapter(), FakeWhisperAdapter()
    ).transcribe(
        TranscriptionRequest(
            source=source,
            cli_path=Path("whisper-cli"),
            model_path=Path("model.bin"),
            output_dir=not_a_directory,
        ),
        cancellation=CancellationToken(),
    )

    assert isinstance(result, Failed)
    assert result.stage == "preparing"


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not installed")
def test_ffprobe_adapter_reads_real_wav_duration(tmp_path):
    source = tmp_path / "one-second.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)

    duration = FfmpegMediaAdapter().duration_seconds(
        source, CancellationToken(), lambda event: None)

    assert duration == pytest.approx(1.0, abs=0.05)
