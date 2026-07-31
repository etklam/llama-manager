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
    _is_ansi_safe,
    Failed,
    FfmpegMediaAdapter,
    TranscriptionRequest,
    WhisperCliAdapter,
    WhisperTranscriber,
)
from whisper_policy import (
    CHUNK_SECONDS,
    chunking_enabled,
    chunking_label,
    decide,
    set_chunking_enabled,
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
    # 60s is plainly short; the exact threshold is the policy's to test.
    source = tmp_path / "speech.wav"
    source.write_bytes(b"wav")
    media = FakeMediaAdapter(duration=60.0)
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
    duration = 3600.0
    media = FakeMediaAdapter(duration=duration)
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
    plan = decide(duration)
    assert media.rendered_chunks == [
        (i * plan.chunk_seconds,
         min(plan.chunk_seconds, duration - i * plan.chunk_seconds))
        for i in range(plan.chunk_count)
    ]
    assert len(whisper.calls) == plan.chunk_count
    merged = result.srt_path.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:02,000" in merged
    chunk_minutes = int(plan.chunk_seconds // 60)
    assert f"00:{chunk_minutes:02d}:01,000 --> 00:{chunk_minutes:02d}:02,000" in merged


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
    duration = 25 * 60 * 60
    media = FakeMediaAdapter(duration=duration)

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
    plan = decide(duration)
    last_start = (plan.chunk_count - 1) * plan.chunk_seconds
    merged = result.srt_path.read_text(encoding="utf-8")
    assert f"{int(last_start // 3600):02d}:{int(last_start % 3600 // 60):02d}:01,000" in merged


def test_probe_rounding_does_not_create_zero_length_final_chunk(tmp_path):
    source = tmp_path / "long.wav"
    source.write_bytes(b"wav")
    duration = 3600.0004
    media = FakeMediaAdapter(duration=duration)

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
    plan = decide(duration)
    assert len(media.rendered_chunks) == plan.chunk_count
    assert all(chunk_duration > 0 for _, chunk_duration in media.rendered_chunks)


class TestChunkingPolicy:
    """The Chunking Policy is a deep module: callers and tests cross the same
    seam. The engine tests above execute the plan it returns; here the rule
    itself is pinned.
    """

    def test_short_duration_is_never_chunked(self):
        plan = decide(60.0)

        assert plan.chunked is False
        assert plan.chunk_count == 1
        assert plan.chunk_seconds == CHUNK_SECONDS

    def test_threshold_boundary(self):
        assert decide(30 * 60).chunked is False
        assert decide(30 * 60 + 0.001).chunked is False
        assert decide(30 * 60 + 0.002).chunked is True

    def test_chunk_size_and_count_come_from_the_plan(self):
        plan = decide(3600.0)

        assert plan.chunked is True
        assert plan.chunk_seconds == 1800.0
        assert plan.chunk_count == 2

    def test_probe_rounding_yields_no_zero_length_final_chunk(self):
        plan = decide(3600.0004)

        assert plan.chunked is True
        assert plan.chunk_count == 2

    def test_day_long_media_chunks_without_wrapping(self):
        assert decide(25 * 3600.0).chunk_count == 50

    def test_flag_round_trips_through_config(self, tmp_path):
        from config_manager import ConfigManager
        cm = ConfigManager(str(tmp_path / "cfg.json"))
        cm.load()

        assert chunking_enabled(cm) is False
        set_chunking_enabled(cm, True)
        assert chunking_enabled(cm) is True
        set_chunking_enabled(cm, False)
        assert chunking_enabled(cm) is False

    def test_label_is_derived_from_chunk_seconds(self):
        assert chunking_label() == (
            f"Chunk long audio ({CHUNK_SECONDS // 60} min, anti-repeat)"
        )


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


class TestAnsiUnsafePaths:
    """whisper-cli receives paths as narrow ANSI bytes, not UTF-16.

    A character the active codepage cannot encode arrives at the tool as a
    literal "?", so it opens nothing and reports a not-found path that looks
    plausible (e.g. "D:\\collection\\sorted\\???\\prepared.wav"). Such media is
    staged through an ANSI-safe scratch directory instead. The predicate is
    patched in these tests so they assert the routing on any codepage.
    """

    def _request(self, audio, **kw):
        params = dict(
            source=audio,
            cli_path=Path("whisper-cli.exe"),
            model_path=Path("D:/models/ggml.bin"),
        )
        params.update(kw)
        return TranscriptionRequest(**params)

    @patch("whisper_transcription._run_process")
    def test_representable_path_runs_directly_without_staging(self, mock_run, tmp_path):
        audio = tmp_path / "prepared.wav"
        audio.write_bytes(b"wav")
        destination = tmp_path / "result.srt"

        def create_output(command, cancellation, emit, stage):
            destination.write_text("srt", encoding="utf-8")
            return ""

        mock_run.side_effect = create_output
        with patch("whisper_transcription._is_ansi_safe", return_value=True):
            WhisperCliAdapter().transcribe(
                audio, destination, self._request(audio),
                CancellationToken(), lambda event: None)

        # The real path went straight to the tool: no scratch indirection.
        command = mock_run.call_args.args[0]
        assert command[command.index("-f") + 1] == str(audio)
        assert destination.read_text(encoding="utf-8") == "srt"

    @patch("whisper_transcription._run_process")
    def test_unrepresentable_path_is_staged_and_srt_returned(self, mock_run, tmp_path):
        # Mirrors the reported failure: a Japanese folder unencodable in cp950.
        media_dir = tmp_path / "七瀬ひな"
        media_dir.mkdir()
        audio = media_dir / "prepared.wav"
        audio.write_bytes(b"wav")
        destination = media_dir / "result.srt"
        seen = {}

        def create_output(command, cancellation, emit, stage):
            audio_arg = command[command.index("-f") + 1]
            base_arg = command[command.index("--output-file") + 1]
            seen["audio"] = audio_arg
            seen["base"] = base_arg
            # The staged input must really exist where the tool is told to read.
            assert Path(audio_arg).read_bytes() == b"wav"
            Path(base_arg).with_suffix(".srt").write_text(
                "staged srt", encoding="utf-8")
            return ""

        mock_run.side_effect = create_output

        def only_scratch_is_safe(text):
            return "瀬" not in text

        with patch("whisper_transcription._is_ansi_safe",
                   side_effect=only_scratch_is_safe):
            WhisperCliAdapter().transcribe(
                audio, destination, self._request(audio),
                CancellationToken(), lambda event: None)

        # whisper-cli never saw the unrepresentable path...
        assert "瀬" not in seen["audio"]
        assert "瀬" not in seen["base"]
        # ...yet the SRT landed at the real destination, and staging is gone.
        assert destination.read_text(encoding="utf-8") == "staged srt"
        assert not Path(seen["audio"]).exists()

    @patch("whisper_transcription._run_process")
    def test_unrepresentable_model_path_fails_with_a_clear_reason(
        self, mock_run, tmp_path
    ):
        media_dir = tmp_path / "七瀬ひな"
        media_dir.mkdir()
        audio = media_dir / "prepared.wav"
        audio.write_bytes(b"wav")

        # Staging cannot rescue a model path the tool also cannot open.
        with patch("whisper_transcription._is_ansi_safe",
                   side_effect=lambda text: "瀬" not in text):
            with pytest.raises(RuntimeError, match="ANSI"):
                WhisperCliAdapter().transcribe(
                    audio, media_dir / "result.srt",
                    self._request(audio, model_path=media_dir / "ggml.bin"),
                    CancellationToken(), lambda event: None)

        mock_run.assert_not_called()

    @patch("whisper_transcription._run_process")
    def test_staged_run_producing_no_srt_is_reported_as_failure(
        self, mock_run, tmp_path
    ):
        media_dir = tmp_path / "七瀬ひな"
        media_dir.mkdir()
        audio = media_dir / "prepared.wav"
        audio.write_bytes(b"wav")
        mock_run.return_value = ""

        with patch("whisper_transcription._is_ansi_safe",
                   side_effect=lambda text: "瀬" not in text):
            with pytest.raises(RuntimeError, match="no SRT"):
                WhisperCliAdapter().transcribe(
                    audio, media_dir / "result.srt", self._request(audio),
                    CancellationToken(), lambda event: None)

    def test_ansi_safe_accepts_plain_ascii_paths(self):
        assert _is_ansi_safe("D:/media/clip.wav") is True
