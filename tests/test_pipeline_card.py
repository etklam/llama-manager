"""Headless tests for PipelineCard log routing and model resolution."""
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline_card import PipelineCard
from pipeline_runner import PipelineRunner
from translation.preflight import PreflightPlan
from translation.server_probe import ServerInfo


def _make_card():
    card = object.__new__(PipelineCard)
    card._on_log = Mock()
    card._on_debug_log = Mock()
    card._pipe_status_label = Mock()

    toplevel = Mock()
    toplevel.after = lambda delay, fn: fn()
    card.winfo_toplevel = lambda: toplevel
    return card


def test_failed_pipeline_is_not_reported_as_all_done():
    card = _make_card()
    card._pipe_start_btn = Mock()
    card._pipe_stop_btn = Mock()
    card._pipeline_runner = Mock(failed_files=1, startup_error=None)
    card._update_pipeline_done(False)
    card._pipe_status_label.config.assert_called_with(
        text='1 file(s) failed — see log', foreground='red')


def test_startup_error_remains_visible_instead_of_stopped():
    card = _make_card()
    card._pipe_start_btn = Mock()
    card._pipe_stop_btn = Mock()
    card._pipeline_runner = Mock(startup_error='HTTP 401', failed_files=0)
    card._update_pipeline_done(False)
    card._pipe_status_label.config.assert_called_with(
        text='Cannot start: HTTP 401', foreground='red')


def test_pipeline_step_emits_once_on_pipeline_log_callback():
    """Debug subscribes to the bus, so a second callback duplicates the line."""
    card = _make_card()

    card._pipeline_step("Translating a.srt: 10/20")

    card._on_log.assert_called_once_with("INFO", "Translating a.srt: 10/20")
    card._on_debug_log.assert_not_called()


def test_pipeline_error_updates_status_in_red_without_duplicate_log():
    card = _make_card()

    card._pipeline_step("Error: llama-server not reachable")

    card._pipe_status_label.config.assert_called_once_with(
        text="Error: llama-server not reachable", foreground="red")
    card._on_log.assert_called_once()
    card._on_debug_log.assert_not_called()


def test_unreachable_preflight_emits_one_pipeline_log_line():
    """The progress callback already publishes the preflight error to the log."""
    card = _make_card()
    config = Mock()
    config.get.return_value = 3
    message = (
        "llama-server 未回應 (http://localhost:8080/props): "
        "ReadError: [WinError 10054]\n"
        "請先在 Server 分頁啟動伺服器並等模型載入完成。"
    )
    runner = PipelineRunner(
        config_manager=config,
        get_port=lambda: 8080,
        get_current_model=lambda: "test-model",
        resolve_whisper_model_path=lambda model_dir, name: name,
        get_whisper_models=lambda: [],
        on_log=lambda msg: card._on_log("INFO", msg),
        on_progress=card._pipeline_step,
        on_done=Mock(),
    )

    with patch(
        "pipeline_runner.run_preflight",
        return_value=PreflightPlan(
            reachable=False,
            workers=3,
            note=message,
            info=ServerInfo(reachable=False, error="ReadError"),
        ),
    ):
        runner.run(
            files=["/test/a.srt"],
            target_lang="zh-cn",
            language="en",
            replace_original=False,
            whisper_cli_path=Path("whisper-cli.exe"),
            whisper_model_name="tiny",
            whisper_model_dir="/models",
        )

    card._on_log.assert_called_once_with("INFO", f"Error: {message}")


def test_completed_file_moves_out_of_pending():
    card = _make_card()
    card._file_listbox_widget = Mock()
    card._file_listbox_widget.remove.return_value = True
    card._completed_files = []
    card._completed_listbox = Mock()

    card._move_pipeline_file_to_completed("D:/media/a.mp4")

    card._file_listbox_widget.remove.assert_called_once_with("D:/media/a.mp4")
    assert card._completed_files == ["D:/media/a.mp4"]
    card._completed_listbox.insert.assert_called_once_with("end", "a.mp4")


def test_clear_pipeline_files_clears_pending_and_completed():
    card = _make_card()
    card._file_listbox_widget = Mock()
    card._completed_files = ["D:/media/a.mp4"]
    card._completed_listbox = Mock()

    card._clear_pipeline_files()

    card._file_listbox_widget.clear.assert_called_once_with()
    assert card._completed_files == []
    card._completed_listbox.delete.assert_called_once_with(0, "end")


def test_resolve_whisper_model_path_uses_the_shared_registry_lookup():
    card = object.__new__(PipelineCard)
    card._get_whisper_models = lambda: [{"name": "tiny", "path": "D:/models/tiny.bin"}]

    assert card._resolve_whisper_model_path("D:/models", "tiny") == "D:/models/tiny.bin"
    assert card._resolve_whisper_model_path("D:/models", "other.bin") == str(
        Path("D:/models") / "other.bin")


class TestStartGate:
    """The llama-server requirement belongs to local mode only."""

    def _card_for_start(self, *, mode):
        card = object.__new__(PipelineCard)
        card._config = Mock()
        card._config.get.side_effect = lambda key, default=None: (
            mode if key == 'llm.mode' else default)
        card._pipeline_runner = None
        card._get_server_running = lambda: False
        card._pipe_files = []  # empty: the run stops at the next check
        return card

    def test_local_mode_requires_the_server_process(self):
        card = self._card_for_start(mode='local')
        with patch('pipeline_card.messagebox.showwarning') as warn:
            PipelineCard._start_pipeline(card)
        assert warn.call_args[0][1] == "Start the llama server first!"

    def test_remote_mode_passes_the_server_gate_without_llama_server(self):
        card = self._card_for_start(mode='remote')
        with patch('pipeline_card.messagebox.showwarning') as warn:
            PipelineCard._start_pipeline(card)
        # The gate passed; the run stopped only at the empty file list.
        assert warn.call_args[0][1] == "Select files first!"
