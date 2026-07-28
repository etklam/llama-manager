"""Headless tests for PipelineCard log routing."""
from unittest.mock import Mock

from pipeline_card import PipelineCard


def _make_card():
    card = object.__new__(PipelineCard)
    card._on_log = Mock()
    card._on_debug_log = Mock()
    card._pipe_status_label = Mock()

    toplevel = Mock()
    toplevel.after = lambda delay, fn: fn()
    card.winfo_toplevel = lambda: toplevel
    return card


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
