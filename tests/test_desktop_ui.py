"""Exercise actual Tk layouts and log timers without starting external models."""
import tkinter as tk
from unittest.mock import patch

import pytest
from tkinterdnd2 import TkinterDnD

from llama_manager import LlamaManager
from ui_helpers import LOG_DRAIN_MAX, log_bus


@pytest.fixture(scope="module")
def desktop():
    try:
        root = TkinterDnD.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display/runtime unavailable: {exc}")
    root.withdraw()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    with patch("llama_manager.ConfigManager.load"), \
         patch("llama_manager.ConfigManager.save"), \
         patch.object(LlamaManager, "scan_models"):
        app = LlamaManager(root)
        root.update_idletasks()
        try:
            yield app
        finally:
            app._close_debug_win()
            root.destroy()
    assert not errors


def test_tabs_fit_minimum_window_width(desktop):
    widths = {name: getattr(desktop, name).winfo_reqwidth() for name in
              ("main_tab", "server_tab", "subtitle_tab", "whisper_tab")}
    assert max(widths.values()) <= desktop.root.minsize()[0], widths


def test_debug_batches_logs_and_releases_timer_on_close(desktop):
    desktop._open_debug_win()
    desktop._debug_win.withdraw()
    desktop.root.after_cancel(desktop._debug_after)
    for i in range(1000):
        log_bus.emit("INFO", f"message {i}", "pipeline")
    desktop._poll_debug_queue()
    assert desktop._debug_queue.qsize() == 1000 - LOG_DRAIN_MAX
    assert "[pipeline] message 199" in desktop._debug_text.get("1.0", tk.END)
    assert desktop._debug_text.cget("state") == "disabled"
    timer = desktop._debug_after
    desktop._close_debug_win()
    assert timer not in desktop.root.tk.call("after", "info")


def test_destroyed_log_widget_unsubscribes(desktop):
    tab = desktop.whisper_tab
    before = len(log_bus._subscribers)
    tab.destroy()
    assert len(log_bus._subscribers) == before - 1
