"""Small native-Tk tests for the manga tab's UI/worker boundary."""

import copy
import tkinter as tk
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from config_manager import ConfigManager, DEFAULT_CONFIG
from manga_tab import MangaTab
from manga.runner import RunSnapshot


@pytest.fixture(scope="module")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display/runtime unavailable: {exc}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def tab(tk_root, tmp_path):
    config = ConfigManager(str(tmp_path / "config.json"))
    # Keep the fixture's config entirely in memory; the tab may still call set().
    config._config = copy.deepcopy(DEFAULT_CONFIG)
    config._save_to_file = Mock()
    instance = MangaTab(tk_root, Mock(), config)
    tk_root.update_idletasks()
    yield instance
    instance.destroy()


def test_pages_can_be_added_reordered_and_removed_without_worker(tab, tmp_path):
    first = tmp_path / "一.png"
    second = tmp_path / "二.png"
    first.write_bytes(b"image")
    second.write_bytes(b"image")
    tab.files = [str(first), str(second)]
    tab.refresh_pages()
    assert tab.pages.size() == 2
    tab.pages.selection_set(1)
    tab.move(-1)
    assert tab.files == [str(second), str(first)]
    tab.pages.selection_clear(0, "end")
    tab.pages.selection_set(0)
    tab.remove()
    assert tab.files == [str(first)]


def test_start_builds_immutable_snapshot_and_worker_does_not_read_tk_variables(tab, tmp_path):
    tab.files = [str(tmp_path / "page.png")]
    tab.vars["output"].set(str(tmp_path / "out"))
    tab.vars["executable"].set(str(tmp_path / "llama-server"))
    tab.vars["device"].set("Vulkan0")
    tab.vars["ocr_model"].set(str(tmp_path / "ocr.gguf"))
    tab.vars["mmproj"].set(str(tmp_path / "mmproj.gguf"))
    tab.vars["translation_model"].set(str(tmp_path / "translate.gguf"))
    runner = Mock()
    runner.run = Mock()
    threads = []

    class ImmediateThread:
        def __init__(self, **kwargs):
            threads.append(kwargs)

        def start(self):
            pass

    with patch("manga_tab.MangaRunner", return_value=runner) as runner_cls, \
         patch("manga_tab.threading.Thread", ImmediateThread):
        tab.start()
    snapshot = runner_cls.call_args.args[1]
    assert isinstance(snapshot, RunSnapshot)
    assert snapshot.files == (str(tmp_path / "page.png"),)
    assert snapshot.device == "Vulkan0"
    assert snapshot.run_id == tab.run_id
    tab.vars["device"].set("Vulkan9")
    tab.files.append(str(tmp_path / "later.png"))
    assert snapshot.device == "Vulkan0"
    assert snapshot.files == (str(tmp_path / "page.png"),)
    assert threads and threads[0]["target"] == runner.run


def test_late_done_event_is_ignored_but_current_done_releases_active(tab):
    tab.run_id = 2
    tab.active = True
    tab.events.put({"run_id": 1, "stage": "done", "report": {"status": "completed", "failures": 0, "stages": [], "restore_error": None}})
    tab.poll()
    assert tab.active is True
    tab.events.put({"run_id": 2, "stage": "done", "report": {"status": "completed", "failures": 0, "stages": [], "restore_error": None}})
    tab.poll()
    assert tab.active is False


def test_cancel_requests_stopping_and_correction_is_blocked_while_active(tab):
    runner = Mock()
    tab.runner, tab.active = runner, True
    tab.cancel()
    runner.cancel.assert_called_once_with()
    assert "Stopping" in tab.status.get()
    tab.page_location = (Path("unused"), "a" * 24)
    with patch("manga_tab.correct_page") as correct:
        tab.save_correction()
    correct.assert_not_called()


def test_device_discovery_runs_off_ui_callback_and_queues_combobox_update(tab):
    tab.vars["executable"].set("runtime")

    class ImmediateThread:
        def __init__(self, target, daemon=True):
            self.target = target

        def start(self):
            self.target()

    info = {"devices": ["Vulkan0", "Vulkan1"], "version": "llama.cpp", "device_list": "Vulkan0 Vulkan1"}
    with patch("manga_tab.inspect_runtime", return_value=info), \
         patch("manga_tab.threading.Thread", ImmediateThread):
        tab.discover()
    tab.poll()
    assert tuple(tab.device_combo.cget("values")) == ("Vulkan0", "Vulkan1")
    assert tab.vars["device"].get() == "Vulkan0"
    assert "unverified" in tab.status.get()

def test_job_reload_restores_order_profiles_and_mode(tab, tmp_path):
    from dataclasses import asdict
    from manga.runtime import Profile
    from manga.store import write_json
    snapshot = RunSnapshot(('page10.png', 'page2.png'), str(tmp_path / 'output'),
        'runtime', 'Vulkan1', Profile('ocr', 'mmproj'), Profile('translation'),
        mode='retranslate', independent=True, two_step=True)
    path = tmp_path / 'manga-job.json'
    write_json(path, asdict(snapshot))
    with patch('manga_tab.filedialog.askopenfilename', return_value=str(path)):
        tab.load_job()
    assert tab.files == ['page10.png', 'page2.png']
    assert tab.mode.get() == 'retranslate'
    assert tab.vars['mmproj'].get() == 'mmproj'
    assert tab.vars['device'].get() == 'Vulkan1'
    assert tab.group.get() == '每張獨立'
    assert tab.two_step.get()


def test_manga_controls_fit_existing_minimum_window_width(tab):
    tab.update_idletasks()
    assert tab.winfo_reqwidth() <= 960
