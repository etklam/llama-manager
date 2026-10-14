"""Regression coverage for shared run status, Manga outcomes and contrast."""

import copy
from dataclasses import replace
import tkinter as tk
from tkinter import ttk
from unittest.mock import Mock

import pytest

import app_shell
import manga_tab
import ui_helpers
from config_manager import ConfigManager, DEFAULT_CONFIG
from ui_helpers import CHANNEL_MANGA, CHANNEL_TRANSLATE, CHANNEL_WHISPER
from ui_theme import apply_theme
from ui_tokens import PALETTES, audit_palette, contrast_ratio


@pytest.fixture
def buses(monkeypatch):
    runs = ui_helpers._RunBus()
    logs = ui_helpers._LogBus()
    monkeypatch.setattr(ui_helpers, "run_bus", runs)
    monkeypatch.setattr(ui_helpers, "log_bus", logs)
    monkeypatch.setattr(app_shell, "run_bus", runs)
    monkeypatch.setattr(app_shell, "log_bus", logs)
    monkeypatch.setattr(manga_tab, "run_bus", runs)
    return runs, logs


@pytest.fixture(scope="module")
def tk_root():
    try:
        window = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk display/runtime unavailable: {error}")
    window.withdraw()
    errors = []
    window.report_callback_exception = lambda *args: errors.append(args)
    apply_theme(window)
    try:
        yield window
    finally:
        window.destroy()
    assert not errors


@pytest.fixture
def root(tk_root, buses):
    yield tk_root
    for child in tk_root.winfo_children():
        child.destroy()


@pytest.fixture
def shell(root):
    instance = app_shell.AppShell(root)
    for key, channel in (("whisper", CHANNEL_WHISPER),
                         ("translate", CHANNEL_TRANSLATE)):
        instance.register(key, ttk.Frame(instance.content), key,
                          channels=(channel,))
    return instance


@pytest.fixture
def manga(shell):
    config = ConfigManager()
    config._config = copy.deepcopy(DEFAULT_CONFIG)
    config._save_to_file = Mock()
    instance = manga_tab.MangaTab(shell.content, Mock(), config)
    shell.register("manga", instance, "Manga", channels=(CHANNEL_MANGA,))
    return instance


@pytest.mark.parametrize("terminal", ["completed", "aborted"])
def test_other_job_finishing_keeps_active_job_in_header(shell, buses, terminal):
    runs, _logs = buses
    runs.progress(CHANNEL_WHISPER, "Transcribing", 42, "episode.mkv")
    runs.begin(CHANNEL_TRANSLATE, "Translating")
    if terminal == "completed":
        runs.end(CHANNEL_TRANSLATE, "Translation complete")
    else:
        runs.clear(CHANNEL_TRANSLATE)

    assert shell.header._run_label.cget("text") == "Transcribing"
    assert shell.header._run_detail.cget("text") == "episode.mkv"
    assert shell.header._progress_var.get() == 42
    assert shell.header._spinner._job is not None
    assert shell.nav._rows["whisper"][1].cget("text") == "●"

    runs.end(CHANNEL_WHISPER, "Transcription complete")
    assert shell.header._run_label.cget("text") == "Transcription complete"
    assert shell.header._spinner._job is None


@pytest.mark.parametrize("event", [
    {"stage": "loading", "model": "ocr"},
    {"stage": "loading", "model": "translation"},
    {"stage": "review_required", "count": 2},
    {"stage": "restore_failed", "error": "server unavailable"},
])
def test_manga_lifecycle_events_keep_job_active(manga, shell, buses, event):
    runs, _logs = buses
    manga._set_running(True)
    runs.begin(CHANNEL_MANGA, "Manga")
    manga._handle_event(event)

    assert manga.active
    assert runs.states()[CHANNEL_MANGA].active
    assert shell.header._spinner._job is not None
    assert "disabled" in manga._start_button.state()


@pytest.mark.parametrize("report, expected_state, expected_level", [
    ({"status": "completed"}, "ok", "SUCCESS"),
    ({"status": "ocr_finished"}, "ok", "SUCCESS"),
    ({"status": "awaiting_review"}, "warn", "WARNING"),
    ({"status": "partial"}, "warn", "WARNING"),
    ({"status": "cancelled"}, "warn", "WARNING"),
    ({"status": "cancelled", "error": "cancelled", "failures": 1},
     "warn", "WARNING"),
    ({"status": "failed"}, "error", "ERROR"),
    ({"status": "ocr_finished", "counts": {"needs_review": 1}},
     "warn", "WARNING"),
    ({"status": "completed", "failures": 1}, "error", "ERROR"),
    ({"status": "completed", "restore_error": "restart failed"},
     "error", "ERROR"),
    ({"status": "cancelled", "restore_error": "restart failed"},
     "error", "ERROR"),
])
def test_manga_done_reports_actual_outcome(
        manga, shell, buses, report, expected_state, expected_level):
    runs, logs = buses
    emitted = []
    logs.subscribe(lambda level, message: emitted.append((level, message)),
                   channels=(CHANNEL_MANGA,))
    manga._set_running(True)
    manga.progress.configure(maximum=10, value=3)
    report = {"failures": 0, "stages": [], "restore_error": None, **report}
    manga._handle_event({"stage": "done", "report": report})

    assert not manga.active
    assert runs.states()[CHANNEL_MANGA].state == expected_state
    assert emitted[-1][0] == expected_level
    assert manga.progress.cget("style") == {
        "ok": "Ok.Horizontal.TProgressbar",
        "warn": "Warn.Horizontal.TProgressbar",
        "error": "Danger.Horizontal.TProgressbar",
    }[expected_state]
    assert float(manga.progress.cget("value")) == (
        10 if expected_state == "ok" else 3)
    assert shell.header._spinner._job is None


def test_new_manga_run_resets_previous_progress_outcome(manga):
    manga.progress.configure(style="Danger.Horizontal.TProgressbar", value=75)
    manga._set_running(True)
    assert manga.progress.cget("style") == "Accent.Horizontal.TProgressbar"
    assert float(manga.progress.cget("value")) == 0


def test_manga_languages_have_distinct_grid_cells(manga):
    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)

    cells = []
    for key in ("source", "locale"):
        entry = next(widget for widget in descendants(manga)
                     if widget.winfo_class() == "TEntry"
                     and str(widget.cget("textvariable")) == str(manga.vars[key]))
        info = entry.master.grid_info()
        cells.append((str(info["in"]), info["row"], info["column"]))
    assert cells[0] != cells[1]


@pytest.mark.parametrize("palette", PALETTES.values(), ids=PALETTES)
@pytest.mark.parametrize("foreground, backgrounds, minimum", [
    ("ink_faint", ("surface", "canvas"), 4.5),
    ("border", ("surface", "raised", "canvas"), 3.0),
    ("border_strong", ("surface", "raised", "overlay", "canvas"), 3.0),
])
def test_palette_meets_documented_contrast(palette, foreground, backgrounds, minimum):
    for background in backgrounds:
        assert contrast_ratio(getattr(palette, foreground),
                              getattr(palette, background)) >= minimum


@pytest.mark.parametrize("palette", PALETTES.values(), ids=PALETTES)
@pytest.mark.parametrize("foreground, background, minimum", [
    ("ink_faint", "surface", 4.5),
    ("ink_faint", "canvas", 4.5),
    ("border", "surface", 3.0),
    ("border", "raised", 3.0),
    ("border_strong", "surface", 3.0),
    ("border_strong", "raised", 3.0),
    ("border_strong", "overlay", 3.0),
])
def test_audit_rejects_contrast_regressions(palette, foreground, background, minimum):
    broken = replace(palette, **{foreground: getattr(palette, background)})
    result = next(pair for pair in audit_palette(broken)
                  if pair[:2] == (foreground, background))
    assert result[3] >= minimum
    assert not result[4]


@pytest.mark.parametrize("palette", PALETTES.values(), ids=PALETTES)
def test_complete_palette_audit_passes(palette):
    assert all(pair[4] for pair in audit_palette(palette))
