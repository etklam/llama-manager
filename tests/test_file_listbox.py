"""Tests for FileListboxModel: pure-Python state behind the file-list widget.

The model owns: a list of files, add-only-if-supported-and-not-duplicate,
clear, drop-data parsing (via existing parse_dropped_paths), and an
on_change callback that fires on add/clear but NOT on no-ops.

No tkinter is imported here — model is testable without a Tk root.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import pytest

from constants import SUPPORTED_SUBTITLE, SUPPORTED_MEDIA
from file_listbox import FileListboxModel


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_supported_subtitle(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE)
        assert m.add("/tmp/a.srt") is True
        assert m.files == ["/tmp/a.srt"]

    def test_add_rejects_unsupported(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE)
        assert m.add("/tmp/a.mp4") is False
        assert m.files == []

    def test_add_rejects_duplicate(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE)
        m.add("/tmp/a.srt")
        assert m.add("/tmp/a.srt") is False
        assert m.files == ["/tmp/a.srt"]

    def test_add_case_insensitive_ext(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE)
        assert m.add("/tmp/A.SRT") is True

    def test_add_media_extensions(self):
        m = FileListboxModel(SUPPORTED_MEDIA)
        assert m.add("/x/a.wav") is True
        assert m.add("/x/a.mp4") is True
        assert m.add("/x/a.srt") is False

    def test_add_many_only_appends_new(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE)
        m.add("/tmp/a.srt")
        added = m.add_many(["/tmp/a.srt", "/tmp/b.srt", "/tmp/c.txt"])
        assert added == 2
        assert m.files == ["/tmp/a.srt", "/tmp/b.srt", "/tmp/c.txt"]


# ---------------------------------------------------------------------------
# remove()
# ---------------------------------------------------------------------------

class TestRemove:
    def test_remove_existing_file(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE)
        m.add("/tmp/a.srt")

        assert m.remove("/tmp/a.srt") is True
        assert m.files == []

    def test_remove_missing_file_is_noop(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE)

        assert m.remove("/tmp/a.srt") is False
        assert m.files == []


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_empties_list(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE)
        m.add("/tmp/a.srt")
        m.clear()
        assert m.files == []

    def test_clear_when_already_empty(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE)
        m.clear()
        assert m.files == []


# ---------------------------------------------------------------------------
# add_dropped() — uses existing parse_dropped_paths
# ---------------------------------------------------------------------------

class TestAddDropped:
    def test_braced_paths(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE)
        added = m.add_dropped("{C:/a.srt} {C:/b.srt}")
        assert added == 2
        assert m.files == ["C:/a.srt", "C:/b.srt"]

    def test_unbraced_paths(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE)
        added = m.add_dropped("/tmp/a.srt /tmp/b.txt")
        assert added == 2

    def test_unsupported_in_drop_is_silently_skipped(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE)
        added = m.add_dropped("{C:/a.mp4} {C:/b.srt}")
        assert added == 1
        assert m.files == ["C:/b.srt"]

    def test_duplicates_in_drop_skipped(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE)
        m.add("/tmp/a.srt")
        added = m.add_dropped("/tmp/a.srt /tmp/b.srt")
        assert added == 1
        assert m.files == ["/tmp/a.srt", "/tmp/b.srt"]

    def test_empty_drop_returns_zero(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE)
        assert m.add_dropped("") == 0
        assert m.files == []


# ---------------------------------------------------------------------------
# on_change callback
# ---------------------------------------------------------------------------

class TestOnChange:
    def test_fires_on_add(self):
        events = []
        m = FileListboxModel(SUPPORTED_SUBTITLE, on_change=lambda: events.append(1))
        m.add("/tmp/a.srt")
        assert len(events) == 1

    def test_fires_on_clear(self):
        events = []
        m = FileListboxModel(SUPPORTED_SUBTITLE, on_change=lambda: events.append(1))
        m.add("/tmp/a.srt")
        m.clear()
        assert len(events) == 2

    def test_does_not_fire_on_noop_add(self):
        events = []
        m = FileListboxModel(SUPPORTED_SUBTITLE, on_change=lambda: events.append(1))
        m.add("/tmp/a.mp4")  # unsupported
        m.add("/tmp/a.srt")
        m.add("/tmp/a.srt")  # duplicate
        assert len(events) == 1

    def test_does_not_fire_on_clear_when_empty(self):
        events = []
        m = FileListboxModel(SUPPORTED_SUBTITLE, on_change=lambda: events.append(1))
        m.clear()  # already empty
        assert len(events) == 0

    def test_optional_callback_none_is_fine(self):
        m = FileListboxModel(SUPPORTED_SUBTITLE, on_change=None)
        m.add("/tmp/a.srt")
        m.clear()
        assert m.files == []
