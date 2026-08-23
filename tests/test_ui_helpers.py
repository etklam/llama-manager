"""Tests for ui_helpers module: populate_language_combo, extract_combo_code, LogMixin."""
import queue
import tkinter as tk
from tkinter import ttk
from unittest.mock import MagicMock, patch

import pytest

from ui_helpers import (
    LOG_POLL_MS,
    populate_language_combo,
    extract_combo_code,
    LogMixin,
)


# ---------------------------------------------------------------------------
# populate_language_combo (uses mock combo to avoid Tkinter display issues)
# ---------------------------------------------------------------------------

class TestPopulateLanguageCombo:

    def _make_mock_combo(self):
        """Create a mock combo that simulates ttk.Combobox behavior."""
        combo = MagicMock()
        combo._values = []
        combo._current = None

        def set_values(val):
            combo._values = list(val)

        def get_current():
            return combo._current

        def set_current(idx):
            combo._current = idx

        combo.__setitem__ = lambda self, key, val: set_values(val) if key == 'values' else None
        combo.__getitem__ = lambda self, key: tuple(combo._values) if key == 'values' else None
        combo.current = MagicMock(side_effect=set_current)
        combo.current.return_value = None
        # Make current() return the stored value when called with no args
        combo.current = MagicMock(side_effect=lambda idx=None: (
            set_current(idx) if idx is not None else get_current()
        ))
        return combo

    def test_empty_languages(self):
        combo = self._make_mock_combo()
        languages = {}
        result = populate_language_combo(combo, languages)
        assert result['values'] == ()
        # No items, so current() should not be called
        combo.current.assert_not_called()

    def test_populates_values_correctly(self):
        combo = self._make_mock_combo()
        languages = {"en": "English", "zh": "Chinese"}
        populate_language_combo(combo, languages)
        values = list(combo['values'])
        assert "en - English" in values
        assert "zh - Chinese" in values

    def test_no_saved_value_selects_first(self):
        combo = self._make_mock_combo()
        languages = {"en": "English", "zh": "Chinese", "ja": "Japanese"}
        populate_language_combo(combo, languages)
        # Should call current(0) to select the first item
        combo.current.assert_called_with(0)

    def test_saved_value_selects_matching_entry(self):
        combo = self._make_mock_combo()
        languages = {"en": "English", "zh": "Chinese", "ja": "Japanese"}
        populate_language_combo(combo, languages, saved_value="zh")
        # Should call current(1) for zh (index 1)
        combo.current.assert_called_with(1)

    def test_saved_value_not_found_selects_first(self):
        combo = self._make_mock_combo()
        languages = {"en": "English", "zh": "Chinese"}
        populate_language_combo(combo, languages, saved_value="xx")
        # The else branch of the for loop fires, calling current(0)
        combo.current.assert_called_with(0)

    def test_saved_value_empty_string_selects_first(self):
        combo = self._make_mock_combo()
        languages = {"en": "English", "zh": "Chinese"}
        # empty string is falsy, so it should fall through to the elif
        populate_language_combo(combo, languages, saved_value="")
        combo.current.assert_called_with(0)

    def test_returns_combo_object(self):
        combo = self._make_mock_combo()
        languages = {"en": "English"}
        result = populate_language_combo(combo, languages)
        assert result is combo

    def test_single_language(self):
        combo = self._make_mock_combo()
        languages = {"en": "English"}
        populate_language_combo(combo, languages)
        values = list(combo['values'])
        assert len(values) == 1
        assert values[0] == "en - English"
        combo.current.assert_called_with(0)

    def test_preserves_insertion_order(self):
        """Languages dict in Python 3.7+ preserves insertion order."""
        combo = self._make_mock_combo()
        languages = {"ja": "Japanese", "en": "English", "zh": "Chinese"}
        populate_language_combo(combo, languages)
        values = list(combo['values'])
        assert values[0] == "ja - Japanese"
        assert values[1] == "en - English"
        assert values[2] == "zh - Chinese"


# ---------------------------------------------------------------------------
# extract_combo_code (pure logic, no Tkinter needed)
# ---------------------------------------------------------------------------

class TestExtractComboCode:

    def test_standard_format(self):
        assert extract_combo_code("en - English") == "en"

    def test_code_with_dash_in_name(self):
        assert extract_combo_code("zh-cn - Chinese (Simplified)") == "zh-cn"

    def test_no_separator(self):
        assert extract_combo_code("en") == "en"

    def test_empty_string(self):
        assert extract_combo_code("") == ""

    def test_separator_only(self):
        # " - " is in the string, so it splits and returns ""
        assert extract_combo_code(" - ") == ""

    def test_multiple_separators(self):
        # split on first " - " only
        result = extract_combo_code("en - English - US")
        assert result == "en"


# ---------------------------------------------------------------------------
# LogMixin._insert_log auto-cleanup
# We test the trimming logic without a real Tkinter widget by mocking the
# ScrolledText widget.
# ---------------------------------------------------------------------------

class TestLogMixinTrimLogic:
    """Test the line-trimming logic in _insert_log using mocks."""

    def test_insert_log_calls_insert_with_formatted_line(self):
        """_insert_log should insert a formatted line with timestamp and level."""
        mixin = LogMixin()
        mock_text = MagicMock()
        # Simulate index returning "2.0" (meaning 1 line after insert)
        mock_text.index.return_value = "2.0"
        mixin._log_text = mock_text
        mixin._log_max_lines = 2000

        mixin._insert_log("INFO", "test message")

        mock_text.config.assert_any_call(state="normal")
        mock_text.insert.assert_called_once()
        inserted_line = mock_text.insert.call_args[0][1]
        assert "[INFO]" in inserted_line
        assert "test message" in inserted_line
        mock_text.config.assert_any_call(state="disabled")

    def test_delete_called_when_exceeding_max_lines(self):
        """When line count > max_lines, delete should be called to trim."""
        mixin = LogMixin()
        mock_text = MagicMock()
        # Simulate 12 lines after insert
        mock_text.index.return_value = "12.0"
        mixin._log_text = mock_text
        mixin._log_max_lines = 10

        mixin._insert_log("INFO", "overflow")

        # Should delete lines 1 through 2 (12 - 10 = 2)
        mock_text.delete.assert_called_once_with(1.0, "2.0")

    def test_delete_not_called_when_within_max_lines(self):
        """When line count <= max_lines, delete should NOT be called."""
        mixin = LogMixin()
        mock_text = MagicMock()
        mock_text.index.return_value = "5.0"
        mixin._log_text = mock_text
        mixin._log_max_lines = 10

        mixin._insert_log("INFO", "within limit")

        mock_text.delete.assert_not_called()

    def test_delete_not_called_at_exact_max_lines(self):
        """When line count == max_lines, no trimming needed."""
        mixin = LogMixin()
        mock_text = MagicMock()
        mock_text.index.return_value = "10.0"
        mixin._log_text = mock_text
        mixin._log_max_lines = 10

        mixin._insert_log("INFO", "at limit")

        mock_text.delete.assert_not_called()

    def test_see_end_called(self):
        """_insert_log should call see(tk.END) to scroll to bottom."""
        mixin = LogMixin()
        mock_text = MagicMock()
        mock_text.index.return_value = "1.0"
        mixin._log_text = mock_text
        mixin._log_max_lines = 2000

        mixin._insert_log("INFO", "scroll test")

        mock_text.see.assert_called_once()

    def test_large_overflow_trims_correctly(self):
        """When many lines over max, trims the right number."""
        mixin = LogMixin()
        mock_text = MagicMock()
        mock_text.index.return_value = "2050.0"
        mixin._log_text = mock_text
        mixin._log_max_lines = 2000

        mixin._insert_log("INFO", "big overflow")

        # Should delete lines 1 through 50 (2050 - 2000 = 50)
        mock_text.delete.assert_called_once_with(1.0, "50.0")


class TestLogMixinDefaultMaxLines:
    """Test that the class-level default is 2000."""

    def test_class_level_default(self):
        assert LogMixin._log_max_lines == 2000


# ---------------------------------------------------------------------------
# LogMixin queue pump. Emitters include background threads (the server stdout
# monitor, workers), so the bus handler must not touch Tk directly.
# ---------------------------------------------------------------------------

class TestLogQueuePump:
    def _make_mixin(self):
        mixin = LogMixin()
        mixin._log_text = MagicMock()
        mixin._log_text.index.return_value = "1.0"
        mixin._log_max_lines = 2000
        mixin._log_queue = queue.Queue()
        return mixin

    def test_handler_enqueues_without_touching_widget(self):
        mixin = self._make_mixin()

        mixin._log_bus_handler("INFO", "from a worker thread")

        assert mixin._log_queue.qsize() == 1
        mixin._log_text.insert.assert_not_called()

    def test_poll_drains_queue_and_reschedules(self):
        mixin = self._make_mixin()
        mixin._log_bus_handler("INFO", "a")
        mixin._log_bus_handler("ERROR", "b")

        mixin._poll_log_queue()

        assert mixin._log_queue.qsize() == 0
        assert mixin._log_text.insert.call_count == 2
        mixin._log_text.after.assert_called_once_with(
            LOG_POLL_MS, mixin._poll_log_queue)

    def test_poll_batches_many_lines_in_one_widget_pass(self):
        mixin = self._make_mixin()
        mixin._log_text.index.return_value = "1.0"
        for i in range(50):
            mixin._log_bus_handler("INFO", f"line {i}")

        mixin._poll_log_queue()

        # One enable/see/disable pass for the whole batch, not per line.
        assert mixin._log_text.insert.call_count == 50
        assert mixin._log_text.see.call_count == 1

    def test_poll_survives_destroyed_widget(self):
        mixin = self._make_mixin()
        mixin._log_text.after.side_effect = tk.TclError("bad window path")
        mixin._log_bus_handler("INFO", "too late")

        mixin._poll_log_queue()  # must not raise
