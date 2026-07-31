"""Headless tests for SubtitleTranslationTab consuming a PreflightPlan.

The tab is constructed with object.__new__ and its Tk vars replaced by fakes, the
same approach tests/test_server_tab.py uses: the logic under test is ordinary
Python, and standing up a real Tk widget tree would make these tests need a
display.

run_preflight itself (probe → clamp → note) is tested in
tests/translation/test_preflight.py; here it is patched, because the tab's job is
to present the plan, not to re-derive it.
"""
from unittest.mock import Mock, patch

from subtitle_tab import SubtitleTranslationTab
from translation.preflight import PreflightPlan
from translation.server_probe import ServerInfo


class FakeVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeConfig:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


def _config_for(*, workers=3, api_url='http://localhost:8080/v1'):
    """The dict _start_translation assembles on the UI thread and hands off.

    Built here rather than by calling _start_translation so these tests exercise
    the preflight in isolation, without needing a real Tk thread.
    """
    return {
        'api_url': api_url,
        'model': 'test-model',
        'max_tokens': 16384,
        'temperature': 0.2,
        'batch_size': 15,
        'max_workers': workers,
        'single_step': True,
    }


def _plan(*, reachable=True, workers=3, note=None, info=None):
    """A PreflightPlan of the shape run_preflight would produce."""
    return PreflightPlan(
        reachable=reachable,
        workers=workers,
        note=note,
        info=info or ServerInfo(reachable=reachable),
    )


def _make_tab(*, workers=3, files=('/test/a.srt',), config=None):
    tab = object.__new__(SubtitleTranslationTab)
    tab._config_manager = FakeConfig(config)
    tab._get_port = lambda: 8080
    tab._get_model = lambda: 'test-model'
    tab._file_list = list(files)
    tab._translating = False
    tab._stop_requested = False
    tab._translator = None

    tab._batch_var = FakeVar(15)
    tab._temp_var = FakeVar(0.2)
    tab._tokens_var = FakeVar(16384)
    tab._workers_var = FakeVar(workers)
    tab._fast_mode_var = FakeVar(True)
    tab._replace_var = FakeVar(False)
    tab._progress_var = FakeVar(0)

    tab._start_btn = Mock()
    tab._stop_btn = Mock()
    tab._file_listbox = Mock()
    tab._progress_label = Mock()
    tab._clear_log = Mock()
    tab._log = Mock()
    tab._run_translation = Mock()

    # after() callbacks are what the real widget would marshal onto the UI
    # thread; run them inline so the assertions see their effects.
    toplevel = Mock()
    toplevel.after = lambda delay, fn: fn()
    tab.winfo_toplevel = lambda: toplevel
    return tab


class TestPreflightBlocksRun:

    def test_unreachable_server_does_not_translate(self):
        tab = _make_tab()
        plan = _plan(reachable=False, workers=3,
                     note="llama-server 未回應",
                     info=ServerInfo(reachable=False,
                                     error="ConnectError: refused"))

        with patch('subtitle_tab.run_preflight', return_value=plan), \
             patch('subtitle_tab.messagebox.showerror') as show_error, \
             patch('subtitle_tab.LocalLLMTranslator') as translator_cls:
            tab._preflight_and_translate(_config_for(workers=tab._workers_var.get()))

        tab._run_translation.assert_not_called()
        translator_cls.assert_not_called()
        show_error.assert_called_once()

    def test_unreachable_server_releases_the_buttons(self):
        """A run that never starts must leave the tab usable.

        Start is disabled on click; without the abort path the user is left with
        a dead Start button and a Stop button that stops nothing.
        """
        tab = _make_tab()
        tab._translating = True
        plan = _plan(reachable=False, workers=3, note="down",
                     info=ServerInfo(reachable=False, error="down"))

        with patch('subtitle_tab.run_preflight', return_value=plan), \
             patch('subtitle_tab.messagebox.showerror'):
            tab._preflight_and_translate(_config_for(workers=tab._workers_var.get()))

        assert tab._translating is False
        tab._start_btn.config.assert_called_with(state="normal")
        tab._stop_btn.config.assert_called_with(state="disabled")

    def test_reachable_server_proceeds(self):
        tab = _make_tab()
        plan = _plan(reachable=True, workers=3, note=None)

        with patch('subtitle_tab.run_preflight', return_value=plan), \
             patch('subtitle_tab.LocalLLMTranslator') as translator_cls:
            tab._preflight_and_translate(_config_for(workers=tab._workers_var.get()))

        tab._run_translation.assert_called_once()
        translator_cls.assert_called_once()

    def test_asks_for_one_plan_with_the_requested_workers(self):
        """The tab calls the preflight module with the URL and requested count."""
        tab = _make_tab(workers=3)
        plan = _plan(reachable=True, workers=3, note=None)

        with patch('subtitle_tab.run_preflight',
                   return_value=plan) as preflight, \
             patch('subtitle_tab.LocalLLMTranslator'):
            config = _config_for(workers=tab._workers_var.get())
            tab._preflight_and_translate(config)

        preflight.assert_called_once_with(config['api_url'], 3)

    def test_unreachable_messagebox_shows_the_plan_note(self):
        """The plan's note is the text the user sees; the tab must not re-derive it."""
        tab = _make_tab()
        note = "llama-server 未回應 (http://localhost:8080/props): down"
        plan = _plan(reachable=False, workers=3, note=note,
                     info=ServerInfo(reachable=False, error="down"))

        with patch('subtitle_tab.run_preflight', return_value=plan), \
             patch('subtitle_tab.messagebox.showerror') as show_error, \
             patch('subtitle_tab.LocalLLMTranslator'):
            tab._preflight_and_translate(_config_for(workers=tab._workers_var.get()))

        show_error.assert_called_once()
        assert show_error.call_args[0][1] == note
        assert any(call.args[1] == note for call in tab._log.call_args_list)


class TestWorkerClamping:

    def test_workers_clamped_to_slots(self):
        """3 workers against a 1-slot server is 1 worker's throughput."""
        tab = _make_tab(workers=3)
        plan = _plan(reachable=True, workers=1, note="clamped")

        with patch('subtitle_tab.run_preflight', return_value=plan), \
             patch('subtitle_tab.LocalLLMTranslator') as translator_cls:
            tab._preflight_and_translate(_config_for(workers=tab._workers_var.get()))

        assert translator_cls.call_args[0][0]['max_workers'] == 1

    def test_clamp_is_reported_to_the_user(self):
        """Silently fixing it would leave the mismatched setting in place."""
        tab = _make_tab(workers=3)
        plan = _plan(reachable=True, workers=1,
                     note="workers 3 → server 只有 1 slot，已調整為 1")

        with patch('subtitle_tab.run_preflight', return_value=plan), \
             patch('subtitle_tab.LocalLLMTranslator'):
            tab._preflight_and_translate(_config_for(workers=tab._workers_var.get()))

        assert any(call.args[0] == "WARNING" for call in tab._log.call_args_list)

    def test_workers_kept_when_slots_allow(self):
        tab = _make_tab(workers=3)
        plan = _plan(reachable=True, workers=3, note=None)

        with patch('subtitle_tab.run_preflight', return_value=plan), \
             patch('subtitle_tab.LocalLLMTranslator') as translator_cls:
            tab._preflight_and_translate(_config_for(workers=tab._workers_var.get()))

        assert translator_cls.call_args[0][0]['max_workers'] == 3

    def test_unknown_slot_count_leaves_workers_alone(self):
        """An older build reporting no total_slots must not serialize the client."""
        tab = _make_tab(workers=3)
        plan = _plan(reachable=True, workers=3, note=None)

        with patch('subtitle_tab.run_preflight', return_value=plan), \
             patch('subtitle_tab.LocalLLMTranslator') as translator_cls:
            tab._preflight_and_translate(_config_for(workers=tab._workers_var.get()))

        assert translator_cls.call_args[0][0]['max_workers'] == 3


class TestWorkerPersistence:

    def test_saves_the_requested_count_not_the_clamped_one(self):
        """The clamp describes today's server, so it must not edit the setting.

        Persisting the clamped value would ratchet the slider down after a single
        run against a 1-slot server, and the user would never get it back by
        raising the server's slot count.
        """
        tab = _make_tab(workers=3)
        plan = _plan(reachable=True, workers=1, note="clamped")

        with patch('subtitle_tab.run_preflight', return_value=plan), \
             patch('subtitle_tab.LocalLLMTranslator'):
            config = _config_for(workers=tab._workers_var.get())
            tab._config_manager.set("ui.max_workers", config['max_workers'])
            tab._preflight_and_translate(config)

        assert tab._config_manager.get("ui.max_workers") == 3


class TestWorkerSyncFromConfig:

    def test_slider_follows_a_preset_written_by_the_server_tab(self):
        """A Server-tab preset writes ui.max_workers; the slider must catch up."""
        tab = _make_tab(workers=3, config={"ui.max_workers": 1})
        tab._workers_label = Mock()

        tab._sync_workers_from_config()

        assert tab._workers_var.get() == 1
        tab._workers_label.config.assert_called_once_with(text="1")
