"""Run state freeze: a run keeps the settings it started with.

A tab switch used to replace an active run's translator (via
refresh_model on notebook entry), and worker threads re-read Tk
variables per file, so mid-run UI edits re-targeted the remaining files.
These tests replay exactly that: a two-file run with a tab switch and UI
edits between files.
"""
from unittest.mock import Mock, patch

from pipeline_runner import PipelineRunner
from subtitle_tab import RunSnapshot, SubtitleTranslationTab

from tests.test_subtitle_tab_preflight import (
    FakeVar,
    _config_for,
    _make_tab,
)


class TestSubtitleTabRunFreeze:

    def test_two_file_run_survives_tab_switch_and_ui_edits(self):
        """Both files must retain the original plan, target and settings."""
        tab = _make_tab(files=('/test/a.srt', '/test/b.srt'))
        tab._run_seq = 7
        tab._translating = True  # the run is active
        translator = Mock()

        seen = []

        def translate_file(translator_arg, path, target_lang,
                           replace_original=False):
            seen.append((translator_arg, path, target_lang, replace_original))
            if len(seen) == 1:
                # Between files: the user enters the subtitle tab (notebook
                # change fires refresh_model) and edits the UI. Everything
                # they touch must configure the *next* run only.
                tab.refresh_model()
                tab._target_var = FakeVar('en')        # UI edit...
                tab._replace_var = FakeVar(True)       # ...mid-run
                tab._workers_var = FakeVar(8)
                tab._batch_var = FakeVar(1)

        tab._translate_file = translate_file
        tab._init_translator = Mock(
            side_effect=AssertionError('refresh must not reconfigure a run'))
        tab._update_model_display = Mock()
        tab._sync_workers_from_config = Mock()
        tab._refresh_llm_profile_combo = Mock()

        snapshot = RunSnapshot(
            files=('/test/a.srt', '/test/b.srt'),
            target_lang='zh-tw',
            replace_original=False,
            config=_config_for(),
            run_id=7,
        )
        SubtitleTranslationTab._run_translation(tab, translator, snapshot)

        assert len(seen) == 2
        for entry in seen:
            assert entry[0] is translator       # same run translator
            assert entry[2] == 'zh-tw'          # snapshot target, not Tk var
            assert entry[3] is False            # snapshot replace flag
        tab._init_translator.assert_not_called()

    def test_worker_thread_never_reads_tk_variables(self):
        """The run loop takes target/replace from the snapshot only."""
        source = open('subtitle_tab.py', encoding='utf-8').read()
        run_translation = source[source.index('def _run_translation'):]
        run_translation = run_translation[:run_translation.index('def _on_translation')]
        assert '_target_var' not in run_translation
        assert '_replace_var' not in run_translation
        assert '_get_target_code' not in run_translation

    def test_late_done_callback_cannot_mutate_a_newer_run(self):
        tab = _make_tab()
        tab._run_seq = 2  # a newer run owns the UI
        tab._translating = True
        tab._failed_files = 0
        SubtitleTranslationTab._on_translation_done(tab, run_id=1)
        tab._start_btn.config.assert_not_called()
        assert tab._translating is True  # untouched by the stale callback

        # The current run's callback still works.
        SubtitleTranslationTab._on_translation_done(tab, run_id=2)
        tab._start_btn.config.assert_called_with(state="normal")

    def test_late_abort_callback_cannot_release_a_newer_run(self):
        tab = _make_tab()
        tab._run_seq = 2
        tab._translating = True
        SubtitleTranslationTab._on_translation_aborted(tab, run_id=1)
        assert tab._translating is True
        tab._start_btn.config.assert_not_called()

    def test_preflight_applies_discovered_capacity_to_the_run_copy_only(self):
        """Discovered workers/context go to a copy; the snapshot stays frozen."""
        from translation.preflight import PreflightPlan
        from translation.server_probe import ServerInfo

        tab = _make_tab()
        plan = PreflightPlan(reachable=True, workers=1, note='clamped',
                             info=ServerInfo(reachable=True, slots=1,
                                             context_size=4096))
        config = _config_for(workers=3)
        snapshot = RunSnapshot(
            files=('/test/a.srt',), target_lang='zh-tw',
            replace_original=False, config=config, run_id=1)

        with patch('subtitle_tab.plan_for_target', return_value=plan), \
             patch('subtitle_tab.LocalLLMTranslator') as translator_cls:
            SubtitleTranslationTab._preflight_and_translate(tab, snapshot)

        run_config = translator_cls.call_args[0][0]
        assert run_config['max_workers'] == 1
        assert run_config['context_size'] == 4096
        # The snapshot's config is untouched: a re-run asks for 3 again.
        assert config['max_workers'] == 3
        assert config.get('context_size') is None


class TestPipelineSnapshotFreeze:

    def _make_runner(self, tmp_path):
        from unittest.mock import MagicMock
        from config_manager import ConfigManager
        callbacks = {name: Mock() for name in
                     ('on_log', 'on_progress', 'on_done', 'on_file_completed')}
        runner = PipelineRunner(
            config_manager=ConfigManager(str(tmp_path / 'config.json')),
            get_port=lambda: 8080,
            get_current_model=lambda: 'test-model',
            resolve_whisper_model_path=lambda d, n: n,
            get_whisper_models=lambda: [{'name': 'tiny', 'path': '/m'}],
            on_log=callbacks['on_log'],
            on_progress=callbacks['on_progress'],
            on_done=callbacks['on_done'],
            on_file_completed=callbacks['on_file_completed'],
        )
        return runner, callbacks

    def _run_two_files(self, tmp_path, mutate):
        from pathlib import Path
        from translation.preflight import PreflightPlan
        from translation.server_probe import ServerInfo

        runner, callbacks = self._make_runner(tmp_path)
        reachable = PreflightPlan(reachable=True, workers=4,
                                  info=ServerInfo(reachable=True, slots=4))

        for name in ('a.srt', 'b.srt'):
            (tmp_path / name).write_text(
                '1\n00:00:00,000 --> 00:00:01,000\nHello\n', encoding='utf-8')
        files = [str(tmp_path / 'a.srt'), str(tmp_path / 'b.srt')]

        with patch('pipeline_runner.LocalLLMTranslator') as translator_cls, \
             patch('pipeline_runner.run_preflight', return_value=reachable):
            mock_translator = Mock()
            mock_translator.translate_srt.return_value = []
            translator_cls.return_value = mock_translator

            def on_file_completed(filepath):
                if mutate:
                    mutate()
            callbacks['on_file_completed'].side_effect = on_file_completed

            runner.run(
                files=files, target_lang='zh-cn', language='en',
                replace_original=False,
                whisper_cli_path=Path('whisper-cli.exe'),
                whisper_model_name='tiny', whisper_model_dir='/models',
            )
            return translator_cls, mock_translator

    def test_ui_edits_between_files_do_not_re_target_the_run(self, tmp_path):
        runner, _callbacks = self._make_runner(tmp_path)

        def mutate():
            # Between files the user saves different UI settings.
            runner._config_manager.set('ui.temperature', 1.9)
            runner._config_manager.set('ui.batch_size', 1)

        translator_cls, mock_translator = self._run_two_files(tmp_path, mutate)

        # One translator, built once from the run's frozen snapshot.
        assert translator_cls.call_count == 1
        built_config = translator_cls.call_args[0][0]
        assert built_config['temperature'] != 1.9
        assert mock_translator.translate_srt.call_count == 2
