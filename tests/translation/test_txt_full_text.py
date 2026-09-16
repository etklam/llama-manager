"""Full-text (TXT) translation: budgets, completeness, and both entry points.

Full-text translation is one request with its own rules — an upfront
context-window budget and finish-reason completeness — deliberately
separate from short-subtitle recovery. These tests cover the translator
method and the Subtitle tab's use of it; the Pipeline entry point is
covered in tests/test_pipeline_runner.py.
"""
from unittest.mock import Mock

import pytest

from config_helpers import build_translation_config
from llm_target import LLMTarget, api_url_for_port
from translation.llm_client import CompletionResult
from translation.local_llm_translator import (
    LocalLLMTranslator,
    TranslationIncompleteError,
    TxtTooLargeError,
)
from utils.srt_parser import Cue, output_path_for


class NoneConfig:
    def get(self, key, default=None):
        return default


class MetadataClient:
    """A client whose responses carry finish reasons."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def complete_with_metadata(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def complete(self, **kwargs):
        raise AssertionError('full-text path must use metadata')


def _translator(client, **overrides):
    target = LLMTarget(mode='local', name='x',
                       api_url=api_url_for_port(8080), model='test')
    config = build_translation_config(NoneConfig(), target)
    config.update(overrides)
    return LocalLLMTranslator(config, client=client)


class TestTranslateFullText:

    def test_complete_document_is_returned(self):
        client = MetadataClient(CompletionResult(
            '- id: 1\n  translation: 你好，世界。', 'stop'))
        translator = _translator(client)

        result = translator.translate_full_text('Hello, world.', 'zh-tw')

        assert result == '你好，世界。'
        # The output budget is sized to the document, capped by the user's
        # configured ceiling.
        budget = client.calls[0]['max_tokens']
        assert 1024 <= budget <= translator.max_tokens

    def test_length_stopped_document_is_incomplete_not_published(self):
        client = MetadataClient(CompletionResult(
            '- id: 1\n  translation: 你好，世', 'length'))
        translator = _translator(client)

        with pytest.raises(TranslationIncompleteError):
            translator.translate_full_text('A long document', 'zh-tw')

    def test_oversized_document_is_rejected_before_any_request(self):
        client = MetadataClient()
        # A tiny context_size makes even a short document not fit.
        translator = _translator(client, context_size=64,
                                 max_tokens=16384)

        with pytest.raises(TxtTooLargeError):
            translator.translate_full_text('This document will not fit', 'zh-tw')
        assert client.calls == []

    def test_empty_text_needs_no_request(self):
        client = MetadataClient()
        translator = _translator(client)
        assert translator.translate_full_text('   ', 'zh-tw') == ''
        assert client.calls == []

    def test_cancelled_before_dispatch_sends_nothing(self):
        client = MetadataClient()
        translator = _translator(client)

        with pytest.raises(Exception):
            translator.translate_full_text(
                'Some document', 'zh-tw', cancel_check=lambda: True)
        assert client.calls == []


class TestSubtitleTabTxtAndOutputIntegrity:
    """The tab's entry point: routing, rejection, and protected commits."""

    def _make_tab(self, tmp_path, translator):
        from subtitle_tab import SubtitleTranslationTab
        tab = object.__new__(SubtitleTranslationTab)
        tab._translator = translator
        tab._log = Mock()
        tab._on_progress = Mock()
        tab._stop_requested = False
        tab._tmp_path = tmp_path
        return tab

    def test_txt_file_routes_through_full_text_not_recovery(self, tmp_path):
        from subtitle_tab import SubtitleTranslationTab
        source = tmp_path / 'doc.txt'
        source.write_text('Hello document', encoding='utf-8')

        client = MetadataClient(CompletionResult(
            '- id: 1\n  translation: 你好文件', 'stop'))
        translator = _translator(client)
        tab = self._make_tab(tmp_path, translator)

        SubtitleTranslationTab._translate_file(
            tab, translator, str(source), 'zh-tw')

        out = tmp_path / output_path_for(str(source), 'zh-tw', False)
        assert out.name == 'doc_Traditional Chinese.txt'
        assert out.read_text(encoding='utf-8') == '你好文件'
        assert len(client.calls) == 1

    def test_length_stopped_txt_writes_no_output(self, tmp_path):
        from subtitle_tab import SubtitleTranslationTab
        source = tmp_path / 'doc.txt'
        source.write_text('Hello document', encoding='utf-8')

        client = MetadataClient(CompletionResult(
            '- id: 1\n  translation: 部分', 'length'))
        translator = _translator(client)
        tab = self._make_tab(tmp_path, translator)

        with pytest.raises(TranslationIncompleteError):
            SubtitleTranslationTab._translate_file(
                tab, translator, str(source), 'zh-tw')

        assert not (tmp_path / 'doc_Traditional Chinese.txt').exists()
        assert not any(call.args[0] == 'SUCCESS'
                       for call in tab._log.call_args_list)

    def test_invalid_nonempty_srt_with_zero_cues_is_rejected(self, tmp_path):
        """Corrupt input must fail loudly, never publish an empty 'success'.

        With Replace original this used to overwrite the source file with
        an empty SRT and log Saved.
        """
        from subtitle_tab import SubtitleTranslationTab
        source = tmp_path / 'broken.srt'
        original = 'not a subtitle file, just prose'
        source.write_text(original, encoding='utf-8')

        tab = self._make_tab(tmp_path, Mock())

        with pytest.raises(RuntimeError, match='no valid subtitle cues'):
            SubtitleTranslationTab._translate_file(
                tab, Mock(), str(source), 'zh-tw', replace_original=True)

        # The original file is untouched.
        assert source.read_text(encoding='utf-8') == original

    def test_stop_landing_before_commit_publishes_nothing(self, tmp_path):
        """A Stop between the last response and the write suppresses output."""
        from subtitle_tab import SubtitleTranslationTab
        source = tmp_path / 'a.srt'
        source.write_text(
            '1\n00:00:00,000 --> 00:00:01,000\nHello\n', encoding='utf-8')

        tab = self._make_tab(tmp_path, None)
        tab._stop_requested = False

        class StopAfterTranslate:
            def translate_srt(self, *args, **kwargs):
                translated = [
                    Cue(line=1, start_time=0, end_time=1000, text='你好')]
                tab._stop_requested = True  # Stop lands right here
                return translated

        translator = StopAfterTranslate()

        SubtitleTranslationTab._translate_file(
            tab, translator, str(source), 'zh-tw')

        assert not (tmp_path / 'a_Traditional Chinese.srt').exists()
        logged = ' '.join(str(c.args[-1]) for c in tab._log.call_args_list)
        # Neither the success path nor a half-published output happened.
        assert 'Saved' not in logged
        assert 'cancelled before commit' not in logged.lower()
