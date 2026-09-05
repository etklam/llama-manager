"""Replay incomplete model responses through the full subtitle translation path."""
from unittest.mock import Mock

import pytest

from config_helpers import build_translation_config
from translation.local_llm_translator import LocalLLMTranslator
from utils.srt_parser import Cue


def translator_with(*responses):
    config = build_translation_config(NoneConfig(), 8080, "test")
    config['max_workers'] = 1
    client = Mock()
    client.complete.side_effect = responses
    return LocalLLMTranslator(config, client=client), client


class NoneConfig:
    def get(self, key, default=None):
        return default


def cues(count):
    return [Cue(line=i + 1, start_time=i * 1000, end_time=(i + 1) * 1000,
                text=f"Source {i + 1}") for i in range(count)]


def test_preamble_does_not_hide_a_valid_first_translation():
    translator, client = translator_with(
        'Here is the translation:\n```yaml\n- id: 1\n  translation: 你好\n```',
        RuntimeError("retry should not be needed"),
    )
    result = translator.translate_srt(cues(1), 'zh-tw')
    assert result[0].text == '你好'
    assert client.complete.call_count == 1


def test_missing_numbered_line_gets_targeted_retry():
    translator, client = translator_with(
        '1. 一\n3. 三\n4. 四', '- id: 1\n  translation: 二',
    )
    result = translator.translate_srt(cues(4), 'zh-tw')
    assert [cue.text for cue in result] == ['一', '二', '三', '四']
    assert client.complete.call_count == 2


@pytest.mark.parametrize('first_reply', ['3. 三', '- id: 3\n  translation: 三'])
def test_sparse_response_preserves_ids_and_recovers_only_missing_lines(first_reply):
    translator, client = translator_with(
        first_reply, '- id: 1\n  translation: 一',
        '- id: 1\n  translation: 二', '- id: 1\n  translation: 四',
    )
    result = translator.translate_srt(cues(4), 'zh-tw')
    assert [cue.text for cue in result] == ['一', '二', '三', '四']
    assert client.complete.call_count == 4


def test_exhausted_recovery_is_failure_instead_of_success_with_source():
    translator, _ = translator_with(
        '- id: 1\n  translation:', RuntimeError('server timeout'),
    )
    with pytest.raises(RuntimeError, match='L1'):
        translator.translate_srt(cues(1), 'zh-tw')


def test_empty_recovery_is_failure_instead_of_blank_subtitle():
    translator, _ = translator_with(RuntimeError('batch failed'), '')
    with pytest.raises(RuntimeError, match='L1'):
        translator.translate_srt(cues(1), 'zh-tw')


@pytest.mark.parametrize('reply', ['', '- id: 1\n  translation:'])
def test_single_translation_rejects_empty_or_unfinished_structured_response(reply):
    translator, _ = translator_with(reply)
    with pytest.raises(RuntimeError):
        translator.translate('Hello', 'zh-tw')


@pytest.mark.parametrize('route', ['subtitle', 'pipeline'])
@pytest.mark.parametrize('replace_original', [False, True])
def test_failed_translation_preserves_existing_files(tmp_path, route, replace_original):
    from pipeline_runner import PipelineRunner
    from subtitle_tab import SubtitleTranslationTab
    from utils.srt_parser import output_path_for

    source = tmp_path / 'source.srt'
    original = '1\n00:00:00,000 --> 00:00:01,000\nHello\n'
    source.write_text(original, encoding='utf-8')
    output = tmp_path / output_path_for(str(source), 'zh-tw', replace_original)
    if not replace_original:
        output.write_text('previous complete translation', encoding='utf-8')
    previous = output.read_bytes()
    translator, _ = translator_with('- id: 1\n  translation:', RuntimeError('timeout'))

    if route == 'subtitle':
        caller = object.__new__(SubtitleTranslationTab)
        caller._translator = translator
        caller._log = Mock()
        caller._on_progress = Mock()
    else:
        caller = object.__new__(PipelineRunner)
        caller._config_manager = NoneConfig()
        caller._get_port = lambda: 8080
        caller._get_current_model = lambda: 'test'
        caller._preflight_plan = None
        caller._get_translator = lambda config: translator
        caller._on_progress = Mock()
        caller._on_log = Mock()

    with pytest.raises(RuntimeError, match='L1'):
        caller._translate_file(str(source), 'zh-tw', replace_original=replace_original)
    assert output.read_bytes() == previous
    assert source.read_text(encoding='utf-8') == original
