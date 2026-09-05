"""English instructions with language-independent output schemas."""
import pytest

from translation.prompt_builder import build_translation_prompt


@pytest.mark.parametrize('mode', ['single_step', 'two_step'])
@pytest.mark.parametrize('locale', ['simplified', 'traditional'])
@pytest.mark.parametrize('target', ['English', 'Simplified Chinese', 'Traditional Chinese'])
def test_instructions_are_english_and_keep_target_and_schema(mode, locale, target):
    source = '- id: 7\n  source: "Hello"'
    result = build_translation_prompt(mode, locale, target, source)
    assert result.isascii()
    assert target in result
    assert result.endswith(source)
    assert result.count(source) == 1
    assert 'id exactly once' in result
    assert 'Do not skip, merge, split, or add entries.' in result
    if mode == 'single_step':
        assert 'translation: TARGET_TEXT' in result
        assert 'step1' not in result and 'step2' not in result
    else:
        assert 'step1: TARGET_TEXT' in result
        assert 'step2: TARGET_TEXT' in result
        assert 'translation: TARGET_TEXT' not in result


@pytest.mark.parametrize('mode', ['single_step', 'two_step'])
def test_locale_does_not_change_english_instructions(mode):
    assert build_translation_prompt(mode, 'simplified', 'English', '') == (
        build_translation_prompt(mode, 'traditional', 'English', ''))


def test_empty_input_retains_input_marker():
    result = build_translation_prompt('single_step', 'traditional', 'zh-tw', '')
    assert result.endswith('Translate the following input:\n\n')


def test_non_english_source_and_context_are_preserved():
    source = '- id: 1\n  source: "你好"'
    result = build_translation_prompt('single_step', 'traditional', 'en', source,
                                      context='餐廳對話')
    assert result.endswith(source)
    assert 'Context: "餐廳對話"' in result
    assert result.startswith('Translate each source into English')


@pytest.mark.parametrize('mode,locale,invalid', [
    ('invalid', 'simplified', 'mode'),
    ('single_step', 'invalid', 'locale'),
])
def test_invalid_arguments_are_rejected(mode, locale, invalid):
    with pytest.raises(ValueError, match=invalid):
        build_translation_prompt(mode, locale, 'English', '')
