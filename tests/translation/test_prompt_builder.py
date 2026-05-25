# -*- coding: utf-8 -*-
"""
Unit tests for the prompt_builder module.

Tests the deduplicated prompt generation for translation tasks,
covering all combinations of mode (single_step / two_step) and
locale (simplified / traditional).

TDD RED phase: these tests are written BEFORE the implementation.
"""
import pytest
import sys
import os

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from translation.prompt_builder import build_translation_prompt


class TestBuildTranslationPromptSingleStep:
    """Test single_step mode prompts."""

    def test_single_step_simplified_contains_translation_field(self):
        """Single-step prompt should instruct the model to use 'translation' field."""
        result = build_translation_prompt(
            mode='single_step',
            locale='simplified',
            target_language='Simplified Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert 'translation' in result
        assert 'step1' not in result
        assert 'step2' not in result

    def test_single_step_simplified_contains_target_language(self):
        """Prompt should include the target language name."""
        result = build_translation_prompt(
            mode='single_step',
            locale='simplified',
            target_language='Simplified Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert 'Simplified Chinese' in result

    def test_single_step_simplified_yaml_appended(self):
        """The yaml_text should appear at the end of the prompt."""
        yaml_text = '- id: 1\n  source: Hello world'
        result = build_translation_prompt(
            mode='single_step',
            locale='simplified',
            target_language='Simplified Chinese',
            yaml_text=yaml_text
        )
        # yaml_text should be at the end
        assert result.endswith(yaml_text)
        assert result.count(yaml_text) == 1

    def test_single_step_simplified_example_format(self):
        """Single-step prompt should show example with 'translation' field."""
        result = build_translation_prompt(
            mode='single_step',
            locale='simplified',
            target_language='Simplified Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert 'translation: 意译结果' in result

    def test_single_step_simplified_example_request(self):
        """Prompt should contain the example request section."""
        result = build_translation_prompt(
            mode='single_step',
            locale='simplified',
            target_language='Simplified Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert 'source: Source' in result

    def test_single_step_simplified_instruction_text(self):
        """Prompt should contain the key instruction phrases in simplified Chinese."""
        result = build_translation_prompt(
            mode='single_step',
            locale='simplified',
            target_language='Simplified Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert '意译' in result
        assert '信达雅' in result
        assert '术语' in result
        assert '翻译结果' in result

    def test_single_step_simplified_start_marker(self):
        """Prompt should contain the '开始翻译:' start marker."""
        result = build_translation_prompt(
            mode='single_step',
            locale='simplified',
            target_language='Simplified Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert '开始翻译:' in result

    def test_single_step_traditional_contains_translation_field(self):
        """Single-step traditional prompt should use 'translation' field."""
        result = build_translation_prompt(
            mode='single_step',
            locale='traditional',
            target_language='Traditional Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert 'translation' in result
        assert 'step1' not in result
        assert 'step2' not in result

    def test_single_step_traditional_example_format(self):
        """Single-step traditional prompt should show '意譯結果' in example."""
        result = build_translation_prompt(
            mode='single_step',
            locale='traditional',
            target_language='Traditional Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert 'translation: 意譯結果' in result

    def test_single_step_traditional_instruction_text(self):
        """Prompt should contain key phrases in traditional Chinese."""
        result = build_translation_prompt(
            mode='single_step',
            locale='traditional',
            target_language='Traditional Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert '意譯' in result
        assert '信達雅' in result
        assert '術語' in result
        assert '翻譯結果' in result

    def test_single_step_traditional_start_marker(self):
        """Traditional prompt should use '開始翻譯:' start marker."""
        result = build_translation_prompt(
            mode='single_step',
            locale='traditional',
            target_language='Traditional Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert '開始翻譯:' in result

    def test_single_step_traditional_yaml_appended(self):
        """The yaml_text should appear at the end of the traditional prompt."""
        yaml_text = '- id: 1\n  source: 測試文本'
        result = build_translation_prompt(
            mode='single_step',
            locale='traditional',
            target_language='Traditional Chinese',
            yaml_text=yaml_text
        )
        assert result.endswith(yaml_text)


class TestBuildTranslationPromptTwoStep:
    """Test two_step mode prompts."""

    def test_two_step_simplified_contains_step_fields(self):
        """Two-step prompt should instruct the model to use 'step1' and 'step2' fields."""
        result = build_translation_prompt(
            mode='two_step',
            locale='simplified',
            target_language='Simplified Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert 'step1' in result
        assert 'step2' in result
        assert 'translation' not in result

    def test_two_step_simplified_example_format(self):
        """Two-step prompt should show step1/step2 example."""
        result = build_translation_prompt(
            mode='two_step',
            locale='simplified',
            target_language='Simplified Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert 'step1: 直译结果' in result
        assert 'step2: 意译结果' in result

    def test_two_step_simplified_instruction_text(self):
        """Two-step prompt should describe the two-step translation process."""
        result = build_translation_prompt(
            mode='two_step',
            locale='simplified',
            target_language='Simplified Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert '直接翻译' in result
        assert '进行意译' in result
        assert '信达雅' in result
        assert '术语' in result

    def test_two_step_simplified_yaml_appended(self):
        """The yaml_text should appear at the end."""
        yaml_text = '- id: 1\n  source: Hello'
        result = build_translation_prompt(
            mode='two_step',
            locale='simplified',
            target_language='Simplified Chinese',
            yaml_text=yaml_text
        )
        assert result.endswith(yaml_text)

    def test_two_step_traditional_contains_step_fields(self):
        """Two-step traditional prompt should use step1/step2 fields."""
        result = build_translation_prompt(
            mode='two_step',
            locale='traditional',
            target_language='Traditional Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert 'step1' in result
        assert 'step2' in result
        assert 'translation' not in result

    def test_two_step_traditional_example_format(self):
        """Two-step traditional prompt should show '直譯結果' / '意譯結果'."""
        result = build_translation_prompt(
            mode='two_step',
            locale='traditional',
            target_language='Traditional Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert 'step1: 直譯結果' in result
        assert 'step2: 意譯結果' in result

    def test_two_step_traditional_instruction_text(self):
        """Two-step traditional prompt should use traditional characters."""
        result = build_translation_prompt(
            mode='two_step',
            locale='traditional',
            target_language='Traditional Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert '直接翻譯' in result
        assert '進行意譯' in result
        assert '信達雅' in result
        assert '術語' in result

    def test_two_step_traditional_yaml_appended(self):
        """The yaml_text should appear at the end."""
        yaml_text = '- id: 1\n  source: 測試文本'
        result = build_translation_prompt(
            mode='two_step',
            locale='traditional',
            target_language='Traditional Chinese',
            yaml_text=yaml_text
        )
        assert result.endswith(yaml_text)


class TestBuildTranslationPromptLocaleDifferences:
    """Test that simplified and traditional produce correctly different characters."""

    def test_simplified_uses_simplified_characters(self):
        """Simplified prompt should use 请, 将, 译, 结果, etc."""
        result = build_translation_prompt(
            mode='single_step',
            locale='simplified',
            target_language='Simplified Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        # Check simplified-only characters
        assert '请' in result
        assert '将' in result
        assert '对象' in result
        assert '术语' in result
        assert '结果' in result
        assert '数组' in result
        assert '开始' in result

    def test_traditional_uses_traditional_characters(self):
        """Traditional prompt should use 請, 將, 對象, 術語, 結果, etc."""
        result = build_translation_prompt(
            mode='single_step',
            locale='traditional',
            target_language='Traditional Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        # Check traditional-only characters
        assert '請' in result
        assert '將' in result
        assert '對象' in result
        assert '術語' in result
        assert '結果' in result
        assert '陣列' in result
        assert '開始' in result

    def test_simplified_and_traditional_differ(self):
        """Simplified and traditional prompts should NOT be identical."""
        simplified = build_translation_prompt(
            mode='single_step',
            locale='simplified',
            target_language='Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        traditional = build_translation_prompt(
            mode='single_step',
            locale='traditional',
            target_language='Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert simplified != traditional

    def test_both_modes_differ_for_same_locale(self):
        """single_step and two_step prompts should differ for the same locale."""
        single = build_translation_prompt(
            mode='single_step',
            locale='simplified',
            target_language='Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        two = build_translation_prompt(
            mode='two_step',
            locale='simplified',
            target_language='Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        assert single != two


class TestBuildTranslationPromptExactText:
    """Test that the generated prompts match the exact expected text from the original code."""

    def test_single_step_simplified_exact_text(self):
        """Verify exact text for single_step + simplified matches the original prompt."""
        result = build_translation_prompt(
            mode='single_step',
            locale='simplified',
            target_language='Simplified Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        expected = (
            "请将下面 YAML 对象里的 source 字段意译为 Simplified Chinese，"
            "力求信达雅，保留特定的术语或媒体名称（如有），"
            "让文本更通俗易懂，符合中文的表达习惯。\n"
            "将翻译结果放入 YAML 数组中的 translation 字段。\n\n"
            "示例格式:\n"
            "  示例请求:\n"
            "    - id: 1\n"
            "      source: Source\n"
            "  示例结果:\n"
            "    - id: 1\n"
            "      translation: 意译结果\n\n"
            "开始翻译:\n\n"
            "- id: 1\n  source: Hello"
        )
        assert result == expected

    def test_single_step_traditional_exact_text(self):
        """Verify exact text for single_step + traditional."""
        result = build_translation_prompt(
            mode='single_step',
            locale='traditional',
            target_language='Traditional Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        expected = (
            "請將下面 YAML 對象裡的 source 字段意譯為 Traditional Chinese，"
            "力求信達雅，保留特定的術語或媒體名稱（如有），"
            "讓文本更通俗易懂，符合中文的表達習慣。\n"
            "將翻譯結果放入 YAML 陣列中的 translation 字段。\n\n"
            "示例格式:\n"
            "  示例請求:\n"
            "    - id: 1\n"
            "      source: Source\n"
            "  示例結果:\n"
            "    - id: 1\n"
            "      translation: 意譯結果\n\n"
            "開始翻譯:\n\n"
            "- id: 1\n  source: Hello"
        )
        assert result == expected

    def test_two_step_simplified_exact_text(self):
        """Verify exact text for two_step + simplified matches the original prompt."""
        result = build_translation_prompt(
            mode='two_step',
            locale='simplified',
            target_language='Simplified Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        expected = (
            "请根据以下要求完成翻译任务：\n"
            "1. 将下面 YAML 对象里的 source 字段直接翻译为 Simplified Chinese，"
            "保留原文特定的术语或媒体名称（如有）。"
            "将本次翻译的结果放入 YAML 数组中的 step1 字段。\n"
            "2. 根据第一次翻译的结果进行意译，力求信达雅，"
            "但还是要保留特定的术语或媒体名称（如有），"
            "在遵守原意的前提下让文本更通俗易懂，符合中文的表达习惯，"
            "将第二次翻译的结果放入 YAML 数组中的 step2 字段。\n\n"
            "示例格式:\n"
            "  示例请求:\n"
            "    - id: 1\n"
            "      source: Source\n"
            "  示例结果:\n"
            "    - id: 1\n"
            "      step1: 直译结果\n"
            "      step2: 意译结果\n\n"
            "开始翻译:\n\n"
            "- id: 1\n  source: Hello"
        )
        assert result == expected

    def test_two_step_traditional_exact_text(self):
        """Verify exact text for two_step + traditional."""
        result = build_translation_prompt(
            mode='two_step',
            locale='traditional',
            target_language='Traditional Chinese',
            yaml_text='- id: 1\n  source: Hello'
        )
        expected = (
            "請根據以下要求完成翻譯任務：\n"
            "1. 將下面 YAML 對象裡的 source 字段直接翻譯為 Traditional Chinese，"
            "保留原文特定的術語或媒體名稱（如有）。"
            "將本次翻譯的結果放入 YAML 陣列中的 step1 字段。\n"
            "2. 根據第一次翻譯的結果進行意譯，力求信達雅，"
            "但還是要保留特定的術語或媒體名稱（如有），"
            "在遵守原意的前提下讓文本更通俗易懂，符合中文的表達習慣，"
            "將第二次翻譯的結果放入 YAML 陣列中的 step2 字段。\n\n"
            "示例格式:\n"
            "  示例請求:\n"
            "    - id: 1\n"
            "      source: Source\n"
            "  示例結果:\n"
            "    - id: 1\n"
            "      step1: 直譯結果\n"
            "      step2: 意譯結果\n\n"
            "開始翻譯:\n\n"
            "- id: 1\n  source: Hello"
        )
        assert result == expected


class TestBuildTranslationPromptEdgeCases:
    """Test edge cases and input validation."""

    def test_empty_yaml_text(self):
        """Should handle empty yaml_text gracefully."""
        result = build_translation_prompt(
            mode='single_step',
            locale='simplified',
            target_language='Chinese',
            yaml_text=''
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert result.endswith('开始翻译:\n\n')

    def test_multi_line_yaml_text(self):
        """Should handle multi-line yaml_text."""
        yaml_text = "- id: 1\n  source: Hello\n- id: 2\n  source: World"
        result = build_translation_prompt(
            mode='single_step',
            locale='simplified',
            target_language='Chinese',
            yaml_text=yaml_text
        )
        assert result.endswith(yaml_text)

    def test_invalid_mode_raises(self):
        """Should raise ValueError for invalid mode."""
        with pytest.raises(ValueError, match="mode"):
            build_translation_prompt(
                mode='invalid_mode',
                locale='simplified',
                target_language='Chinese',
                yaml_text='- id: 1\n  source: Hello'
            )

    def test_invalid_locale_raises(self):
        """Should raise ValueError for invalid locale."""
        with pytest.raises(ValueError, match="locale"):
            build_translation_prompt(
                mode='single_step',
                locale='korean',
                target_language='Chinese',
                yaml_text='- id: 1\n  source: Hello'
            )

    def test_returns_string(self):
        """Should always return a string."""
        result = build_translation_prompt(
            mode='single_step',
            locale='simplified',
            target_language='Chinese',
            yaml_text='test'
        )
        assert isinstance(result, str)
