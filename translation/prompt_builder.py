# -*- coding: utf-8 -*-
"""
Prompt Builder Module

Deduplicates the translation prompt templates used by LocalLLMTranslator.

Instead of maintaining 4 nearly-identical prompt variants
(single_step/two_step x simplified/traditional), we store 2 base templates
(one per mode) and convert simplified Chinese to traditional using a
character mapping at runtime.
"""

# Character mapping: simplified Chinese -> traditional Chinese
# Covers all characters that differ between the simplified and traditional
# prompt variants.  "数组" -> "陣列" is a vocabulary change, not a simple
# per-character conversion, so it is handled as a two-character sequence.
_SIMPLIFIED_TO_TRADITIONAL = str.maketrans({
    # In prompt order of first appearance:
    '请': '請',
    '将': '將',
    '对': '對',
    '里': '裡',
    '译': '譯',
    '为': '為',
    '达': '達',
    '术': '術',
    '让': '讓',
    '惯': '慣',
    '结': '結',     # 结果 -> 結果
    '开': '開',     # 开始 -> 開始
    '据': '據',     # 根据 -> 根據
    '还': '還',     # 还是 -> 還是
    '进': '進',     # 进行 -> 進行
    '语': '語',     # 术语 -> 術語
    '体': '體',     # 媒体 -> 媒體
    '称': '稱',     # 名称 -> 名稱
    '习': '習',     # 习惯 -> 習慣
    '务': '務',     # 任务 -> 任務
})

# Multi-character vocabulary replacements (simplified -> traditional)
# These are word-level differences, not simple character conversions.
_VOCAB_REPLACEMENTS = [
    ('数组', '陣列'),   # "array" - simplified uses 数组, traditional uses 陣列
]


def _to_traditional(text: str) -> str:
    """Convert simplified Chinese prompt text to traditional Chinese.

    Applies per-character conversion via str.translate(), then performs
    multi-character vocabulary replacements (e.g. 数组 -> 陣列).
    """
    result = text.translate(_SIMPLIFIED_TO_TRADITIONAL)
    for simplified_word, traditional_word in _VOCAB_REPLACEMENTS:
        result = result.replace(simplified_word, traditional_word)
    return result


# ---------------------------------------------------------------------------
# Base prompt templates (simplified Chinese).
# These are the single source of truth for the prompt text.
# ---------------------------------------------------------------------------

_SINGLE_STEP_TEMPLATE = (
    "请将下面 YAML 对象里的 source 字段意译为 {target_language}，"
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
    "{yaml_text}"
)

_TWO_STEP_TEMPLATE = (
    "请根据以下要求完成翻译任务：\n"
    "1. 将下面 YAML 对象里的 source 字段直接翻译为 {target_language}，"
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
    "{yaml_text}"
)

# Pre-computed traditional Chinese templates (converted from simplified)
_SINGLE_STEP_TEMPLATE_TRADITIONAL = _to_traditional(_SINGLE_STEP_TEMPLATE)
_TWO_STEP_TEMPLATE_TRADITIONAL = _to_traditional(_TWO_STEP_TEMPLATE)

# Template lookup: (mode, locale) -> template string
_TEMPLATES = {
    ('single_step', 'simplified'): _SINGLE_STEP_TEMPLATE,
    ('single_step', 'traditional'): _SINGLE_STEP_TEMPLATE_TRADITIONAL,
    ('two_step', 'simplified'): _TWO_STEP_TEMPLATE,
    ('two_step', 'traditional'): _TWO_STEP_TEMPLATE_TRADITIONAL,
}

_VALID_MODES = {'single_step', 'two_step'}
_VALID_LOCALES = {'simplified', 'traditional'}


def build_translation_prompt(
    mode: str,
    locale: str,
    target_language: str,
    yaml_text: str,
) -> str:
    """Build a translation prompt for the given mode and locale.

    Args:
        mode: Translation mode - 'single_step' or 'two_step'.
            single_step: produces prompts with a 'translation' field.
            two_step: produces prompts with 'step1' and 'step2' fields.
        locale: Chinese locale variant - 'simplified' or 'traditional'.
            Controls whether simplified or traditional Chinese characters
            are used in the prompt instructions.
        target_language: The target language name to embed in the prompt
            (e.g. 'Simplified Chinese', 'English').
        yaml_text: The YAML-formatted input text to append at the end.

    Returns:
        The formatted prompt string ready to use as user message content.

    Raises:
        ValueError: If mode or locale is not a valid option.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {sorted(_VALID_MODES)}"
        )
    if locale not in _VALID_LOCALES:
        raise ValueError(
            f"Invalid locale '{locale}'. Must be one of: {sorted(_VALID_LOCALES)}"
        )

    template = _TEMPLATES[(mode, locale)]
    return template.format(
        target_language=target_language,
        yaml_text=yaml_text,
    )
