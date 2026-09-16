"""Subtitle content and ID integrity: A/B/C replay scenarios.

Position-derived IDs used to let a reordered, short, or duplicated batch
answer shift translations onto the wrong cues. These tests replay the
named shapes: ID-less B/C, reordered explicit IDs, duplicate IDs, a
numeric dialogue line in the SRT itself, and a truncated final entry.
"""
from unittest.mock import Mock

import pytest

from config_helpers import build_translation_config
from llm_target import LLMTarget, api_url_for_port
from translation.local_llm_translator import (
    LocalLLMTranslator,
    TranslationIncompleteError,
)
from utils.srt_parser import parse_srt_from_string, Cue


class NoneConfig:
    def get(self, key, default=None):
        return default


def _translator(*responses):
    target = LLMTarget(mode='local', name='x',
                       api_url=api_url_for_port(8080), model='test')
    config = build_translation_config(NoneConfig(), target)
    config['max_workers'] = 1
    client = Mock()
    client.complete.side_effect = list(responses)
    return LocalLLMTranslator(config, client=client), client


def abc_cues():
    return [Cue(line=i + 1, start_time=i * 1000, end_time=(i + 1) * 1000,
                text=text)
            for i, text in enumerate(("Alpha line", "Beta line", "Gamma line"))]


def _texts(cues):
    return [cue.text for cue in cues]


class TestBatchIdMapping:

    def test_reordered_explicit_ids_keep_their_own_translations(self):
        translator, client = _translator(
            '- id: 2\n  translation: 二\n'
            '- id: 1\n  translation: 一\n'
            '- id: 3\n  translation: 三\n')

        result = translator.translate_srt(abc_cues(), 'zh-tw')

        assert _texts(result) == ['一', '二', '三']
        assert client.complete.call_count == 1

    def test_duplicate_id_keeps_first_entry_and_recovers_nothing_wrong(self):
        translator, client = _translator(
            '- id: 1\n  translation: 一\n'
            '- id: 1\n  translation: 另一個一\n'
            '- id: 2\n  translation: 二\n',
            # recovery for line 3 only
            '- id: 1\n  translation: 三\n')

        result = translator.translate_srt(abc_cues(), 'zh-tw')

        assert _texts(result) == ['一', '二', '三']

    def test_out_of_range_id_is_ignored_and_recovered(self):
        translator, _ = _translator(
            '- id: 99\n  translation: 不是任何一行\n'
            '- id: 1\n  translation: 一\n',
            '- id: 1\n  translation: 二\n',
            '- id: 1\n  translation: 三\n')

        result = translator.translate_srt(abc_cues(), 'zh-tw')

        assert _texts(result) == ['一', '二', '三']

    def test_id_less_short_answer_is_not_positionally_guessed(self):
        """ID-less B/C answering A/B/C must not be mapped onto A/B.

        The old code invented ids from result position, so 'B' landed on
        line A and 'C' on line B while line C fell to recovery — a silent
        two-line mistranslation.
        """
        translator, client = _translator(
            '- translation: 乙\n- translation: 丙\n',   # B and C, no ids
            '- id: 1\n  translation: 一\n',
            '- id: 1\n  translation: 二\n',
            '- id: 1\n  translation: 三\n')

        result = translator.translate_srt(abc_cues(), 'zh-tw')

        # All three came from targeted recovery with the right source.
        assert _texts(result) == ['一', '二', '三']
        # The id-less batch was never trusted positionally.
        first_call = client.complete.call_args_list[0]
        assert first_call is not None

    def test_full_id_less_answer_with_matching_count_maps_positionally(self):
        translator, client = _translator(
            '- translation: 一\n- translation: 二\n- translation: 三\n')

        result = translator.translate_srt(abc_cues(), 'zh-tw')

        assert _texts(result) == ['一', '二', '三']
        assert client.complete.call_count == 1

    def test_mixed_explicit_and_id_less_keeps_only_explicit(self):
        translator, _ = _translator(
            '- id: 3\n  translation: 三\n'
            '- translation: 乙（無法定位）\n',
            '- id: 1\n  translation: 一\n',
            '- id: 1\n  translation: 二\n')

        result = translator.translate_srt(abc_cues(), 'zh-tw')

        assert _texts(result) == ['一', '二', '三']

    def test_truncated_final_entry_is_recovered_not_accepted(self):
        """A final entry cut mid-field is unfinished, not a translation."""
        translator, _ = _translator(
            '- id: 1\n  translation: 一\n'
            '- id: 2\n  translation: 二\n'
            '- id: 3\n  translation:',        # unfinished field
            '- id: 1\n  translation: 三\n')

        result = translator.translate_srt(abc_cues(), 'zh-tw')

        assert _texts(result) == ['一', '二', '三']

    def test_commentary_and_refusal_are_never_translations(self):
        """A refusal answer produces a failed file, not 'translated' text."""
        translator, _ = _translator(
            "I'm sorry, I cannot help with translating this content.",
            "I'm sorry, I cannot help with translating this content.",
        )

        with pytest.raises(TranslationIncompleteError):
            translator.translate_srt(abc_cues(), 'zh-tw')


class TestSrtDialogueDigits:

    def test_digit_only_dialogue_line_is_text_not_a_block_number(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:02,000\nRoom number\n123\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nNext cue\n"
        )
        cues = parse_srt_from_string(srt)
        assert len(cues) == 2
        assert cues[0].text == 'Room number\n123'

    def test_numeric_subtitle_123_before_timestamp_is_a_block_number(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:02,000\nFirst\n\n"
            "123\n00:00:03,000 --> 00:00:04,000\nAfter the number\n"
        )
        cues = parse_srt_from_string(srt)
        assert len(cues) == 2
        assert cues[1].text == 'After the number'

    def test_multiline_dialogue_and_dashes_are_preserved(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:02,000\n- What?\n- Nothing.\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nSecond\n"
        )
        cues = parse_srt_from_string(srt)
        assert cues[0].text == '- What?\n- Nothing.'
        assert cues[1].text == 'Second'

    def test_trailing_digit_line_without_following_timestamp_is_dialogue(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:02,000\nThe year\n1998\n"
        )
        cues = parse_srt_from_string(srt)
        assert cues[0].text == 'The year\n1998'
