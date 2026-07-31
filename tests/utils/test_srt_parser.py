"""
Comprehensive test suite for SRT parser module.

These tests follow TDD (Test-Driven Development) principles and define
the expected behavior of the SRT parser before implementation exists.

The module `utils.srt_parser` does not exist yet - these tests WILL FAIL
until the implementation is created.
"""

import pytest
from pathlib import Path


# This import will FAIL because the module doesn't exist yet
from utils.srt_parser import (
    Cue,
    parse_srt_from_file,
    parse_srt_from_string,
    generate_srt_from_list,
    format_time,
    time_to_milliseconds,
    milliseconds_to_time,
    output_path_for,
    collapse_repeats,
)


# Sample SRT content for testing
SAMPLE_SRT_CONTENT = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:03,500 --> 00:00:06,000
This is a test subtitle

3
00:00:07,000 --> 00:00:10,000
Multiline subtitle
with two lines
"""

# Sample SRT content with various time formats
SRT_WITH_TIME_FORMATS = """1
00:00:01,000 --> 00:00:02,000
First subtitle

2
0:0:5.500 --> 0:0:7.500
Second subtitle with different format

3
001:02:03,999 --> 001:02:05,999
Third subtitle with padded hours
"""

# Malformed SRT content
MALFORMED_SRT = """1
Missing timestamp
This has no timestamp

2
00:00:03,000 --> 00:00:05,000
This one is valid

3
Invalid timestamp format --> invalid
This should be skipped
"""

# Empty SRT
EMPTY_SRT = ""

# SRT with multiline text
SRT_WITH_MULTILINE = """1
00:00:01,000 --> 00:00:04,000
Line 1
Line 2
Line 3

2
00:00:05,000 --> 00:00:08,000
Another multiline
subtitle block
"""


class TestParseSrtFromFile:
    """Test parsing SRT from file."""

    def test_parse_srt_from_file(self, tmp_path):
        """Test parsing SRT content from a temporary file."""
        # Create temporary SRT file
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(SAMPLE_SRT_CONTENT, encoding='utf-8')

        # Parse the file
        result = parse_srt_from_file(str(srt_file))

        # Verify structure
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(entry, Cue) for entry in result)

        # Verify first subtitle
        assert result[0].line == 1
        assert result[0].start_time == 1000  # 00:00:01,000 in milliseconds
        assert result[0].end_time == 3000
        assert result[0].text == "Hello world"
        # The timestamp string is derived from ms at generation time
        assert generate_srt_from_list(result[:1]).startswith(
            "1\n00:00:01,000 --> 00:00:03,000\nHello world")

        # Verify second subtitle
        assert result[1].line == 2
        assert result[1].start_time == 3500
        assert result[1].end_time == 6000
        assert result[1].text == "This is a test subtitle"

        # Verify third subtitle
        assert result[2].line == 3
        assert result[2].start_time == 7000
        assert result[2].end_time == 10000
        assert result[2].text == "Multiline subtitle\nwith two lines"

    def test_parse_srt_from_file_with_gbk_encoding(self, tmp_path):
        """Test parsing SRT file with GBK encoding."""
        # Create SRT file with GBK encoding
        srt_file = tmp_path / "test_gbk.srt"
        gbk_content = "1\n00:00:01,000 --> 00:00:03,000\n中文字幕\n"
        srt_file.write_text(gbk_content, encoding='gbk')

        # Should handle GBK encoding
        result = parse_srt_from_file(str(srt_file))
        assert len(result) == 1
        assert result[0].text == "中文字幕"

    def test_parse_srt_from_file_not_found(self, tmp_path):
        """Test parsing non-existent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            parse_srt_from_file(str(tmp_path / "nonexistent.srt"))


class TestParseSrtFromString:
    """Test parsing SRT from string directly."""

    def test_parse_srt_from_string(self):
        """Test parsing SRT content from string."""
        result = parse_srt_from_string(SAMPLE_SRT_CONTENT)

        # Verify structure matches file parsing
        assert isinstance(result, list)
        assert len(result) == 3

        # Verify structure
        assert result[0].line == 1
        assert result[0].start_time == 1000
        assert result[0].end_time == 3000
        assert result[0].text == "Hello world"

    def test_parse_srt_from_string_with_whitespace(self):
        """Test parsing SRT string with extra whitespace."""
        srt_with_whitespace = """

        1
        00:00:01,000 --> 00:00:02,000
        Test subtitle


        2
        00:00:03,000 --> 00:00:04,000
        Another test

        """
        result = parse_srt_from_string(srt_with_whitespace)
        assert len(result) == 2
        assert result[0].text == "Test subtitle"
        assert result[1].text == "Another test"


class TestGenerateSrtFromList:
    """Test generating SRT format from a list of Cue objects."""

    def test_generate_srt_from_list(self):
        """Test generating SRT string from a list of Cues."""
        subtitle_list = [
            Cue(line=1, start_time=1000, end_time=3000, text='First subtitle'),
            Cue(line=2, start_time=4000, end_time=6000, text='Second subtitle')
        ]

        result = generate_srt_from_list(subtitle_list)

        # Verify SRT format
        assert isinstance(result, str)
        assert '1\n00:00:01,000 --> 00:00:03,000\nFirst subtitle' in result
        assert '2\n00:00:04,000 --> 00:00:06,000\nSecond subtitle' in result

    def test_generate_srt_from_list_numbers_by_position(self):
        """Entries are numbered by position, not by their line field."""
        subtitle_list = [
            Cue(line=7, start_time=1000, end_time=3000, text='First'),
            Cue(line=3, start_time=4000, end_time=6000, text='Second'),
        ]

        result = generate_srt_from_list(subtitle_list)

        assert result.startswith('1\n00:00:01,000 --> 00:00:03,000\nFirst')
        assert '2\n00:00:04,000 --> 00:00:06,000\nSecond' in result

    def test_generate_srt_from_list_multiline_text(self):
        """Test generating SRT with multiline subtitle text."""
        subtitle_list = [
            Cue(line=1, start_time=1000, end_time=4000,
                text='Line 1\nLine 2\nLine 3')
        ]

        result = generate_srt_from_list(subtitle_list)
        assert 'Line 1\nLine 2\nLine 3' in result

    def test_generate_srt_from_empty_list(self):
        """Test generating SRT from empty list."""
        result = generate_srt_from_list([])
        assert result == ""


class TestEmptySrt:
    """Test handling of empty SRT content."""

    def test_empty_srt_string(self):
        """Test parsing empty SRT string returns empty list."""
        result = parse_srt_from_string(EMPTY_SRT)
        assert result == []

    def test_empty_srt_file(self, tmp_path):
        """Test parsing empty SRT file returns empty list."""
        srt_file = tmp_path / "empty.srt"
        srt_file.write_text("", encoding='utf-8')

        result = parse_srt_from_file(str(srt_file))
        assert result == []


class TestMalformedSrt:
    """Test handling of malformed SRT content."""

    def test_malformed_srt_skips_invalid_entries(self):
        """Test that malformed SRT skips invalid entries but parses valid ones."""
        result = parse_srt_from_string(MALFORMED_SRT)

        # Should only parse the valid subtitle
        assert len(result) == 1
        assert result[0].line == 1
        assert result[0].start_time == 3000
        assert result[0].end_time == 5000
        assert result[0].text == "This one is valid"

    def test_srt_with_missing_timestamps(self):
        """Test SRT with missing end timestamp."""
        malformed_srt = """1
00:00:01,000 -->
Missing end time

2
00:00:03,000 --> 00:00:05,000
Valid subtitle
"""
        result = parse_srt_from_string(malformed_srt)
        # Should skip the malformed entry and parse the valid one
        assert len(result) == 1
        assert result[0].text == "Valid subtitle"

    def test_srt_with_invalid_time_format(self):
        """Test SRT with completely invalid time format."""
        invalid_srt = """1
invalid:time:format --> invalid:time:format
This should be skipped

2
00:00:01,000 --> 00:00:02,000
Valid subtitle
"""
        result = parse_srt_from_string(invalid_srt)
        assert len(result) == 1
        assert result[0].text == "Valid subtitle"


class TestMultilineText:
    """Test handling of multiline subtitle text."""

    def test_multiline_text_preserved(self):
        """Test that multiline text is preserved in the text field."""
        result = parse_srt_from_string(SRT_WITH_MULTILINE)

        assert len(result) == 2

        # First subtitle has 3 lines
        assert result[0].text == "Line 1\nLine 2\nLine 3"

        # Second subtitle has 2 lines
        assert result[1].text == "Another multiline\nsubtitle block"

    def test_multiline_text_converted_to_single_line_in_srt(self):
        """Test that multiline text is preserved when generating SRT."""
        subtitle_list = [
            Cue(line=1, start_time=1000, end_time=4000,
                text='Line 1\nLine 2\nLine 3')
        ]

        result = generate_srt_from_list(subtitle_list)

        # Multiline should be preserved in SRT format
        assert 'Line 1\nLine 2\nLine 3' in result


class TestTimeFormatParsing:
    """Test time format conversion and parsing."""

    def test_time_to_milliseconds_standard_format(self):
        """Test converting standard SRT time format to milliseconds."""
        # Standard format: HH:MM:SS,mmm
        assert time_to_milliseconds("00:00:01,000") == 1000
        assert time_to_milliseconds("00:01:00,000") == 60000
        assert time_to_milliseconds("01:00:00,000") == 3600000
        assert time_to_milliseconds("01:23:45,678") == 5025678

    def test_time_to_milliseconds_variations(self):
        """Test converting various time format variations."""
        # Format with dots instead of comma
        assert time_to_milliseconds("00:00:01.000") == 1000

        # Format without leading zeros
        assert time_to_milliseconds("0:0:1,0") == 1000

        # Format with extra padding
        assert time_to_milliseconds("001:02:03,999") == 3723999

    def test_milliseconds_to_time(self):
        """Test converting milliseconds back to SRT time format."""
        assert milliseconds_to_time(1000) == "00:00:01,000"
        assert milliseconds_to_time(60000) == "00:01:00,000"
        assert milliseconds_to_time(3600000) == "01:00:00,000"
        assert milliseconds_to_time(5025678) == "01:23:45,678"
        assert milliseconds_to_time(0) == "00:00:00,000"

    def test_format_time_normalizes_format(self):
        """Test that format_time normalizes various time formats."""
        # Standard format should remain unchanged
        assert format_time("00:00:01,000") == "00:00:01,000"

        # Format with dots should be converted to commas
        assert format_time("00:00:01.000") == "00:00:01,000"

        # Format without leading zeros should be normalized
        assert format_time("0:0:1,0") == "00:00:01,000"

        # Format with extra padding should be normalized
        assert format_time("001:02:03,999") == "01:02:03,999"

        # Empty string should return default
        assert format_time("") == "00:00:00,000"

    def test_parse_srt_with_various_time_formats(self):
        """Test parsing SRT with various time format variations."""
        result = parse_srt_from_string(SRT_WITH_TIME_FORMATS)

        # Should parse all three entries correctly
        assert len(result) == 3

        # First: standard format
        assert result[0].start_time == 1000
        assert result[0].end_time == 2000

        # Second: non-standard format (0:0:5.500)
        assert result[1].start_time == 5500
        assert result[1].end_time == 7500

        # Third: padded hours (001:02:03,999)
        assert result[2].start_time == 3723999
        assert result[2].end_time == 3725999


class TestSrtStructure:
    """Test the structure and required fields of parsed SRT."""

    def test_parsed_srt_has_required_fields(self):
        """Test that parsed SRT entries are Cue objects with all fields."""
        result = parse_srt_from_string(SAMPLE_SRT_CONTENT)

        for entry in result:
            assert isinstance(entry, Cue)
            assert isinstance(entry.line, int)
            assert isinstance(entry.start_time, int)
            assert isinstance(entry.end_time, int)
            assert isinstance(entry.text, str)
            assert entry.line >= 1
            assert entry.start_time >= 0
            assert entry.end_time >= entry.start_time

    def test_line_numbers_are_sequential(self):
        """Test that line numbers are sequential starting from 1."""
        result = parse_srt_from_string(SAMPLE_SRT_CONTENT)

        for i, entry in enumerate(result, start=1):
            assert entry.line == i

    def test_timestamps_are_derived_from_ms_at_generation(self):
        """The SRT timestamp string is derived at generation, not stored."""
        result = parse_srt_from_string(SAMPLE_SRT_CONTENT)

        generated = generate_srt_from_list(result[:1])
        assert generated.startswith(
            "1\n00:00:01,000 --> 00:00:03,000\nHello world")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_srt_with_special_characters(self):
        """Test SRT with special characters in text."""
        srt_with_special = """1
00:00:01,000 --> 00:00:02,000
Text with <i>HTML</i> tags & "quotes"

2
00:00:03,000 --> 00:00:04,000
More special chars: @#$%^&*()
"""
        result = parse_srt_from_string(srt_with_special)

        # HTML tags should be preserved
        assert '<i>HTML</i>' in result[0].text
        assert '& "quotes"' in result[0].text
        assert '@#$%^&*()' in result[1].text

    def test_srt_with_unicode_characters(self):
        """Test SRT with Unicode characters."""
        srt_unicode = """1
00:00:01,000 --> 00:00:02,000
Unicode: 中文 日本語 한국어

2
00:00:03,000 --> 00:00:04,000
Emojis: 😀 🎉 🚀
"""
        result = parse_srt_from_string(srt_unicode)

        assert '中文 日本語 한국어' in result[0].text
        assert '😀 🎉 🚀' in result[1].text

    def test_srt_with_very_long_timestamp(self):
        """Test SRT with very long timestamps (e.g., for long videos)."""
        srt_long_time = """1
10:30:45,123 --> 10:30:50,456
Long video subtitle
"""
        result = parse_srt_from_string(srt_long_time)

        assert result[0].start_time == 37845123  # 10h 30m 45s 123ms
        assert result[0].end_time == 37850456

    def test_srt_with_consecutive_blank_lines(self):
        """Test SRT with consecutive blank lines between entries."""
        srt_blank_lines = """1
00:00:01,000 --> 00:00:02,000
First



2
00:00:03,000 --> 00:00:04,000
Second
"""
        result = parse_srt_from_string(srt_blank_lines)

        assert len(result) == 2
        assert result[0].text == "First"
        assert result[1].text == "Second"


class TestRoundTrip:
    """Test round-trip conversion: parse -> generate -> parse."""

    def test_round_trip_preserves_data(self):
        """Test that parsing and generating SRT preserves the data."""
        # Original SRT
        original = SAMPLE_SRT_CONTENT

        # Parse
        parsed = parse_srt_from_string(original)

        # Generate
        generated = generate_srt_from_list(parsed)

        # Parse again
        reparsed = parse_srt_from_string(generated)

        # Cue equality covers line, times, and text
        assert parsed == reparsed

    def test_round_trip_from_hand_built_cues(self):
        """Cues built by hand survive generate -> parse unchanged."""
        cues = [
            Cue(line=1, start_time=1000, end_time=3000, text="First cue"),
            Cue(line=2, start_time=3500, end_time=6000,
                text="Second cue\nwith two lines"),
        ]

        generated = generate_srt_from_list(cues)
        reparsed = parse_srt_from_string(generated)

        assert reparsed == cues

    def test_round_trip_preserves_unicode(self):
        """Unicode and special characters survive the round trip."""
        original = """1
00:00:01,000 --> 00:00:02,000
中文 日本語 한국어 😀
"""
        parsed = parse_srt_from_string(original)
        generated = generate_srt_from_list(parsed)
        reparsed = parse_srt_from_string(generated)

        assert reparsed == parsed


class TestOutputPathFor:
    # ponytail: one fn, two callers, no OutputPathResolver class

    def test_replace_original_returns_input(self):
        assert output_path_for("/x/video.srt", "zh-cn", True) == "/x/video.srt"

    def test_no_replace_zh_cn_uses_lang_name(self):
        out = Path(output_path_for("/x/video.srt", "zh-cn", False))
        assert out.name == "video_Simplified Chinese.srt"

    def test_no_replace_en(self):
        out = Path(output_path_for("/x/video.srt", "en", False))
        assert out.name == "video_English.srt"

    def test_unknown_lang_falls_back_to_code(self):
        out = Path(output_path_for("/x/video.srt", "xx", False))
        assert out.name == "video_xx.srt"

    def test_works_for_txt(self):
        out = Path(output_path_for("/x/notes.txt", "en", False))
        assert out.name == "notes_English.txt"
        assert out.suffix == ".txt"

    def test_windows_backslashes(self):
        out = Path(output_path_for(r"C:\dir\video.srt", "en", False))
        assert out.parent.drive == "C:"
        assert out.name == "video_English.srt"

    def test_preserves_directory(self):
        out = Path(output_path_for("/some/deep/path/v.srt", "en", False))
        assert out.name == "v_English.srt"
        assert len(out.parent.parts) == 4


class TestCollapseRepeats:
    """Tests for collapse_repeats: compressing pathological repetition."""

    def test_collapses_separator_delimited_run(self):
        # The reported case: one short unit + separator repeated ~100 times.
        text = "、".join(["あ"] * 100)
        assert collapse_repeats(text) == "あ..."

    def test_collapses_contiguous_run(self):
        assert collapse_repeats("あ" * 20) == "あ..."

    def test_collapses_multichar_unit(self):
        assert collapse_repeats("なにこれ" * 6) == "なにこれ..."

    def test_collapses_long_vowel_mark_run(self):
        assert collapse_repeats("ノー" * 20) == "ノー..."

    def test_preserves_text_around_run(self):
        text = "Help! " + "あ、" * 30 + "end"
        assert collapse_repeats(text) == "Help! あ...end"

    def test_short_emphasis_is_untouched(self):
        # Below _MIN_REPEAT_COUNT (5): ordinary emphasis must survive.
        assert collapse_repeats("はは") == "はは"
        assert collapse_repeats("あああ") == "あああ"
        assert collapse_repeats("は、は") == "は、は"

    def test_plain_text_unchanged(self):
        text = "no repeats here at all"
        assert collapse_repeats(text) == text

    def test_empty_and_none_safe(self):
        assert collapse_repeats("") == ""
        assert collapse_repeats(None) is None

    def test_multiple_distinct_runs_each_collapsed(self):
        text = "あ" * 10 + " then " + "ね、" * 10
        assert collapse_repeats(text) == "あ... then ね..."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
