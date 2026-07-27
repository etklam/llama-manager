"""
SRT Parser Module

This module provides functionality for parsing and generating SRT subtitle files.
It handles various time formats, encoding issues, and malformed SRT content.

Adapted from pyvideotrans for use in llama-manager.
"""

import re
import copy
from pathlib import Path
from typing import List, Dict, Optional

from constants import TARGET_LANGUAGES


# --- Repetition compression -------------------------------------------------

# A single subtitle cue sometimes contains one short unit repeated dozens of
# times - screams, stutters, or ASR artefacts like
# "あ、あ、あ、あ、..." (often 50-100+ copies). Sent verbatim this burns huge
# amounts of tokens, splits into tiny KV-hungry requests, and can make the model
# hang or emit garbage. We collapse such runs to "<unit>..." before translating.
_MAX_REPEAT_UNIT = 4        # longest repeating unit (in chars) we try to detect
_MIN_REPEAT_COUNT = 5       # collapse only when the unit repeats at least this many times

_REPEAT_RE = re.compile(
    r'(?P<unit>.{1,%d}?)'                    # the repeating unit (non-greedy)
    r'(?P<sep>[\s、，,。.！!？?・･…]*)'          # optional separator between repeats
    r'(?:(?P=unit)(?P=sep)){%d,}'            # unit+sep repeated
    r'(?P=unit)?'                            # optional trailing unit without a separator
    % (_MAX_REPEAT_UNIT, _MIN_REPEAT_COUNT - 1),
    flags=re.DOTALL,
)


def collapse_repeats(text: str, marker: str = '...') -> str:
    """Collapse a run of the same short unit repeated many times to "unit...".

    Handles both separator-delimited runs ("あ、あ、あ、...") and contiguous ones
    ("ああああ..."). A unit is 1-4 characters; a run must reach _MIN_REPEAT_COUNT
    copies before it is collapsed, so ordinary emphasis such as "はは" or "あああ"
    survives untouched. Multiple distinct runs in one string are each collapsed.

    Args:
        text: The (subtitle) text to normalize.
        marker: The suffix appended to the surviving unit (default "...").

    Returns:
        The text with over-long repetitions collapsed; unchanged if none qualify.
    """
    if not text:
        return text
    return _REPEAT_RE.sub(lambda m: m.group('unit') + marker, text)


def format_time(s_time: str = "", separate: str = ',') -> str:
    """
    Normalize various time formats to standard SRT format (HH:MM:SS,mmm).

    Handles various input formats:
    - Standard: 00:00:01,000
    - With dots: 00:00:01.000
    - Without leading zeros: 0:0:1,0
    - With extra padding: 001:02:03,999

    Args:
        s_time: Time string to normalize
        separate: Separator character (default ',')

    Returns:
        Normalized time string in HH:MM:SS,mmm format
    """
    if not s_time.strip():
        return f'00:00:00{separate}000'

    hou, min, sec, ms = 0, 0, 0, 0

    tmp = s_time.strip().split(':')
    if len(tmp) >= 3:
        hou, min, sec = tmp[-3].strip(), tmp[-2].strip(), tmp[-1].strip()
    elif len(tmp) == 2:
        min, sec = tmp[0].strip(), tmp[1].strip()
    elif len(tmp) == 1:
        sec = tmp[0].strip()

    if re.search(r',|\.', str(sec)):
        t = re.split(r',|\.', str(sec))
        sec = t[0].strip()
        ms = t[1].strip() if len(t) > 1 else '0'
    else:
        ms = 0

    hou = f'{int(hou):02}'[-2:]
    min = f'{int(min):02}'[-2:]
    sec = f'{int(sec):02}'
    ms = f'{int(ms):03}'[-3:]

    return f"{hou}:{min}:{sec}{separate}{ms}"


def time_to_milliseconds(time_str: str) -> int:
    """
    Convert SRT time string to milliseconds.

    Handles various formats:
    - HH:MM:SS,mmm
    - HH:MM:SS.mmm
    - With or without leading zeros

    Args:
        time_str: Time string in SRT format

    Returns:
        Time in milliseconds
    """
    # Normalize the time format first
    normalized = format_time(time_str, ',')

    # Parse the normalized time
    match = re.match(r'(\d+):(\d+):(\d+),(\d+)', normalized)
    if match:
        hours, minutes, seconds, milliseconds = map(int, match.groups())
        return hours * 3600000 + minutes * 60000 + seconds * 1000 + milliseconds

    return 0


def milliseconds_to_time(ms: int) -> str:
    """
    Convert milliseconds to SRT time string format.

    Args:
        ms: Time in milliseconds

    Returns:
        Time string in HH:MM:SS,mmm format
    """
    total_seconds, milliseconds = divmod(max(0, int(ms)), 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def _ms_to_time_string(*, ms: int = 0, seconds: Optional[int] = None, sepflag: str = ',') -> str:
    """
    Convert milliseconds or seconds to SRT time string format.

    Args:
        ms: Time in milliseconds
        seconds: Time in seconds (alternative to ms)
        sepflag: Separator character (default ',')

    Returns:
        Time string in HH:MM:SS,mmm format
    """
    total_ms = int(ms if seconds is None else seconds * 1000)
    total_seconds, milliseconds = divmod(max(0, total_ms), 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02}:{minutes:02}:{seconds:02}{sepflag}{milliseconds:03}"


def parse_srt_from_string(srt_string: str) -> List[Dict]:
    """
    Parse SRT content from a string.

    This function parses SRT subtitle format and returns a list of subtitle
    dictionaries. It handles various time formats and skips malformed entries.

    Args:
        srt_string: SRT content as a string

    Returns:
        List of subtitle dictionaries, each containing:
        - line: Sequential line number (int)
        - start_time: Start time in milliseconds (int)
        - end_time: End time in milliseconds (int)
        - text: Subtitle text (str)
        - time: Full time string "HH:MM:SS,mmm --> HH:MM:SS,mmm" (str)
        - startraw: Start time string "HH:MM:SS,mmm" (str)
        - endraw: End time string "HH:MM:SS,mmm" (str)
    """
    if not srt_string.strip():
        return []

    srt_list = []
    time_pattern = r'\s?(\d+):(\d+):(\d+)([,.]\d+)?\s*?-{1,2}>\s*?(\d+):(\d+):(\d+)([,.]\d+)?\s*'
    lines = srt_string.splitlines()
    i = 0

    while i < len(lines):
        time_match = re.match(time_pattern, lines[i].strip())
        if time_match:
            # Parse timestamp
            start_time_groups = time_match.groups()[0:4]
            end_time_groups = time_match.groups()[4:8]

            def parse_time(time_groups):
                h, m, s, ms = time_groups
                ms = ms.replace(',', '').replace('.', '') if ms else "0"
                try:
                    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
                except (ValueError, TypeError):
                    return None

            start_time = parse_time(start_time_groups)
            end_time = parse_time(end_time_groups)

            if start_time is None or end_time is None:
                i += 1
                continue

            i += 1
            text_lines = []

            # Collect text lines until we hit the next subtitle block
            while i < len(lines):
                current_line = lines[i].strip()

                # Stop if we hit an empty line (end of current subtitle)
                if not current_line:
                    i += 1
                    break

                # Stop if we hit a line number (start of next subtitle)
                if re.fullmatch(r'^\d+$', current_line):
                    break

                # Stop if we hit another time pattern
                if re.match(time_pattern, current_line):
                    break

                # Otherwise, it's text
                text_lines.append(lines[i])  # Keep original formatting
                i += 1

            # Skip any empty lines between subtitles
            while i < len(lines) and not lines[i].strip():
                i += 1

            text = ('\n'.join(text_lines)).strip()
            text = re.sub(r'\n{2,}', '\n', text, flags=re.I | re.S).strip()

            it = {
                "line": len(srt_list) + 1,
                "start_time": int(start_time),
                "end_time": int(end_time),
                "text": text if text else "",
            }
            it['startraw'] = _ms_to_time_string(ms=it['start_time'])
            it['endraw'] = _ms_to_time_string(ms=it['end_time'])
            it["time"] = f"{it['startraw']} --> {it['endraw']}"
            srt_list.append(it)
        else:
            i += 1

    return srt_list


def parse_srt_from_file(srt_file: str) -> List[Dict]:
    """
    Parse SRT content from a file.

    This function reads an SRT file and parses its content. It handles
    both UTF-8 and GBK encodings automatically.

    Args:
        srt_file: Path to the SRT file

    Returns:
        List of subtitle dictionaries (see parse_srt_from_string)

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    file_path = Path(srt_file)

    if not file_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_file}")

    content = ""

    # Try UTF-8 first
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    except UnicodeDecodeError:
        # Fall back to GBK encoding
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read().strip()
        except UnicodeDecodeError:
            # If both fail, try with errors='ignore'
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()

    return parse_srt_from_string(content)


def generate_srt_from_list(subtitle_list: List[Dict]) -> str:
    """
    Generate SRT format string from a list of subtitle dictionaries.

    This function converts a list of subtitle dictionaries back into
    SRT format string. It handles various input formats:
    - With 'time' field: "00:00:01,000 --> 00:00:02,000"
    - With 'startraw' and 'endraw' fields
    - With 'start_time' and 'end_time' fields (milliseconds)

    Args:
        subtitle_list: List of subtitle dictionaries

    Returns:
        SRT format string
    """
    if not subtitle_list:
        return ""

    txt = ""
    line = 0

    for it in subtitle_list:
        line += 1

        if "startraw" not in it:
            # Check for complete time field
            if 'time' in it:
                startraw, endraw = it['time'].strip().split(" --> ")
                startraw = format_time(startraw.strip().replace('.', ','), ',')
                endraw = format_time(endraw.strip().replace('.', ','), ',')
            elif 'start_time' in it and 'end_time' in it:
                # Use millisecond values
                startraw = _ms_to_time_string(ms=it['start_time'])
                endraw = _ms_to_time_string(ms=it['end_time'])
            else:
                # Default to 00:00:00,000 if no time information
                startraw = "00:00:00,000"
                endraw = "00:00:00,000"
        else:
            # Use existing raw time strings
            startraw = it['startraw']
            endraw = it['endraw']

        txt += f"{line}\n{startraw} --> {endraw}\n{it['text']}\n\n"

    return txt.strip()


def output_path_for(input_path: str, target_lang: str, replace_original: bool) -> str:
    # ponytail: one fn, unifies subtitle_tab + pipeline_runner naming. No class.
    if replace_original:
        return input_path
    p = Path(input_path)
    lang_name = TARGET_LANGUAGES.get(target_lang, target_lang)
    return str(p.parent / f"{p.stem}_{lang_name}{p.suffix}")
