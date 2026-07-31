# -*- coding: utf-8 -*-
"""Reproduce/verify the 'returns original text' translation bug.

Fakes key their translations off the SOURCE text (like a real model),
so a single-line retry of a dropped line returns that line's real
translation rather than an id-derived artifact.
"""
import re

from translation.local_llm_translator import LocalLLMTranslator
from utils.srt_parser import Cue


def _cfg(**kw):
    c = {
        'api_url': 'http://localhost:8080/v1', 'model': 'test',
        'max_tokens': 4096, 'temperature': 0.3, 'batch_size': 10,
        'max_workers': 1, 'single_step': True,
    }
    c.update(kw)
    return c


def _sources(messages):
    # The prompt template embeds an example line "source: Source"; skip it so
    # the fake only sees the real batch sources.
    found = re.findall(r'^\s*source:\s*(.+)$', messages[1]['content'], re.MULTILINE)
    return [s for s in found if s != 'Source']


class DropOneClient:
    """Model that omits the 2nd line from a multi-line batch response."""
    def complete(self, messages, model, max_tokens, temperature):
        sources = _sources(messages)
        lines = []
        out_id = 0
        for pos, src in enumerate(sources):
            if len(sources) > 1 and pos == 1:
                continue  # drop the 2nd line of a real batch
            out_id += 1
            lines.append(f"- id: {out_id}")
            lines.append(f"  translation: TR[{src}]")
        return "\n".join(lines)


class ReorderClient:
    """Model returns lines in reversed order but keeps correct ids."""
    def complete(self, messages, model, max_tokens, temperature):
        sources = _sources(messages)
        pairs = list(enumerate(sources, start=1))
        lines = []
        for out_id, src in reversed(pairs):
            lines.append(f"- id: {out_id}")
            lines.append(f"  translation: TR[{src}]")
        return "\n".join(lines)


def _srt(n):
    return [Cue(line=i + 1, start_time=i * 1000, end_time=(i + 1) * 1000,
                text=f'src{i+1}') for i in range(n)]


print("=== Case 1: model drops the 2nd line ===")
t = LocalLLMTranslator(_cfg(), client=DropOneClient())
for e in t.translate_srt(_srt(4), 'zh-cn'):
    print(e.line, repr(e.text))

print("\n=== Case 2: model reorders ids ===")
t = LocalLLMTranslator(_cfg(), client=ReorderClient())
for e in t.translate_srt(_srt(4), 'zh-cn'):
    print(e.line, repr(e.text))
