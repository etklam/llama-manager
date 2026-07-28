# -*- coding: utf-8 -*-
"""
TDD tests for translation speed optimizations.

Plan A: Concurrent batch processing via ThreadPoolExecutor
Plan B: Single-step translation mode (skip step1 literal translation)
Plan C: Larger default batch size
"""
import re
import threading
import time

import pytest

from translation.local_llm_translator import (
    LocalLLMTranslator,
    DEFAULT_BATCH_SIZE,
)


def _full_config(**overrides):
    # ponytail: LocalLLMTranslator trusts the dict (mirrors config_helpers).
    cfg = {
        'api_url': 'http://localhost:8080/v1',
        'model': 'test',
        'max_tokens': 4096,
        'temperature': 0.3,
        'batch_size': DEFAULT_BATCH_SIZE,
        'max_workers': 3,
        'single_step': False,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Fake LLM client with configurable delay and call tracking
# ---------------------------------------------------------------------------

class FakeLLMClient:
    """Thread-safe fake LLM client that tracks calls and simulates latency."""

    def __init__(self, response_text="- id: 1\n  step1: 直译\n  step2: 意译",
                 delay: float = 0.0):
        self.response_text = response_text
        self.delay = delay
        self._lock = threading.Lock()
        self.call_count = 0
        self.call_timestamps: list[float] = []
        self.messages_list: list[list[dict]] = []

    def complete(self, messages, model, max_tokens, temperature) -> str:
        with self._lock:
            self.call_count += 1
            self.call_timestamps.append(time.time())
            self.messages_list.append(messages)

        if self.delay:
            time.sleep(self.delay)

        return self.response_text


class SingleStepFakeLLMClient(FakeLLMClient):
    """Fake client that returns single-step YAML responses."""

    def __init__(self, response_text="- id: 1\n  translation: 意译", **kw):
        super().__init__(response_text=response_text, **kw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_srt_data(n: int) -> list[dict]:
    """Create n fake SRT entries."""
    return [
        {'text': f'Line {i+1} text here', 'time': f'00:00:0{i},000 --> 00:00:0{i+1},000', 'line': i + 1}
        for i in range(n)
    ]


def _batch_yaml_response(batch: list[dict], *, single_step: bool = False) -> str:
    """Generate a fake YAML response matching a batch of SRT entries."""
    lines = []
    for j, entry in enumerate(batch):
        lines.append(f"- id: {j + 1}")
        if single_step:
            lines.append(f"  translation: 翻译{entry['line']}")
        else:
            lines.append(f"  step1: 直译{entry['line']}")
            lines.append(f"  step2: 意译{entry['line']}")
    return "\n".join(lines)


# ===================================================================
# Plan C: Larger default batch size
# ===================================================================

class TestLargerDefaultBatchSize:
    """Plan C: Verify default batch size is increased."""

    def test_default_batch_size_is_at_least_10(self):
        """DEFAULT_BATCH_SIZE should be >= 10 for fewer API calls."""
        assert DEFAULT_BATCH_SIZE >= 10, (
            f"DEFAULT_BATCH_SIZE is {DEFAULT_BATCH_SIZE}, expected >= 10"
        )

    def test_translate_srt_uses_configured_batch_size(self):
        """translate_srt should respect batch_size from config."""
        fake = FakeLLMClient()

        # Echo one YAML entry per requested id. A fixed single-line response
        # would leave every line past the first "missing", triggering per-line
        # retries (the dropped-line recovery path) and inflating call_count —
        # which measures batches, not lines. A realistic model answers every id.
        def _complete(messages, model, max_tokens, temperature):
            with fake._lock:
                fake.call_count += 1
            ids = re.findall(r'id:\s*(\d+)', messages[1]['content'])
            lines = []
            for idx_str in ids:
                lines.append(f"- id: {idx_str}")
                lines.append(f"  step1: 直译{idx_str}")
                lines.append(f"  step2: 意译{idx_str}")
            return "\n".join(lines)

        fake.complete = _complete
        config = _full_config(batch_size=3)
        translator = LocalLLMTranslator(config, client=fake)

        # 10 items, batch_size=3 => 4 batches (3+3+3+1)
        srt_data = _make_srt_data(10)
        translator.translate_srt(srt_data, 'zh-cn')
        assert fake.call_count == 4

    def test_translate_srt_uses_default_batch_size_when_not_configured(self):
        """When batch_size is not in config, DEFAULT_BATCH_SIZE is used."""
        fake = FakeLLMClient()
        config = _full_config()
        del config['batch_size']  # ponytail: batch_size is read lazily by translate_srt
        translator = LocalLLMTranslator(config, client=fake)

        # DEFAULT_BATCH_SIZE items should produce exactly 1 call
        srt_data = _make_srt_data(DEFAULT_BATCH_SIZE)
        translator.translate_srt(srt_data, 'zh-cn')
        assert fake.call_count == 1


# ===================================================================
# Plan B: Single-step translation mode
# ===================================================================

class TestSingleStepTranslationMode:
    """Plan B: Verify single-step (fast) translation mode."""

    # --- Prompt generation tests ---

    def test_single_step_prompt_does_not_request_step1(self):
        """In single-step mode, prompt should NOT mention step1/直译."""
        fake = FakeLLMClient(response_text="- id: 1\n  translation: 结果")
        config = _full_config(single_step=True)
        translator = LocalLLMTranslator(config, client=fake)

        translator.translate("Hello", target_language="zh-cn")

        user_msg = fake.messages_list[0][1]['content']
        assert 'step1' not in user_msg
        assert '直译' not in user_msg

    def test_single_step_prompt_requests_translation_field(self):
        """In single-step mode, prompt should request 'translation' field."""
        fake = FakeLLMClient(response_text="- id: 1\n  translation: 结果")
        config = _full_config(single_step=True)
        translator = LocalLLMTranslator(config, client=fake)

        translator.translate("Hello", target_language="zh-cn")

        user_msg = fake.messages_list[0][1]['content']
        assert 'translation' in user_msg or '意译' in user_msg

    def test_two_step_prompt_requests_both_steps_by_default(self):
        """In two-step mode (default), prompt should request step1 AND step2."""
        fake = FakeLLMClient()
        config = _full_config()  # single_step defaults to False
        translator = LocalLLMTranslator(config, client=fake)

        translator.translate("Hello", target_language="zh-cn")

        user_msg = fake.messages_list[0][1]['content']
        assert 'step1' in user_msg
        assert 'step2' in user_msg

    # --- Response parsing tests ---

    def test_parse_single_step_yaml_result(self):
        """_parse_yaml_result should handle 'translation' field as fallback."""
        raw = "- id: 1\n  translation: 你好世界"
        parsed = LocalLLMTranslator._parse_yaml_result(raw)
        assert len(parsed) == 1
        assert parsed[0]['step2'] == '你好世界'

    def test_parse_single_step_batch_yaml_result(self):
        """Parse multiple single-step YAML items."""
        raw = (
            "- id: 1\n  translation: 你好\n"
            "- id: 2\n  translation: 世界"
        )
        parsed = LocalLLMTranslator._parse_yaml_result(raw)
        assert len(parsed) == 2
        assert parsed[0]['step2'] == '你好'
        assert parsed[1]['step2'] == '世界'

    def test_parse_strips_leaked_translation_marker(self):
        """A value that bleeds into the next item's marker must be truncated.

        Reproduces the observed pollution where the model omits ids and a
        value swallows the following "- translation:" marker, leaving raw
        tokens like "- translation: ..." in the subtitle text.
        """
        raw = (
            "- translation: やば啊！快被發現躲起來！\n"
            "- translation: 水快跑！"
        )
        parsed = LocalLLMTranslator._parse_yaml_result(raw)
        assert len(parsed) == 2
        assert parsed[0]['step2'] == 'やば啊！快被發現躲起來！'
        assert parsed[1]['step2'] == '水快跑！'
        for entry in parsed:
            assert 'translation:' not in entry['step2']
            assert not entry['step2'].startswith('-')

    def test_parse_strips_leading_marker_and_quotes(self):
        """Leftover leading markers and wrapping quotes are stripped."""
        raw = '- id: 1\n  translation: "- translation: 你好世界"'
        parsed = LocalLLMTranslator._parse_yaml_result(raw)
        assert len(parsed) == 1
        assert parsed[0]['step2'] == '你好世界'

    def test_parse_two_step_value_does_not_swallow_next_field(self):
        """step1 must stop at step2 rather than absorbing it."""
        raw = "- id: 1\n  step1: 直译\n  step2: 意译"
        parsed = LocalLLMTranslator._parse_yaml_result(raw)
        assert len(parsed) == 1
        assert parsed[0]['step1'] == '直译'
        assert parsed[0]['step2'] == '意译'

    # --- End-to-end translate test ---

    def test_translate_single_step_returns_translation(self):
        """translate() in single-step mode should return the translation field."""
        fake = FakeLLMClient(response_text="- id: 1\n  translation: 你好")
        config = _full_config(single_step=True)
        translator = LocalLLMTranslator(config, client=fake)

        result = translator.translate("Hello", target_language="zh-cn")
        assert result == '你好'

    def test_translate_srt_single_step_mode(self):
        """translate_srt in single-step mode should work correctly."""
        fake = FakeLLMClient()
        # Use >3 lines to trigger batch mode instead of simple mode
        fake.response_text = (
            "- id: 1\n  translation: 翻译1\n"
            "- id: 2\n  translation: 翻译2\n"
            "- id: 3\n  translation: 翻译3\n"
            "- id: 4\n  translation: 翻译4"
        )

        config = _full_config(single_step=True, batch_size=10)
        translator = LocalLLMTranslator(config, client=fake)

        srt_data = _make_srt_data(4)
        result = translator.translate_srt(srt_data, 'zh-cn')
        assert len(result) == 4
        assert result[0]['text'] == '翻译1'
        assert result[1]['text'] == '翻译2'

    # --- Batch prompt tests ---

    def test_batch_prompt_single_step_does_not_mention_step1(self):
        """Batch prompt in single-step mode should not mention step1."""
        fake = FakeLLMClient()
        fake.response_text = "- id: 1\n  translation: 结果"
        config = _full_config(single_step=True, batch_size=10)
        translator = LocalLLMTranslator(config, client=fake)

        srt_data = _make_srt_data(5)
        translator.translate_srt(srt_data, 'zh-cn')

        user_msg = fake.messages_list[0][1]['content']
        assert 'step1' not in user_msg
        assert '直译' not in user_msg


# ===================================================================
# Plan A: Concurrent batch processing
# ===================================================================

class TestConcurrentBatchProcessing:
    """Plan A: Verify batches are processed concurrently via ThreadPoolExecutor."""

    def test_concurrent_batches_faster_than_sequential(self):
        """
        With 4 batches and 0.2s delay each, concurrent (3 workers) should be
        significantly faster than sequential (4 * 0.2s = 0.8s).
        """
        fake = FakeLLMClient(delay=0.2)

        def _complete(messages, model, max_tokens, temperature):
            time.sleep(0.2)  # Retain the delay
            return (
                f"- id: 1\n  step1: 直译\n  step2: 意译\n"
                f"- id: 2\n  step1: 直译2\n  step2: 意译2"
            )

        fake.complete = _complete

        config = _full_config(batch_size=2, max_workers=3)
        translator = LocalLLMTranslator(config, client=fake)

        srt_data = _make_srt_data(8)  # 4 batches of 2

        start = time.time()
        result = translator.translate_srt(srt_data, 'zh-cn')
        elapsed = time.time() - start

        # Sequential would be ~0.8s. Concurrent with 3 workers should be < 0.5s
        assert elapsed < 0.6, f"Expected concurrent speedup, took {elapsed:.2f}s"
        assert len(result) == 8

    def test_max_workers_default_is_at_least_2(self):
        """Default max_workers should be >= 2 for concurrency."""
        from translation.local_llm_translator import DEFAULT_MAX_WORKERS
        assert DEFAULT_MAX_WORKERS >= 2

    def test_max_workers_from_config(self):
        """max_workers should be read from config."""
        fake = FakeLLMClient()
        config = _full_config(max_workers=5)
        translator = LocalLLMTranslator(config, client=fake)
        assert translator.max_workers == 5

    def test_max_workers_defaults_when_not_configured(self):
        """max_workers should use DEFAULT_MAX_WORKERS when not in config."""
        fake = FakeLLMClient()
        config = _full_config()
        del config['max_workers']  # ponytail: max_workers required by __init__ now
        # Restore via build_translation_config semantics — use DEFAULT explicitly
        from translation.local_llm_translator import DEFAULT_MAX_WORKERS
        config['max_workers'] = DEFAULT_MAX_WORKERS
        translator = LocalLLMTranslator(config, client=fake)
        assert translator.max_workers == DEFAULT_MAX_WORKERS

    def test_concurrent_results_are_in_correct_order(self):
        """Concurrent processing must preserve original line order."""
        fake = FakeLLMClient()

        def _complete(messages, model, max_tokens, temperature):
            # Extract the source texts from the YAML input to return matching translations
            user_msg = messages[1]['content']
            import re
            ids = re.findall(r'id:\s*(\d+)', user_msg)
            lines = []
            for idx_str in ids:
                i = int(idx_str)
                lines.append(f"- id: {i}")
                lines.append(f"  step1: 直译{i}")
                lines.append(f"  step2: 意译{i}")
            return "\n".join(lines)

        fake.complete = _complete

        config = _full_config(batch_size=3, max_workers=2)
        translator = LocalLLMTranslator(config, client=fake)

        srt_data = _make_srt_data(9)  # 3 batches of 3
        result = translator.translate_srt(srt_data, 'zh-cn')

        # Verify order is preserved
        assert len(result) == 9
        for i, entry in enumerate(result):
            expected_line = i + 1
            assert entry['line'] == expected_line, (
                f"Expected line {expected_line}, got {entry['line']}"
            )
            assert f'意译' in entry['text']

    def test_sequential_fallback_on_batch_failure(self):
        """If a concurrent batch fails, fall back to individual translation."""
        fake = FakeLLMClient()
        call_count = 0

        def _complete(messages, model, max_tokens, temperature):
            nonlocal call_count
            call_count += 1
            # Fail on batch calls, succeed on individual calls
            if 'id: 2' in messages[1]['content']:
                # This is a batch with >1 items
                raise RuntimeError("Simulated batch failure")
            return "- id: 1\n  step1: 直译\n  step2: 意译"

        fake.complete = _complete

        config = _full_config(batch_size=2, max_workers=2)
        translator = LocalLLMTranslator(config, client=fake)

        srt_data = _make_srt_data(4)  # 2 batches of 2
        result = translator.translate_srt(srt_data, 'zh-cn')

        # Should still produce results (via fallback)
        assert len(result) == 4

    def test_max_workers_1_means_sequential(self):
        """max_workers=1 should process batches sequentially (no concurrency)."""
        def _complete(messages, model, max_tokens, temperature):
            time.sleep(0.1)
            return "- id: 1\n  step1: 直译\n  step2: 意译\n- id: 2\n  step1: 直译2\n  step2: 意译2"

        fake = FakeLLMClient()
        fake.complete = _complete

        config = _full_config(batch_size=2, max_workers=1)
        translator = LocalLLMTranslator(config, client=fake)

        srt_data = _make_srt_data(4)
        start = time.time()
        translator.translate_srt(srt_data, 'zh-cn')
        elapsed = time.time() - start

        # Sequential: 2 batches * 0.1s = ~0.2s minimum
        assert elapsed >= 0.15, f"Expected sequential timing, took {elapsed:.2f}s"


# ===================================================================
# Integration: A + B + C combined
# ===================================================================

class TestCombinedOptimizations:
    """Test that all three optimizations work together."""

    def test_concurrent_single_step_large_batch(self):
        """
        Large batch (C=15) + single-step (B) + concurrent (A, 3 workers)
        should be fast and correct.
        """
        fake = FakeLLMClient(delay=0.05)

        def _complete(messages, model, max_tokens, temperature):
            import re
            ids = re.findall(r'id:\s*(\d+)', messages[1]['content'])
            lines = []
            for idx_str in ids:
                i = int(idx_str)
                lines.append(f"- id: {i}")
                lines.append(f"  translation: 翻译{i}")
            return "\n".join(lines)

        fake.complete = _complete

        config = _full_config(
            batch_size=15,
            max_workers=3,
            single_step=True,
        )
        translator = LocalLLMTranslator(config, client=fake)

        srt_data = _make_srt_data(30)  # 2 batches of 15
        start = time.time()
        result = translator.translate_srt(srt_data, 'zh-cn')
        elapsed = time.time() - start

        assert len(result) == 30
        # All results should have translated text
        for entry in result:
            assert '翻译' in entry['text']
        # With concurrency, 2 batches of 0.05s should complete in ~0.1s
        assert elapsed < 0.3, f"Expected fast concurrent execution, took {elapsed:.2f}s"

    def test_concurrent_single_step_preserves_timestamps(self):
        """Timestamps must be preserved with concurrent single-step mode."""
        fake = FakeLLMClient()

        def _complete(messages, model, max_tokens, temperature):
            import re
            ids = re.findall(r'id:\s*(\d+)', messages[1]['content'])
            lines = []
            for idx_str in ids:
                lines.append(f"- id: {idx_str}")
                lines.append(f"  translation: 结果{idx_str}")
            return "\n".join(lines)

        fake.complete = _complete

        config = _full_config(
            batch_size=5,
            max_workers=2,
            single_step=True,
        )
        translator = LocalLLMTranslator(config, client=fake)

        srt_data = _make_srt_data(10)
        result = translator.translate_srt(srt_data, 'zh-cn')

        for i, entry in enumerate(result):
            assert entry['time'] == srt_data[i]['time'], (
                f"Timestamp mismatch at line {i+1}"
            )


# ===================================================================
# P1-3: Dynamic max_tokens sized to the batch
# ===================================================================

class TestDynamicMaxTokens:
    """P1-3: max_tokens is sized to the request, capped by the user ceiling."""

    def test_batch_request_uses_dynamic_max_tokens_not_ceiling(self):
        """A batch call should request far fewer tokens than the fixed ceiling."""
        captured = {}

        def _complete(messages, model, max_tokens, temperature):
            captured['max_tokens'] = max_tokens
            import re
            ids = re.findall(r'id:\s*(\d+)', messages[1]['content'])
            lines = []
            for idx_str in ids:
                lines.append(f"- id: {idx_str}")
                lines.append(f"  translation: 结果{idx_str}")
            return "\n".join(lines)

        fake = FakeLLMClient()
        fake.complete = _complete

        # Large ceiling, small batch: dynamic value must stay well under it.
        config = _full_config(
            max_tokens=16384, batch_size=5, max_workers=1, single_step=True
        )
        translator = LocalLLMTranslator(config, client=fake)

        translator.translate_srt(_make_srt_data(5), 'zh-cn')

        assert captured['max_tokens'] < 16384
        # 5 lines x 160 tokens (single-step) = 800, floored at 512 -> 800.
        assert captured['max_tokens'] == 800

    def test_dynamic_max_tokens_capped_by_user_ceiling(self):
        """The dynamic budget never exceeds the user-configured max_tokens."""
        config = _full_config(max_tokens=1000, single_step=True)
        translator = LocalLLMTranslator(config, client=FakeLLMClient())

        # 100 lines x 160 = 16000, but capped at the 1000 ceiling.
        assert translator._dynamic_max_tokens(100) == 1000

    def test_dynamic_max_tokens_floored_for_tiny_batches(self):
        """Tiny batches keep a minimum headroom rather than a razor-thin cap."""
        config = _full_config(max_tokens=16384, single_step=True)
        translator = LocalLLMTranslator(config, client=FakeLLMClient())

        # 1 line x 160 = 160, floored to MIN_DYNAMIC_MAX_TOKENS (512).
        assert translator._dynamic_max_tokens(1) == 512

    def test_two_step_budgets_double_of_single_step(self):
        """Two-step emits step1+step2, so it needs roughly double the budget."""
        two_step = LocalLLMTranslator(
            _full_config(max_tokens=16384, single_step=False),
            client=FakeLLMClient(),
        )
        single = LocalLLMTranslator(
            _full_config(max_tokens=16384, single_step=True),
            client=FakeLLMClient(),
        )

        assert two_step._dynamic_max_tokens(20) == 2 * single._dynamic_max_tokens(20)


class TestRepetitionCollapsing:
    """translate_srt collapses pathological repetition before sending to the LLM."""

    def test_translate_srt_collapses_repeated_source(self):
        """A cue of 100x "あ、" is sent to the client as the collapsed form."""
        fake = FakeLLMClient()

        def _complete(messages, model, max_tokens, temperature):
            import re
            ids = re.findall(r'id:\s*(\d+)', messages[1]['content'])
            fake.messages_list.append(messages)
            lines = []
            for idx_str in ids:
                lines.append(f"- id: {idx_str}")
                lines.append(f"  translation: 结果{idx_str}")
            return "\n".join(lines)

        fake.complete = _complete

        config = _full_config(batch_size=10, max_workers=1, single_step=True)
        translator = LocalLLMTranslator(config, client=fake)

        srt_data = _make_srt_data(4)
        srt_data[0]['text'] = 'あ、' * 100
        translator.translate_srt(srt_data, 'zh-cn')

        # The oversized run must never reach the model verbatim.
        sent = "".join(m[1]['content'] for m in fake.messages_list)
        assert 'あ、' * 100 not in sent
        assert 'あ...' in sent

    def test_translate_srt_does_not_mutate_caller_data(self):
        """Normalization copies entries; the caller's dicts stay unchanged."""
        fake = FakeLLMClient()
        config = _full_config(batch_size=10, max_workers=1, single_step=True)
        translator = LocalLLMTranslator(config, client=fake)

        srt_data = _make_srt_data(4)
        original = 'あ、' * 100
        srt_data[0]['text'] = original
        translator.translate_srt(srt_data, 'zh-cn')

        assert srt_data[0]['text'] == original


class TestBatchAlignmentByModelId:
    """Batch results align to sources by the model's echoed id, not by position.

    Local models intermittently reorder, merge, or drop entries in a YAML batch
    response. Reconstructing positionally shifted every following translation
    onto the wrong subtitle and ran the tail off the end of the parsed list,
    where it fell back to the untranslated source — the "translate returned the
    original text" symptom.
    """

    @staticmethod
    def _batch_sources(messages):
        """Sources from the batch YAML, excluding the prompt's example line.

        The prompt template contains a literal "source: Source" example, which
        a naive regex would pick up as a phantom first line.
        """
        content = messages[1]['content']
        body = content.split('翻譯:' if '翻譯:' in content else '翻译:')[-1]
        return re.findall(r'^\s*source:\s*(.+)$', body, re.MULTILINE)

    def test_reordered_response_maps_each_translation_to_its_own_source(self):
        """Ids returned out of order must still land on the right subtitle."""
        fake = FakeLLMClient()

        def _complete(messages, model, max_tokens, temperature):
            with fake._lock:
                fake.call_count += 1
            sources = self._batch_sources(messages)
            pairs = list(enumerate(sources, start=1))
            lines = []
            for out_id, src in reversed(pairs):  # model answers in reverse
                lines.append(f"- id: {out_id}")
                lines.append(f"  translation: TR[{src}]")
            return "\n".join(lines)

        fake.complete = _complete
        translator = LocalLLMTranslator(
            _full_config(batch_size=10, max_workers=1, single_step=True),
            client=fake,
        )

        srt_data = _make_srt_data(4)
        for i, entry in enumerate(srt_data):
            entry['text'] = f'src{i + 1}'

        result = translator.translate_srt(srt_data, 'zh-cn')

        assert [entry['text'] for entry in result] == [
            'TR[src1]', 'TR[src2]', 'TR[src3]', 'TR[src4]'
        ]

    def test_dropped_line_is_retried_instead_of_returning_original(self):
        """A line absent from the batch response gets its own retry call."""
        fake = FakeLLMClient()

        def _complete(messages, model, max_tokens, temperature):
            with fake._lock:
                fake.call_count += 1
            sources = self._batch_sources(messages)
            lines = []
            for out_id, src in enumerate(sources, start=1):
                # Omit id 3 from the multi-line batch, but answer a
                # single-line retry normally.
                if len(sources) > 1 and out_id == 3:
                    continue
                lines.append(f"- id: {out_id}")
                lines.append(f"  translation: TR[{src}]")
            return "\n".join(lines)

        fake.complete = _complete
        translator = LocalLLMTranslator(
            _full_config(batch_size=10, max_workers=1, single_step=True),
            client=fake,
        )

        srt_data = _make_srt_data(4)
        for i, entry in enumerate(srt_data):
            entry['text'] = f'src{i + 1}'

        result = translator.translate_srt(srt_data, 'zh-cn')

        # Every line is translated; the dropped one no longer leaks its source.
        assert [entry['text'] for entry in result] == [
            'TR[src1]', 'TR[src2]', 'TR[src3]', 'TR[src4]'
        ]
        assert all(
            not entry['text'].startswith('src') for entry in result
        )
        # One batch call plus exactly one targeted retry.
        assert fake.call_count == 2

    def test_well_behaved_response_incurs_no_retry_calls(self):
        """When the model answers every id, no extra per-line calls are made."""
        fake = FakeLLMClient()

        def _complete(messages, model, max_tokens, temperature):
            with fake._lock:
                fake.call_count += 1
            sources = self._batch_sources(messages)
            lines = []
            for out_id, src in enumerate(sources, start=1):
                lines.append(f"- id: {out_id}")
                lines.append(f"  translation: TR[{src}]")
            return "\n".join(lines)

        fake.complete = _complete
        translator = LocalLLMTranslator(
            _full_config(batch_size=10, max_workers=1, single_step=True),
            client=fake,
        )

        srt_data = _make_srt_data(6)
        for i, entry in enumerate(srt_data):
            entry['text'] = f'src{i + 1}'

        result = translator.translate_srt(srt_data, 'zh-cn')

        assert fake.call_count == 1
        assert len(result) == 6

    def test_duplicated_id_does_not_clobber_a_distinct_line(self):
        """A repeated id keeps the first value; the other line is retried."""
        fake = FakeLLMClient()

        def _complete(messages, model, max_tokens, temperature):
            with fake._lock:
                fake.call_count += 1
            sources = self._batch_sources(messages)
            if len(sources) > 1:
                # Model emits id 1 twice and never mentions id 2.
                return (
                    f"- id: 1\n  translation: TR[{sources[0]}]\n"
                    f"- id: 1\n  translation: BOGUS"
                )
            return f"- id: 1\n  translation: TR[{sources[0]}]"

        fake.complete = _complete
        translator = LocalLLMTranslator(
            _full_config(batch_size=10, max_workers=1, single_step=True),
            client=fake,
        )

        srt_data = _make_srt_data(2)
        srt_data[0]['text'] = 'src1'
        srt_data[1]['text'] = 'src2'

        result = translator.translate_srt(srt_data, 'zh-cn')

        assert result[0]['text'] == 'TR[src1]'
        # The second line must not inherit the duplicate's value.
        assert result[1]['text'] == 'TR[src2]'


class TestDuplicateSubtitleDeduplication:
    """Identical subtitle cues are translated once and fanned back out."""

    def test_repeated_sentences_only_call_llm_once_per_unique_text(self):
        fake = FakeLLMClient()

        def _complete(messages, model, max_tokens, temperature):
            import re
            fake.call_count += 1
            fake.messages_list.append(messages)
            sources = re.findall(r'^\s*source:\s*(.+)$', messages[1]['content'], re.MULTILINE)
            source = sources[-1]
            return f"- id: 1\n  translation: TR:{source}"

        fake.complete = _complete
        translator = LocalLLMTranslator(
            _full_config(batch_size=10, max_workers=1, single_step=True),
            client=fake,
        )

        texts = ["Hello", "Hello", "Bye", "Hello", "Bye", "Unique"]
        srt_data = _make_srt_data(len(texts))
        for entry, text in zip(srt_data, texts):
            entry['text'] = text

        result = translator.translate_srt(srt_data, 'zh-cn')

        assert fake.call_count == 3
        assert [entry['text'] for entry in result] == [
            "TR:Hello", "TR:Hello", "TR:Bye",
            "TR:Hello", "TR:Bye", "TR:Unique",
        ]

    def test_deduplication_preserves_each_timestamp_and_line_number(self):
        fake = SingleStepFakeLLMClient(
            response_text="- id: 1\n  translation: 同一句翻譯"
        )
        translator = LocalLLMTranslator(
            _full_config(batch_size=10, max_workers=1, single_step=True),
            client=fake,
        )

        srt_data = _make_srt_data(8)
        for entry in srt_data:
            entry['text'] = "Same sentence"

        result = translator.translate_srt(srt_data, 'zh-cn')

        assert fake.call_count == 1
        assert len(result) == 8
        assert all(entry['text'] == "同一句翻譯" for entry in result)
        assert [entry['time'] for entry in result] == [
            entry['time'] for entry in srt_data
        ]
        assert [entry['line'] for entry in result] == [
            entry['line'] for entry in srt_data
        ]

    def test_whitespace_equivalent_sentences_share_translation(self):
        fake = SingleStepFakeLLMClient(
            response_text="- id: 1\n  translation: 相同翻譯"
        )
        translator = LocalLLMTranslator(
            _full_config(batch_size=10, max_workers=1, single_step=True),
            client=fake,
        )

        srt_data = _make_srt_data(4)
        srt_data[0]['text'] = "Hello world"
        srt_data[1]['text'] = "Hello\nworld"
        srt_data[2]['text'] = "Hello   world"
        srt_data[3]['text'] = "Hello world"

        result = translator.translate_srt(srt_data, 'zh-cn')

        assert fake.call_count == 1
        assert all(entry['text'] == "相同翻譯" for entry in result)


def _long_text(chars: int) -> str:
    """Build long text whose units all differ, so collapse_repeats leaves it."""
    parts = []
    i = 0
    while len(" ".join(parts)) <= chars:
        parts.append(f"sentence{i}")
        i += 1
    return " ".join(parts)


def _echoing_client() -> FakeLLMClient:
    """Client that answers every id present in the request (single-step)."""
    fake = FakeLLMClient()

    def _complete(messages, model, max_tokens, temperature):
        fake.call_count += 1
        fake.messages_list.append(messages)
        ids = re.findall(r'id:\s*(\d+)', messages[1]['content'])
        lines = []
        for idx_str in ids:
            lines.append(f"- id: {idx_str}")
            lines.append(f"  translation: 翻譯{idx_str}")
        return "\n".join(lines)

    fake.complete = _complete
    return fake


class TestLongSourceSoftGuard:
    """The translate path warns about over-long cues but never drops them.

    Translation is tuned for subtitle-sized context (see the translation server
    preset). Long-form input belongs on the chat endpoint, so it is flagged --
    softly, because refusing or truncating a cue would silently lose subtitle
    content the user asked to translate.
    """

    def _translate(self, srt_data):
        logs = []
        fake = _echoing_client()
        translator = LocalLLMTranslator(
            _full_config(batch_size=10, max_workers=1, single_step=True),
            client=fake,
        )
        result = translator.translate_srt(
            srt_data, 'zh-cn', log_callback=lambda lv, m: logs.append((lv, m))
        )
        return result, logs, fake

    def _warnings(self, logs):
        return [m for lv, m in logs if lv == "WARNING" and "exceed" in m]

    def test_long_line_is_warned_about_and_names_the_line(self):
        srt_data = _make_srt_data(4)
        srt_data[2]['text'] = _long_text(600)

        _, logs, _ = self._translate(srt_data)

        warnings = self._warnings(logs)
        assert len(warnings) == 1
        assert "L3" in warnings[0]
        assert "chat" in warnings[0].lower()

    def test_long_line_is_still_translated_not_rejected_or_truncated(self):
        long_text = _long_text(600)
        srt_data = _make_srt_data(4)
        srt_data[2]['text'] = long_text

        result, _, fake = self._translate(srt_data)

        # Soft: the cue reached the model in full and came back translated.
        assert len(result) == 4
        assert result[2]['text'] == "翻譯3"
        assert long_text in "".join(m[1]['content'] for m in fake.messages_list)

    def test_subtitle_sized_lines_produce_no_warning(self):
        result, logs, _ = self._translate(_make_srt_data(4))

        assert self._warnings(logs) == []
        assert len(result) == 4

    def test_many_long_lines_are_summarized_in_one_warning(self):
        srt_data = _make_srt_data(8)
        for entry in srt_data:
            entry['text'] = _long_text(600) + f" tail{entry['line']}"

        _, logs, _ = self._translate(srt_data)

        warnings = self._warnings(logs)
        assert len(warnings) == 1
        assert "8 line(s)" in warnings[0]
        assert "+3 more" in warnings[0]
