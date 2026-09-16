"""Regression tests for run-level guards: provider failure and cancellation.

These replay the failure shapes the hardening phase was scoped around:

- A provider that rejects or drops us must abort the run — never expand
  into N single-cue requests, each paying the full transport retry budget.
- The run's cancellation token must be honored before every HTTP dispatch
  and before a context-limit split issues its second request.
"""
from unittest.mock import Mock

import pytest

from config_helpers import build_translation_config
from llm_target import LLMTarget, api_url_for_port
from translation.local_llm_translator import (
    LocalLLMTranslator,
    TranslationCancelledError,
)
from translation.transport import (
    ContextLengthError,
    FatalProviderError,
    ProviderUnavailableError,
)
from utils.srt_parser import Cue


class NoneConfig:
    def get(self, key, default=None):
        return default


def _translator(client, **overrides):
    target = LLMTarget(mode='local', name='x',
                       api_url=api_url_for_port(8080), model='test')
    config = build_translation_config(NoneConfig(), target)
    config['max_workers'] = 1
    config.update(overrides)
    return LocalLLMTranslator(config, client=client)


def cues(count):
    return [Cue(line=i + 1, start_time=i * 1000, end_time=(i + 1) * 1000,
                text=f"Source {i + 1}") for i in range(count)]


class TestProviderFailureAbortsRun:

    def test_exhausted_transport_retries_abort_without_per_cue_expansion(self):
        """The old path turned one dead-provider batch into N line retries."""
        client = Mock()
        client.complete.side_effect = ProviderUnavailableError(
            'provider unreachable after 3 attempt(s)')
        translator = _translator(client)

        with pytest.raises(ProviderUnavailableError):
            translator.translate_srt(cues(3), 'zh-tw')

        # One batch request, zero per-cue recovery requests.
        assert client.complete.call_count == 1

    def test_auth_failure_fails_immediately_without_recovery(self):
        client = Mock()
        client.complete.side_effect = FatalProviderError(
            'AuthenticationError (status 401): bad key')
        translator = _translator(client)

        with pytest.raises(FatalProviderError):
            translator.translate_srt(cues(3), 'zh-tw')

        assert client.complete.call_count == 1

    def test_fatal_error_stops_dispatching_queued_batches(self):
        """With several batches pending, a fatal first batch drains the queue."""
        client = Mock()
        client.complete.side_effect = FatalProviderError(
            'NotFoundError (status 404): no such model')
        translator = _translator(client, batch_size=1)
        translator.max_workers = 1

        with pytest.raises(FatalProviderError):
            translator.translate_srt(cues(5), 'zh-tw')

        # Only the in-flight batch paid for the failure; the remaining four
        # were never dispatched.
        assert client.complete.call_count == 1

    def test_recovery_does_not_multiply_a_dead_provider(self):
        """Per-line recovery that meets a dead provider aborts, not loops."""
        responses = [
            '- id: 1\n  translation: 一',          # batch answer (line 2 missing)
            ProviderUnavailableError('provider unreachable'),
        ]
        client = Mock()
        client.complete.side_effect = list(responses)
        translator = _translator(client)

        with pytest.raises(ProviderUnavailableError):
            translator.translate_srt(cues(2), 'zh-tw')

        assert client.complete.call_count == 2  # batch + first (aborted) retry


class TestCancellationBoundaries:

    def test_cancel_before_any_dispatch_sends_nothing(self):
        client = Mock()
        translator = _translator(client)

        with pytest.raises(TranslationCancelledError):
            translator.translate_srt(cues(3), 'zh-tw',
                                     cancel_callback=lambda: True)

        client.complete.assert_not_called()

    def test_cancel_during_context_limit_split_sends_no_second_request(self):
        """A split triggered by a context error must respect the token.

        The context-limit handler recursively re-dispatches a smaller batch;
        without a token check at that boundary, a Stop landing between the
        failed request and the split still sent the split's request.
        """
        state = {'cancelled': False}
        client = Mock()

        def complete(**kwargs):
            state['cancelled'] = True  # Stop lands as this request fails
            raise ContextLengthError('prompt is too long (n_ctx)')

        client.complete.side_effect = complete
        translator = _translator(client)

        with pytest.raises(TranslationCancelledError):
            translator.translate_srt(
                cues(2), 'zh-tw', cancel_callback=lambda: state['cancelled'])

        assert client.complete.call_count == 1

    def test_cancel_between_story_analysis_and_translation(self):
        """Story mode checks the token after analysis, before batches."""
        from translation.llm_client import CompletionResult

        class AnalysisClient:
            def __init__(self):
                self.cancelled = False

            def complete_analysis(self, **kwargs):
                self.cancelled = True  # Stop lands as analysis returns
                return CompletionResult(
                    content='{"summary":"s","characters":[],"glossary":[],'
                            '"tone":[],"uncertainties":[]}',
                    finish_reason='stop')

            def complete(self, **kwargs):
                raise AssertionError('translation must not start')

        client = AnalysisClient()
        translator = _translator(client, context_mode='story',
                                 context_size=8192)

        with pytest.raises(TranslationCancelledError):
            translator.translate_srt(
                cues(1), 'zh-tw', cancel_callback=lambda: client.cancelled)

    def test_story_analyzer_cancel_between_requests(self):
        from translation.story_context import (
            StoryContextAnalyzer,
            StoryContextCancelled,
        )
        client = Mock()
        client.complete_analysis.side_effect = Mock(
            content='{"summary":"s","characters":[],"glossary":[],'
                    '"tone":[],"uncertainties":[]}',
            finish_reason='stop')
        state = {'cancelled': False}
        analyzer = StoryContextAnalyzer(
            client=client, model='m', temperature=0.2, max_tokens=768,
            context_size=8192,
            cancel_check=lambda: state['cancelled'])

        state['cancelled'] = True
        with pytest.raises(StoryContextCancelled):
            analyzer.create(cues(2), 'zh-tw')
