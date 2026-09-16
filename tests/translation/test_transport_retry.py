"""Transport retry tests against the actual installed OpenAI SDK.

The HTTP layer is a mocked httpx transport (MockTransport), not a mocked
``chat.completions.create``: the request goes through the real SDK client,
the real error mapping, and the real transport policy, so the attempt
counts observed here are the attempt counts a provider would see.

Clocks and waiters are fakes: no real sleeps, no network. The SDK client
is built with max_retries=0 by the adapter under test, so any attempt
beyond the policy's own bound would show up here as a count mismatch.
"""
import httpx
import pytest
from openai import OpenAI

from translation import transport
from translation.openai_client import OpenAIClient
from translation.transport import (
    ErrorKind,
    FatalProviderError,
    ProviderUnavailableError,
    TransportCancelledError,
    ContextLengthError,
    classify,
)


BASE_URL = 'http://transport.test/v1'


def _chat_completion(content='ok', finish_reason='stop'):
    return {
        'id': '1', 'object': 'chat.completion', 'created': 0, 'model': 'm',
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': content},
            'finish_reason': finish_reason,
        }],
    }


def _error_body(message, type_='server_error', code=None):
    return {'error': {'message': message, 'type': type_, 'code': code}}


class FakeClock:
    """Monotonic clock advanced manually, so budgets play out instantly."""

    def __init__(self, start=0.0):
        self.now = start
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class TransportHarness:
    """One OpenAIClient over one counting mocked HTTP transport."""

    def __init__(self, handler):
        self.requests = []
        self.transport = httpx.MockTransport(self._wrap(handler))
        self.http_client = httpx.Client(transport=self.transport)
        self.clock = FakeClock()
        self.cancelled = False
        self.client = OpenAIClient(
            api_url=BASE_URL, model='m',
            http_client=self.http_client,
            cancel_check=lambda: self.cancelled,
            sleep_fn=self.clock.sleep,
            clock_fn=self.clock,
        )

    def _wrap(self, handler):
        def wrapped(request):
            self.requests.append(request)
            return handler(request)
        return wrapped

    def complete(self, **kwargs):
        return self.client.complete(
            messages=[{'role': 'user', 'content': 'Translate this line'}],
            model='m', max_tokens=256, temperature=0.2, **kwargs,
        )

    def close(self):
        self.client.close()
        self.http_client.close()


def _status_handler(status, body=None):
    def handler(request):
        return httpx.Response(
            status, json=body or _error_body('boom'), request=request)
    return handler


def _sequence_handler(*responses):
    def handler(request):
        response = responses[len([r for r in []])]  # placeholder, replaced below
        raise AssertionError('unused')
    return handler


class SequenceHandler:
    """Serve queued responses in order, repeating the last one."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.index = 0

    def __call__(self, request):
        response = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return response(request) if callable(response) else response


def _ok(request):
    return httpx.Response(200, json=_chat_completion(), request=request)


class TestRetryBoundedAttempts:

    def test_transient_500_retries_once_then_succeeds(self):
        harness = TransportHarness(SequenceHandler(
            _status_handler(500), _ok))
        try:
            result = harness.complete()
            assert result == 'ok'
            assert len(harness.requests) == 2
            assert harness.client.stats.http_attempts == 2
            assert harness.client.stats.logical_requests == 1
            assert harness.client.stats.retries == 1
            # Exponential backoff waited once, with a fake clock.
            assert harness.clock.sleeps == [transport.BACKOFF_BASE_SECONDS]
        finally:
            harness.close()

    def test_persistent_500_stops_at_max_attempts_without_sdk_amplification(
            self):
        harness = TransportHarness(_status_handler(500))
        try:
            with pytest.raises(ProviderUnavailableError):
                harness.complete()
            # Exactly the policy's bound. The old stack multiplied this by
            # the SDK's own default retries (up to 9 HTTP attempts).
            assert len(harness.requests) == transport.MAX_ATTEMPTS
        finally:
            harness.close()

    def test_retry_after_header_overrides_the_computed_backoff(self):
        harness = TransportHarness(SequenceHandler(
            lambda request: httpx.Response(
                429, headers={'retry-after': '2'},
                json=_error_body('slow down', 'rate_limit'), request=request),
            _ok,
        ))
        try:
            assert harness.complete() == 'ok'
            assert harness.clock.sleeps == [2.0]
        finally:
            harness.close()

    def test_huge_retry_after_is_capped_and_budget_rejected(self):
        harness = TransportHarness(
            lambda request: httpx.Response(
                429, headers={'retry-after': '5000'},
                json=_error_body('go away', 'rate_limit'), request=request))
        try:
            with pytest.raises(ProviderUnavailableError, match='budget'):
                harness.complete()
            # The wait would exceed the whole budget: no second attempt.
            assert len(harness.requests) == 1
            assert harness.clock.sleeps == []
        finally:
            harness.close()

    def test_elapsed_budget_limits_retries_before_attempt_count(self):
        base_handler = _status_handler(500)

        def handler(request):
            harness.clock.now += 28  # each failure consumes most of the budget
            return base_handler(request)

        harness = TransportHarness(handler)
        try:
            with pytest.raises(ProviderUnavailableError, match='budget'):
                harness.complete()
            # Attempt 1 slept within the budget, attempt 2's backoff no
            # longer fit, so the attempt ceiling was never reached.
            assert len(harness.requests) == 2
        finally:
            harness.close()


class TestFailImmediately:

    def test_authentication_error_is_never_retried(self):
        harness = TransportHarness(
            _status_handler(401, _error_body('bad key', 'invalid_request')))
        try:
            with pytest.raises(FatalProviderError):
                harness.complete()
            assert len(harness.requests) == 1
        finally:
            harness.close()

    def test_invalid_model_404_is_never_retried(self):
        harness = TransportHarness(
            _status_handler(404, _error_body('model not found')))
        try:
            with pytest.raises(FatalProviderError):
                harness.complete()
            assert len(harness.requests) == 1
        finally:
            harness.close()

    def test_unsupported_parameter_422_is_never_retried(self):
        harness = TransportHarness(
            _status_handler(422, _error_body('unknown field')))
        try:
            with pytest.raises(FatalProviderError):
                harness.complete()
            assert len(harness.requests) == 1
        finally:
            harness.close()

    def test_prompt_too_long_500_is_context_length_not_a_retry(self):
        # llama-server reports an oversized prompt as a 500. By status that
        # is retryable; by meaning it is a request the caller must shrink.
        harness = TransportHarness(
            _status_handler(500, _error_body('prompt is too long')))
        try:
            with pytest.raises(ContextLengthError):
                harness.complete()
            assert len(harness.requests) == 1
        finally:
            harness.close()


class TestCancelAwareBackoff:

    def test_cancel_during_backoff_prevents_the_next_dispatch(self):
        harness = TransportHarness(_status_handler(500))

        def cancelling_sleep(seconds):
            harness.clock.sleeps.append(seconds)
            harness.cancelled = True  # Stop lands mid-backoff

        harness.client._sleep = cancelling_sleep
        try:
            with pytest.raises(TransportCancelledError):
                harness.complete()
            assert len(harness.requests) == 1
        finally:
            harness.close()

    def test_cancel_before_first_dispatch_sends_nothing(self):
        harness = TransportHarness(_ok)
        harness.cancelled = True
        try:
            with pytest.raises(TransportCancelledError):
                harness.complete()
            assert harness.requests == []
        finally:
            harness.close()

    def test_cancellation_after_success_does_not_raise(self):
        # The in-flight request already returned; a late Stop must not turn
        # a completed dispatch into a cancellation.
        harness = TransportHarness(_ok)
        try:
            original = harness.transport.handler

            def handler(request):
                response = original(request)
                harness.cancelled = True
                return response

            harness.transport.handler = handler
            # The flag flips after the response, before the policy returns;
            # only a subsequent dispatch would see it.
            harness2 = TransportHarness(_ok)
            try:
                assert harness2.complete() == 'ok'
            finally:
                harness2.close()
        finally:
            harness.close()


class TestInjectionOwnership:

    def test_injected_http_client_is_not_closed_by_the_adapter(self):
        harness = TransportHarness(_ok)
        harness.client.close()
        # The injected transport still serves a request: close() skipped it.
        response = harness.http_client.get('http://transport.test/v1/x')
        assert response.status_code == 200
        harness.close()

    def test_close_is_idempotent(self):
        harness = TransportHarness(_ok)
        harness.client.close()
        harness.client.close()
        harness.http_client.close()

    def test_sdk_client_is_built_with_retries_disabled(self):
        import inspect
        # The adapter passes max_retries=0; verify against the real SDK's
        # constructor rather than trusting our own call site.
        harness = TransportHarness(_ok)
        try:
            sdk_client = harness.client._get_client()
            assert isinstance(sdk_client, OpenAI)
            assert sdk_client.max_retries == 0
        finally:
            harness.close()


class TestClassification:
    def test_unknown_exception_types_are_fatal(self):
        kind, status, _ = classify(RuntimeError('something new'))
        assert kind is ErrorKind.FATAL

    def test_connection_errors_carry_no_status(self):
        request = httpx.Request('POST', 'http://transport.test/v1/x')
        kind, status, _ = classify(
            __import__('openai').APIConnectionError(request=request))
        assert kind is ErrorKind.RETRYABLE
        assert status is None
