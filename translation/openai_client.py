"""
OpenAI Client Adapter - Production adapter wrapping OpenAI SDK.

This module provides the production implementation of the LLMClient port,
wrapping the OpenAI SDK. The SDK client is built with ``max_retries=0``:
transport retries have exactly one owner, the centralized policy in
``translation.transport``.
"""
import logging
import time
from typing import Callable, List, Dict, Optional

import httpx
from openai import OpenAI, LengthFinishReasonError

from translation.llm_client import CompletionResult
from translation import transport
from translation.transport import (
    ProviderUnavailableError,
    TransportCancelledError,
    TransportStats,
    request_with_retry,
)

logger = logging.getLogger(__name__)

# Kept for callers that report the retry ceiling; the policy owns the value.
RETRY_NUMS = transport.MAX_ATTEMPTS


class OpenAIClient:
    """
    Production adapter: wraps the OpenAI SDK behind the LLMClient port.

    Transport retries, error classification, the elapsed budget and
    cancel-aware backoff all live in translation.transport; this adapter
    only builds the SDK client (with SDK retries disabled) and normalizes
    responses.
    """

    def __init__(
        self,
        api_url: str,
        model: str,
        api_key: str = '',
        proxy: str = None,
        timeout: float = 180.0,
        http_client: Optional[httpx.Client] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
        clock_fn: Optional[Callable[[], float]] = None,
    ):
        """
        Initialize the OpenAI client adapter.

        Args:
            api_url: API endpoint URL
            model: Model name (stored for reference, passed to complete())
            api_key: API key if required (default: empty string)
            proxy: Proxy URL for HTTP client (default: None)
            timeout: Request timeout in seconds (default: 180.0)
            http_client: Pre-built httpx client (tests, shared transports).
                When given, this adapter never closes it; proxy is ignored.
            cancel_check: Run cancellation token checked before every HTTP
                dispatch and transport retry (may also be set as an
                attribute between runs).
            sleep_fn: Backoff waiter (injectable fake clock for tests).
            clock_fn: Elapsed-time clock (injectable fake for tests).
        """
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.proxy = proxy
        self.timeout = timeout
        self.cancel_check = cancel_check
        self.stats = TransportStats()
        self._sleep = sleep_fn or time.sleep
        self._clock = clock_fn or time.monotonic
        self._injected_http_client = http_client is not None

        # Deterministic initialization: the SDK client (and its connection
        # pool) exists from construction, not on first use.
        client_kwargs = {
            'api_key': self.api_key,
            'base_url': self.api_url,
            'timeout': self.timeout,
            # One retry owner: the transport policy in translation.transport.
            'max_retries': 0,
        }
        self._owns_http_client = False
        if http_client is not None:
            client_kwargs['http_client'] = http_client
        elif self.proxy:
            client_kwargs['http_client'] = httpx.Client(
                proxy=self.proxy,
                timeout=httpx.Timeout(self.timeout, connect=30.0),
            )
            self._owns_http_client = True

        self._client = OpenAI(**client_kwargs)
        self._closed = False

    def _get_client(self) -> OpenAI:
        """Return the SDK client built at construction time."""
        return self._client

    def close(self) -> None:
        """Close the clients this adapter created.

        Idempotent. The SDK's close() also closes the httpx client riding
        inside it, so with an injected http_client the SDK close is skipped
        entirely: the injected transport's owner decides its lifetime.
        """
        if self._closed:
            return
        self._closed = True
        if self._injected_http_client:
            return
        try:
            self._client.close()
        except Exception:  # pragma: no cover - defensive on SDK changes
            logger.debug('[OpenAIClient] close ignored an SDK error', exc_info=True)

    def _counted_dispatch(self, kwargs: dict):
        """One logical request through the transport retry policy."""
        self.stats.logical_requests += 1
        client = self._get_client()
        return request_with_retry(
            lambda: client.chat.completions.create(**kwargs),
            cancel_check=self.cancel_check,
            sleeper=self._sleep,
            clock=self._clock,
            stats=self.stats,
            on_event=self._log_attempt,
        )

    @staticmethod
    def _log_attempt(event: dict) -> None:
        # Attempt diagnostics only: status, error type, timing. Never the
        # request body, credentials, or the provider's response payload.
        level_name = event.pop('level', 'DEBUG')
        level = getattr(logging, level_name if level_name in ('DEBUG', 'INFO') else 'WARNING')
        logger.log(level, '[OpenAIClient] %s', event)

    # Retries exist for transient faults (connection drops, rate limits). A
    # length stop is not transient: the same prompt at the same temperature
    # produces the same over-long generation, so re-sending it only adds the
    # backoff delay before failing identically. The transport policy
    # classifies it as context-length and never retries it.
    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """
        Send messages and return the raw text response.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model name to use for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)

        Returns:
            Raw text content from the LLM response

        Raises:
            RuntimeError: If the response has no usable shape
            LengthFinishReasonError: If the response was truncated with no
                usable content
            ProviderUnavailableError: Transport retries exhausted
            FatalProviderError: The provider rejected the request outright
            TransportCancelledError: The run was cancelled before dispatch
        """
        return self._request_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            allow_empty_length=False,
        ).content

    def complete_with_metadata(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> CompletionResult:
        """Complete a request while retaining its finish reason.

        Translation callers keep using ``complete`` and therefore retain the
        existing partial-content salvage behavior. Strict structured callers,
        such as story analysis, can reject a length-stopped JSON document.
        """
        return self._request_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            allow_empty_length=True,
        )

    def complete_analysis(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        response_schema: Optional[Dict] = None,
    ) -> CompletionResult:
        """Reserve structured-analysis output for JSON instead of reasoning.

        Supporting llama-server templates honor this per-request control.
        Other completion callers retain their existing template settings, and
        a template that ignores the control still faces strict length checks.
        """
        return self._request_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            allow_empty_length=True,
            extra_body={'chat_template_kwargs': {'enable_thinking': False}},
            response_format=(
                {'type': 'json_schema', 'json_schema': {
                    'name': 'story_context', 'strict': True, 'schema': response_schema,
                }} if response_schema is not None else None
            ),
        )

    def _request_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        allow_empty_length: bool,
        extra_body: Optional[Dict] = None,
        response_format: Optional[Dict] = None,
    ) -> CompletionResult:
        kwargs = {
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'frequency_penalty': 0,
            'messages': messages,
        }
        if extra_body is not None:
            kwargs['extra_body'] = extra_body
        if response_format is not None:
            kwargs['response_format'] = response_format

        length_error = None

        try:
            response = self._counted_dispatch(kwargs)
        except LengthFinishReasonError as error:
            # SDK parsing paths can raise before returning a chat completion
            # when finish_reason is "length". The completion is
            # still attached to the exception, including any partial content.
            # Normalize that SDK behavior so metadata callers can shrink a
            # structured request instead of failing before seeing the reason.
            response = error.completion
            length_error = error
            logger.warning(
                '[OpenAIClient] SDK reported a length-stopped response; '
                'recovering its completion metadata'
            )
        except (ProviderUnavailableError, TransportCancelledError) as error:
            # The endpoint rides along so a remote-profile failure names the
            # provider instead of surfacing as a bare "connection error".
            logger.error(f'[OpenAIClient] API call failed ({self.api_url}): {error}')
            raise
        except Exception as e:
            logger.error(f'[OpenAIClient] API call failed ({self.api_url}): {e}')
            raise

        # Validate response. A malformed body is a content problem for the
        # caller's recovery path, not a transport fault, so it is never
        # retried here.
        if isinstance(response, str):
            raise RuntimeError(f'Invalid response type: {response}')

        if not hasattr(response, 'choices') or not response.choices:
            raise RuntimeError('Invalid response - no choices')

        content = response.choices[0].message.content

        # A truncated response is not automatically a failed one. Local models
        # often prepend commentary and then emit the YAML we asked for, so by
        # the time the token budget runs out there is usually a complete entry
        # or two in hand. Returning that partial text lets the caller's parser
        # salvage what arrived; discarding it here caused fully translated lines
        # to fall back to their untranslated source. Only raise when nothing
        # usable came back, since then there is genuinely nothing to parse.
        if response.choices[0].finish_reason == 'length':
            if content and content.strip():
                logger.warning(
                    '[OpenAIClient] Response hit the token limit; returning '
                    'the partial content for parsing'
                )
                return CompletionResult(content.strip(), "length")
            if allow_empty_length:
                return CompletionResult('', "length")
            if length_error is not None:
                raise length_error
            raise LengthFinishReasonError(completion=response)

        if content is None:
            raise RuntimeError(
                f"[OpenAIClient] None content - finish_reason: "
                f"{response.choices[0].finish_reason}"
            )

        if not content or not content.strip():
            return CompletionResult('', response.choices[0].finish_reason)

        return CompletionResult(content.strip(), response.choices[0].finish_reason)

    def __repr__(self) -> str:
        """String representation of the client (never includes the key)."""
        return (
            f"OpenAIClient("
            f"api_url='{self.api_url}', "
            f"model='{self.model}', "
            f"timeout={self.timeout})"
        )
