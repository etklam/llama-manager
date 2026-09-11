"""
OpenAI Client Adapter - Production adapter wrapping OpenAI SDK.

This module provides the production implementation of the LLMClient port,
wrapping the OpenAI SDK with retry logic.
"""
import logging
from typing import List, Dict, Optional

import httpx
from openai import OpenAI, LengthFinishReasonError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_not_exception_type,
    before_sleep_log,
    RetryError
)
from translation.llm_client import CompletionResult

logger = logging.getLogger(__name__)

# Constants
RETRY_NUMS = 3
RETRY_DELAY = 1  # Initial delay for exponential backoff


class OpenAIClient:
    """
    Production adapter: wraps OpenAI SDK with retry logic.

    This adapter implements the LLMClient port by wrapping the OpenAI SDK.
    It handles retry logic, proxy configuration, and timeout management.
    """

    def __init__(
        self,
        api_url: str,
        model: str,
        api_key: str = '',
        proxy: str = None,
        timeout: float = 180.0
    ):
        """
        Initialize the OpenAI client adapter.

        Args:
            api_url: API endpoint URL
            model: Model name (stored for reference, passed to complete())
            api_key: API key if required (default: empty string)
            proxy: Proxy URL for HTTP client (default: None)
            timeout: Request timeout in seconds (default: 180.0)
        """
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.proxy = proxy
        self.timeout = timeout

        # Initialize OpenAI client (will be created lazily)
        self._client = None

    def _get_client(self) -> OpenAI:
        """
        Get or create the OpenAI client.

        Returns:
            OpenAI client instance configured with the API URL, timeout, and optional proxy
        """
        if self._client is None:
            client_kwargs = {
                'api_key': self.api_key,
                'base_url': self.api_url,
                'timeout': self.timeout,
            }

            if self.proxy:
                client_kwargs['http_client'] = httpx.Client(
                    proxy=self.proxy,
                    timeout=httpx.Timeout(self.timeout, connect=30.0)
                )

            self._client = OpenAI(**client_kwargs)

        return self._client

    # Retries exist for transient faults (connection drops, rate limits). A
    # length stop is not transient: the same prompt at the same temperature
    # produces the same over-long generation, so re-sending it only adds the
    # backoff delay before failing identically. Excluding it turns three
    # identical failures per line into one.
    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """
        Send messages and return the raw text response.

        This method implements retry logic using tenacity for transient errors.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model name to use for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)

        Returns:
            Raw text content from the LLM response

        Raises:
            RuntimeError: If the API call fails or returns invalid response
            LengthFinishReasonError: If the response was truncated due to length
            Exception: If the API call fails after retries
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

    @retry(
        stop=stop_after_attempt(RETRY_NUMS),
        wait=wait_exponential(multiplier=1, min=RETRY_DELAY, max=10),
        retry=retry_if_not_exception_type(LengthFinishReasonError),
        before_sleep=before_sleep_log(logger, logging.WARNING)
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
        client = self._get_client()
        length_error = None

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=0,
                messages=messages,
                **({'extra_body': extra_body} if extra_body is not None else {}),
                **({'response_format': response_format} if response_format is not None else {}),
            )

            logger.debug(f'[OpenAIClient] Response: {response}')

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
        except Exception as e:
            logger.error(f'[OpenAIClient] API call failed: {e}')
            raise

        # Validate response
        if isinstance(response, str):
            raise RuntimeError(f'Invalid response type: {response}')

        if not hasattr(response, 'choices') or not response.choices:
            raise RuntimeError(f'Invalid response - no choices: {response}')

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
        """String representation of the client."""
        return (
            f"OpenAIClient("
            f"api_url='{self.api_url}', "
            f"model='{self.model}', "
            f"timeout={self.timeout})"
        )
