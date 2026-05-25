"""
OpenAI Client Adapter - Production adapter wrapping OpenAI SDK.

This module provides the production implementation of the LLMClient port,
wrapping the OpenAI SDK with retry logic.
"""
import logging
from typing import List, Dict

import httpx
from openai import OpenAI, LengthFinishReasonError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError
)

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

    @retry(
        stop=stop_after_attempt(RETRY_NUMS),
        wait=wait_exponential(multiplier=1, min=RETRY_DELAY, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
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
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=0,
                messages=messages
            )

            logger.debug(f'[OpenAIClient] Response: {response}')

        except Exception as e:
            logger.error(f'[OpenAIClient] API call failed: {e}')
            raise

        # Validate response
        if isinstance(response, str):
            raise RuntimeError(f'Invalid response type: {response}')

        if not hasattr(response, 'choices') or not response.choices:
            raise RuntimeError(f'Invalid response - no choices: {response}')

        if response.choices[0].finish_reason == 'length':
            raise LengthFinishReasonError(completion=response)

        content = response.choices[0].message.content

        if content is None:
            raise RuntimeError(
                f"[OpenAIClient] None content - finish_reason: "
                f"{response.choices[0].finish_reason}"
            )

        if not content or not content.strip():
            return ''

        return content.strip()

    def __repr__(self) -> str:
        """String representation of the client."""
        return (
            f"OpenAIClient("
            f"api_url='{self.api_url}', "
            f"model='{self.model}', "
            f"timeout={self.timeout})"
        )
