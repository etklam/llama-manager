"""
LLM Client Protocol - Port at the seam between translator and LLM API.

This module defines the Protocol for the LLM client seam, allowing dependency
injection and testability.
"""
from dataclasses import dataclass
from typing import Protocol, List, Dict, Optional


@dataclass(frozen=True)
class CompletionResult:
    """Response content plus the stop reason needed by strict consumers."""

    content: str
    finish_reason: Optional[str] = None


class LLMClient(Protocol):
    """
    Port at the seam between translator and LLM API.

    This Protocol defines the interface that any LLM client adapter must implement.
    It allows the translator to be decoupled from any specific LLM API implementation,
    making it easy to swap implementations (OpenAI, local LLM, mock for testing, etc.).

    The complete() method should handle:
    - Sending messages to the LLM API
    - Retry logic for transient errors
    - Returning the raw text response
    """

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
            Exception: If the API call fails after retries
        """
        ...
