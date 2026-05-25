"""
TDD tests for OpenAI client adapter.
These tests define the behavior of the LLM client port and adapter.
"""
import pytest
from unittest.mock import patch, MagicMock
from openai import RateLimitError

# These imports will FAIL until we create the modules
from translation.llm_client import LLMClient
from translation.openai_client import OpenAIClient, RETRY_NUMS


class TestLLMClientProtocol:
    """Test that LLMClient is a valid Protocol."""

    def test_llm_client_is_protocol(self):
        """Test that LLMClient is defined as a Protocol."""
        from typing import Protocol
        # Check if LLMClient is a Protocol
        assert issubclass(LLMClient, Protocol)


class TestOpenAIClientInitialization:
    """Test OpenAI client adapter initialization."""

    def test_initialization_with_basic_params(self):
        """Test client initialization with basic parameters."""
        client = OpenAIClient(
            api_url="http://localhost:8080/v1",
            model="llama-3.2-3b-instruct"
        )
        assert client.api_url == "http://localhost:8080/v1"
        assert client.model == "llama-3.2-3b-instruct"

    def test_initialization_with_all_params(self):
        """Test client initialization with all parameters."""
        client = OpenAIClient(
            api_url="http://localhost:8080/v1",
            api_key="test-key",
            proxy="http://proxy:8080",
            timeout=120.0,
            model="llama-3.2-3b-instruct"
        )
        assert client.api_url == "http://localhost:8080/v1"
        assert client.api_key == "test-key"
        assert client.proxy == "http://proxy:8080"
        assert client.timeout == 120.0

    def test_default_values(self):
        """Test default parameter values."""
        client = OpenAIClient(
            api_url="http://localhost:8080/v1",
            model="llama-3.2-3b-instruct"
        )
        assert client.api_key == ''
        assert client.proxy is None
        assert client.timeout == 180.0


class TestOpenAIClientComplete:
    """Test the complete() method of OpenAIClient."""

    @patch('translation.openai_client.OpenAI')
    def test_complete_basic_call(self, mock_openai):
        """Test basic API call through complete()."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Test response'
        mock_client.chat.completions.create.return_value = mock_response

        # Create client and call
        client = OpenAIClient(
            api_url="http://localhost:8080/v1",
            model="llama-3.2-3b-instruct"
        )

        messages = [
            {'role': 'system', 'content': 'You are a translator.'},
            {'role': 'user', 'content': 'Translate hello.'}
        ]

        result = client.complete(
            messages=messages,
            model="llama-3.2-3b-instruct",
            max_tokens=1000,
            temperature=0.5
        )

        # Verify result
        assert result == 'Test response'

        # Verify API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['model'] == 'llama-3.2-3b-instruct'
        assert call_args.kwargs['max_tokens'] == 1000
        assert call_args.kwargs['temperature'] == 0.5
        assert 'messages' in call_args.kwargs

    @patch('translation.openai_client.OpenAI')
    def test_complete_with_retry_on_error(self, mock_openai):
        """Test that complete() retries on transient errors."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Fail once, then succeed
        responses = [
            Exception('Connection error'),
            MagicMock(choices=[MagicMock(message=MagicMock(content='Success'))]),
        ]
        mock_client.chat.completions.create.side_effect = responses

        client = OpenAIClient(
            api_url="http://localhost:8080/v1",
            model="llama-3.2-3b-instruct"
        )

        messages = [{'role': 'user', 'content': 'Test'}]

        result = client.complete(
            messages=messages,
            model="llama-3.2-3b-instruct",
            max_tokens=1000,
            temperature=0.5
        )

        assert result == 'Success'
        assert mock_client.chat.completions.create.call_count == 2

    @patch('translation.openai_client.OpenAI')
    def test_complete_raises_after_max_retries(self, mock_openai):
        """Test that complete() raises exception after max retries."""
        from tenacity import RetryError

        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Always fail
        mock_client.chat.completions.create.side_effect = Exception('Permanent error')

        client = OpenAIClient(
            api_url="http://localhost:8080/v1",
            model="llama-3.2-3b-instruct"
        )

        messages = [{'role': 'user', 'content': 'Test'}]

        # Tenacity wraps exceptions in RetryError
        with pytest.raises(RetryError):
            client.complete(
                messages=messages,
                model="llama-3.2-3b-instruct",
                max_tokens=1000,
                temperature=0.5
            )

        # Should have retried multiple times (RETRY_NUMS = 3)
        assert mock_client.chat.completions.create.call_count == RETRY_NUMS

    @patch('translation.openai_client.OpenAI')
    def test_complete_with_rate_limit_retry(self, mock_openai):
        """Test retry on rate limit errors."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Rate limit once, then succeed
        responses = [
            RateLimitError('Rate limit exceeded', response=MagicMock(), body=None),
            MagicMock(choices=[MagicMock(message=MagicMock(content='Success'))]),
        ]
        mock_client.chat.completions.create.side_effect = responses

        client = OpenAIClient(
            api_url="http://localhost:8080/v1",
            model="llama-3.2-3b-instruct"
        )

        messages = [{'role': 'user', 'content': 'Test'}]

        result = client.complete(
            messages=messages,
            model="llama-3.2-3b-instruct",
            max_tokens=1000,
            temperature=0.5
        )

        assert result == 'Success'
        assert mock_client.chat.completions.create.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
