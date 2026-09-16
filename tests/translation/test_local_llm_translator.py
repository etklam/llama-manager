# -*- coding: utf-8 -*-
"""
Unit tests for the Local-LLM translator module.

These tests verify the Local-LLM translator class which uses local LLM models
via OpenAI-compatible API endpoints for translation.

This is a TDD (Test-Driven Development) approach - these tests are written
BEFORE the implementation exists and will FAIL until the module is created.
"""
import pytest
from unittest.mock import patch, MagicMock, Mock
from openai import OpenAI, APIConnectionError
import httpx
import sys
import os

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# This import will FAIL because the module doesn't exist yet - this is expected in TDD!
from translation.local_llm_translator import LocalLLMTranslator
from translation.transport import ProviderUnavailableError
from utils.srt_parser import Cue


def _full_config(**overrides):
    # ponytail: after the cleanup, LocalLLMTranslator trusts the dict (mirrors
    # config_helpers.build_translation_config output).
    cfg = {
        'api_url': 'http://localhost:8080/v1',
        'model': 'llama-3.2-3b-instruct',
        'max_tokens': 4096,
        'temperature': 0.3,
        'batch_size': 15,
        'max_workers': 3,
        'single_step': False,
    }
    cfg.update(overrides)
    return cfg


class TestTranslatorInitialization:
    """Test Local-LLM translator initialization."""

    def test_translator_initialization_with_config(self):
        """Test translator initialization with full configuration."""
        config = _full_config(max_tokens=2000, temperature=0.7)

        translator = LocalLLMTranslator(config)

        assert translator.api_url == 'http://localhost:8080/v1'
        assert translator.model == 'llama-3.2-3b-instruct'
        assert translator.max_tokens == 2000
        assert translator.temperature == 0.7

    def test_translator_initialization_default_values(self):
        """Test translator initialization with default values."""
        config = _full_config()

        translator = LocalLLMTranslator(config)

        assert hasattr(translator, 'max_tokens')
        assert hasattr(translator, 'temperature')
        # Verify defaults are set
        assert translator.max_tokens > 0
        assert 0 <= translator.temperature <= 2

    def test_translator_initialization_minimal_config(self):
        """Test translator initialization with full config (api_url always supplied)."""
        config = _full_config()

        translator = LocalLLMTranslator(config)

        assert translator.model == 'llama-3.2-3b-instruct'
        assert 'localhost:8080' in translator.api_url or '127.0.0.1:8080' in translator.api_url


class TestSingleTextTranslation:
    """Test single text translation functionality."""

    @patch('translation.openai_client.OpenAI')
    def test_translate_single_text(self, mock_openai):
        """Test translating a single text string."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Bonjour le monde'
        mock_client.chat.completions.create.return_value = mock_response

        # Create translator and translate
        config = _full_config()
        translator = LocalLLMTranslator(config)

        result = translator.translate('Hello world', target_language='French')

        # Verify result
        assert result == 'Bonjour le monde'

        # Verify API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['model'] == 'llama-3.2-3b-instruct'
        assert 'messages' in call_args.kwargs

    @patch('translation.openai_client.OpenAI')
    def test_translate_single_text_with_context(self, mock_openai):
        """Test translating with context information."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Hola'
        mock_client.chat.completions.create.return_value = mock_response

        config = _full_config()
        translator = LocalLLMTranslator(config)

        result = translator.translate(
            'Hello',
            target_language='Spanish',
            context='Greeting in a formal setting'
        )

        assert result == 'Hola'
        # Verify context was included in the prompt
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']
        assert any('formal setting' in str(msg).lower() for msg in messages)


class TestBatchTextTranslation:
    """Test batch text translation functionality."""

    @patch('translation.openai_client.OpenAI')
    def test_translate_batch_text(self, mock_openai):
        """Test translating a list of texts."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Setup mock responses for batch
        responses = [
            MagicMock(choices=[MagicMock(message=MagicMock(content='Bonjour'))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content='Monde'))]),
        ]
        mock_client.chat.completions.create.side_effect = responses

        config = _full_config()
        translator = LocalLLMTranslator(config)

        texts = ['Hello', 'World']
        results = translator.translate_batch(texts, target_language='French')

        assert len(results) == 2
        assert results[0] == 'Bonjour'
        assert results[1] == 'Monde'

        # Verify API was called twice
        assert mock_client.chat.completions.create.call_count == 2

    @patch('translation.openai_client.OpenAI')
    def test_translate_batch_empty_list(self, mock_openai):
        """Test translating an empty list returns empty result."""
        config = _full_config()
        translator = LocalLLMTranslator(config)

        results = translator.translate_batch([], target_language='French')

        assert results == []
        # API should not be called
        mock_openai.assert_not_called()


class TestSRTFormatTranslation:
    """Test SRT subtitle format translation."""

    @patch('translation.openai_client.OpenAI')
    def test_translate_srt_format(self, mock_openai):
        """Test translating SRT subtitle list while preserving timestamps."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Bonjour\nÀ tout à l\'heure'
        mock_client.chat.completions.create.return_value = mock_response

        config = _full_config()
        translator = LocalLLMTranslator(config)

        srt_data = [
            Cue(line=1, start_time=1000, end_time=2000, text='Hello'),
            Cue(line=2, start_time=3000, end_time=4000, text='See you later'),
        ]

        result = translator.translate_srt(srt_data, target_language='French')

        # Verify timestamps are preserved
        assert result[0].start_time == 1000
        assert result[0].end_time == 2000
        assert result[1].start_time == 3000
        assert result[1].end_time == 4000

        # Verify line numbers are preserved
        assert result[0].line == 1
        assert result[1].line == 2

        # Verify text is translated
        assert 'Bonjour' in result[0].text
        assert 'tout à l\'heure' in result[1].text

    @patch('translation.openai_client.OpenAI')
    def test_translate_srt_empty_list(self, mock_openai):
        """Test translating empty SRT list."""
        config = _full_config()
        translator = LocalLLMTranslator(config)

        result = translator.translate_srt([], target_language='French')

        assert result == []
        mock_openai.assert_not_called()


class TestAPIErrorRetry:
    """Typed transport failures retry inside the adapter; exhaustion surfaces."""

    @patch('translation.openai_client.time')
    @patch('translation.openai_client.OpenAI')
    def test_api_error_retry_success_after_retries(self, mock_openai, mock_time):
        """Test that translator retries on API errors and eventually succeeds."""
        mock_time.sleep = lambda seconds: None
        mock_time.monotonic = lambda: 0.0
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Setup: fail twice with typed transient errors, then succeed
        responses = [
            APIConnectionError(request=httpx.Request('POST', 'http://x/v1')),
            APIConnectionError(request=httpx.Request('POST', 'http://x/v1')),
            MagicMock(choices=[MagicMock(message=MagicMock(content='Success'))]),
        ]
        mock_client.chat.completions.create.side_effect = responses

        config = _full_config()
        translator = LocalLLMTranslator(config)

        result = translator.translate('Test', target_language='French')

        assert result == 'Success'
        # Verify it retried
        assert mock_client.chat.completions.create.call_count == 3

    @patch('translation.openai_client.time')
    @patch('translation.openai_client.OpenAI')
    def test_api_error_retry_max_retries_exceeded(self, mock_openai, mock_time):
        """Exhausted transport retries raise ProviderUnavailableError."""
        mock_time.sleep = lambda seconds: None
        mock_time.monotonic = lambda: 0.0
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Always fail with a typed transient error
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=httpx.Request('POST', 'http://x/v1'))

        config = _full_config()
        translator = LocalLLMTranslator(config)

        with pytest.raises(ProviderUnavailableError):
            translator.translate('Test', target_language='French')

        # Verify it retried multiple times (typically 3-5 retries)
        assert mock_client.chat.completions.create.call_count >= 2

    @patch('translation.openai_client.OpenAI')
    def test_rate_limit_retry(self, mock_openai):
        """Test retry on rate limit errors."""
        from openai import RateLimitError

        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Rate limit once, then succeed
        responses = [
            RateLimitError('Rate limit exceeded', response=MagicMock(), body=None),
            MagicMock(choices=[MagicMock(message=MagicMock(content='Success'))]),
        ]
        mock_client.chat.completions.create.side_effect = responses

        config = _full_config()
        translator = LocalLLMTranslator(config)

        result = translator.translate('Test', target_language='French')

        assert result == 'Success'
        assert mock_client.chat.completions.create.call_count == 2


class TestPromptGeneration:
    """Test prompt generation for translation."""

    @patch('translation.openai_client.OpenAI')
    def test_prompt_includes_system_message(self, mock_openai):
        """Test that prompt includes system message."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Translated'))]
        )

        config = _full_config()
        translator = LocalLLMTranslator(config)

        translator.translate('Hello', target_language='French')

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']

        # First message should be system message
        assert messages[0]['role'] == 'system'
        # System prompt is now in Chinese: "翻译" (translate)
        assert '翻译' in messages[0]['content'] or 'translate' in messages[0]['content'].lower() or 'translation' in messages[0]['content'].lower()

    @patch('translation.openai_client.OpenAI')
    def test_prompt_includes_user_text(self, mock_openai):
        """Test that prompt includes the text to translate."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Translated'))]
        )

        config = _full_config()
        translator = LocalLLMTranslator(config)

        translator.translate('Hello world', target_language='French')

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']

        # Should have a user message with the text
        user_messages = [m for m in messages if m['role'] == 'user']
        assert len(user_messages) > 0
        assert 'Hello world' in user_messages[0]['content']

    @patch('translation.openai_client.OpenAI')
    def test_prompt_includes_target_language(self, mock_openai):
        """Test that prompt includes target language."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Bonjour'))]
        )

        config = _full_config()
        translator = LocalLLMTranslator(config)

        translator.translate('Hello', target_language='French')

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']

        # Target language should be mentioned in messages
        messages_str = str(messages).lower()
        assert 'french' in messages_str

    @patch('translation.openai_client.OpenAI')
    def test_prompt_includes_context_when_provided(self, mock_openai):
        """Test that prompt includes context when provided."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Translated'))]
        )

        config = _full_config()
        translator = LocalLLMTranslator(config)

        translator.translate(
            'Bank',
            target_language='French',
            context='Financial institution, not river bank'
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']

        messages_str = str(messages).lower()
        assert 'financial' in messages_str or 'context' in messages_str


class TestTemperatureSetting:
    """Test temperature parameter settings."""

    @patch('translation.openai_client.OpenAI')
    def test_temperature_passed_to_api(self, mock_openai):
        """Test that temperature parameter is passed to API."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Translated'))]
        )

        config = _full_config(temperature=0.8)
        translator = LocalLLMTranslator(config)

        translator.translate('Hello', target_language='French')

        call_args = mock_client.chat.completions.create.call_args
        assert 'temperature' in call_args.kwargs
        assert call_args.kwargs['temperature'] == 0.8

    @patch('translation.openai_client.OpenAI')
    def test_different_temperature_values(self, mock_openai):
        """Test different temperature values."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        temperatures = [0.0, 0.5, 1.0, 1.5]

        for temp in temperatures:
            mock_client.chat.completions.create.reset_mock()
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content='Translated'))]
            )

            config = _full_config(temperature=temp)
            translator = LocalLLMTranslator(config)
            translator.translate('Hello', target_language='French')

            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs['temperature'] == temp

    @patch('translation.openai_client.OpenAI')
    def test_temperature_bounds_checking(self, mock_openai):
        """Test that temperature is properly bounded."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Translated'))]
        )

        # Test with temperature > 2.0 (should be clamped)
        config = _full_config(temperature=3.0)
        translator = LocalLLMTranslator(config)
        translator.translate('Hello', target_language='French')

        call_args = mock_client.chat.completions.create.call_args
        # Temperature should be <= 2.0
        assert call_args.kwargs['temperature'] <= 2.0


class TestMaxTokensSetting:
    """Test max_tokens parameter settings."""

    @patch('translation.openai_client.OpenAI')
    def test_max_tokens_passed_to_api(self, mock_openai):
        """max_tokens sent to the API is the request-sized dynamic value.

        config['max_tokens'] is now an upper cap, not the literal value sent.
        A single-line translate() sizes the request via _dynamic_max_tokens(1),
        which stays well under the 1500 cap here.
        """
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Translated'))]
        )

        config = _full_config(max_tokens=1500)
        translator = LocalLLMTranslator(config)

        translator.translate('Hello', target_language='French')

        call_args = mock_client.chat.completions.create.call_args
        assert 'max_tokens' in call_args.kwargs
        assert call_args.kwargs['max_tokens'] == translator._dynamic_max_tokens(1)

    @patch('translation.openai_client.OpenAI')
    def test_different_max_tokens_values(self, mock_openai):
        """The cap bounds the dynamic value: a small cap wins, a large one doesn't."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        max_tokens_values = [500, 1000, 2000, 4000]

        for max_tok in max_tokens_values:
            mock_client.chat.completions.create.reset_mock()
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content='Translated'))]
            )

            config = _full_config(max_tokens=max_tok)
            translator = LocalLLMTranslator(config)
            translator.translate('Hello', target_language='French')

            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs['max_tokens'] == translator._dynamic_max_tokens(1)
            # Never exceeds the configured cap.
            assert call_args.kwargs['max_tokens'] <= max_tok


class TestAPIURLConfiguration:
    """Test API URL configuration."""

    @patch('translation.openai_client.OpenAI')
    def test_custom_api_url(self, mock_openai):
        """Test that custom API URL is used."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Translated'))]
        )

        custom_url = 'http://192.168.1.100:8000/v1'
        config = _full_config(api_url=custom_url)
        translator = LocalLLMTranslator(config)

        translator.translate('Hello', target_language='French')

        # Verify OpenAI client was initialized with custom URL
        mock_openai.assert_called_once()
        call_args = mock_openai.call_args
        assert 'base_url' in call_args.kwargs
        assert call_args.kwargs['base_url'] == custom_url

    @patch('translation.openai_client.OpenAI')
    def test_default_localhost_url(self, mock_openai):
        """Test default localhost:8080 URL when not specified."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Translated'))]
        )

        config = _full_config()
        translator = LocalLLMTranslator(config)

        translator.translate('Hello', target_language='French')

        # Verify OpenAI client was initialized with default URL
        mock_openai.assert_called_once()
        call_args = mock_openai.call_args
        assert 'base_url' in call_args.kwargs
        assert 'localhost:8080' in call_args.kwargs['base_url'] or '127.0.0.1:8080' in call_args.kwargs['base_url']

    @patch('translation.openai_client.OpenAI')
    def test_https_url(self, mock_openai):
        """Test HTTPS URL configuration."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Translated'))]
        )

        config = _full_config(api_url='https://api.example.com/v1')
        translator = LocalLLMTranslator(config)

        translator.translate('Hello', target_language='French')

        call_args = mock_openai.call_args
        assert call_args.kwargs['base_url'] == 'https://api.example.com/v1'


class TestEmptyInputHandling:
    """Test handling of empty inputs."""

    @patch('translation.openai_client.OpenAI')
    def test_empty_string_translation(self, mock_openai):
        """Test translating an empty string."""
        config = _full_config()
        translator = LocalLLMTranslator(config)

        result = translator.translate('', target_language='French')

        # Should return empty string or handle gracefully
        assert result == ''
        # API should not be called for empty input
        mock_openai.assert_not_called()

    @patch('translation.openai_client.OpenAI')
    def test_whitespace_only_translation(self, mock_openai):
        """Test translating whitespace-only text."""
        config = _full_config()
        translator = LocalLLMTranslator(config)

        result = translator.translate('   \n\t  ', target_language='French')

        # Should handle gracefully
        assert result.strip() == '' or result == '   \n\t  '
        # API might or might not be called depending on implementation

    @patch('translation.openai_client.OpenAI')
    def test_none_target_language(self, mock_openai):
        """Test handling of None target language."""
        config = _full_config()
        translator = LocalLLMTranslator(config)

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, TypeError)):
            translator.translate('Hello', target_language=None)


class TestModelConfiguration:
    """Test model configuration."""

    @patch('translation.openai_client.OpenAI')
    def test_model_passed_to_api(self, mock_openai):
        """Test that model parameter is passed to API."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Translated'))]
        )

        config = _full_config()
        translator = LocalLLMTranslator(config)

        translator.translate('Hello', target_language='French')

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['model'] == 'llama-3.2-3b-instruct'

    @patch('translation.openai_client.OpenAI')
    def test_different_model_names(self, mock_openai):
        """Test different model names."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        models = [
            'llama-3.2-3b-instruct',
            'llama-3.1-8b-instruct',
            'mistral-7b-instruct',
            'qwen-7b-instruct',
        ]

        for model in models:
            mock_client.chat.completions.create.reset_mock()
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content='Translated'))]
            )

            config = _full_config(model=model)
            translator = LocalLLMTranslator(config)
            translator.translate('Hello', target_language='French')

            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs['model'] == model


class TestResponseParsing:
    """Test response parsing and handling."""

    @patch('translation.openai_client.OpenAI')
    def test_response_with_no_choices(self, mock_openai):
        """Test handling of response with no choices."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response

        config = _full_config()
        translator = LocalLLMTranslator(config)

        with pytest.raises((ValueError, RuntimeError)):
            translator.translate('Hello', target_language='French')

    @patch('translation.openai_client.OpenAI')
    def test_response_with_empty_content(self, mock_openai):
        """Test handling of response with empty content."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ''
        mock_client.chat.completions.create.return_value = mock_response

        config = _full_config()
        translator = LocalLLMTranslator(config)

        with pytest.raises(RuntimeError, match='empty translation'):
            translator.translate('Hello', target_language='French')

    @patch('translation.openai_client.OpenAI')
    def test_response_with_none_content(self, mock_openai):
        """Test handling of response with None content."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response

        config = _full_config()
        translator = LocalLLMTranslator(config)

        with pytest.raises((ValueError, RuntimeError)):
            translator.translate('Hello', target_language='French')


class TestSpecialCases:
    """Test special cases and edge cases."""

    @patch('translation.openai_client.OpenAI')
    def test_very_long_text(self, mock_openai):
        """Test translating very long text."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Long text translated'))]
        )

        config = _full_config()
        translator = LocalLLMTranslator(config)

        long_text = 'Hello ' * 1000  # 5000 characters
        result = translator.translate(long_text, target_language='French')

        assert result == 'Long text translated'
        # Verify the request was made
        mock_client.chat.completions.create.assert_called_once()

    @patch('translation.openai_client.OpenAI')
    def test_special_characters(self, mock_openai):
        """Test translating text with special characters."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Translated with special chars: @#$%'))]
        )

        config = _full_config()
        translator = LocalLLMTranslator(config)

        text = 'Hello @#$%^&*() World!'
        result = translator.translate(text, target_language='French')

        assert 'Translated' in result
        mock_client.chat.completions.create.assert_called_once()

    @patch('translation.openai_client.OpenAI')
    def test_unicode_text(self, mock_openai):
        """Test translating text with unicode characters."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='你好世界'))]
        )

        config = _full_config()
        translator = LocalLLMTranslator(config)

        text = 'Hello 世界'
        result = translator.translate(text, target_language='Chinese')

        assert '你好' in result or '世界' in result
        mock_client.chat.completions.create.assert_called_once()
