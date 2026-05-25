"""
TDD tests for LocalLLMTranslator with injected LLMClient.
These tests verify dependency injection works correctly.
"""
import pytest
from translation.local_llm_translator import LocalLLMTranslator
from translation.llm_client import LLMClient


class FakeLLMClient:
    """Fake LLM client for testing."""

    def __init__(self, response_text="Fake translated response"):
        self.response_text = response_text
        self.call_count = 0
        self.last_messages = None
        self.last_model = None
        self.last_max_tokens = None
        self.last_temperature = None

    def complete(self, messages, model, max_tokens, temperature) -> str:
        """Fake complete() that returns preset response."""
        self.call_count += 1
        self.last_messages = messages
        self.last_model = model
        self.last_max_tokens = max_tokens
        self.last_temperature = temperature
        return self.response_text


class TestTranslatorWithInjectedClient:
    """Test LocalLLMTranslator with dependency injection."""

    def test_translator_uses_injected_client(self):
        """Test that translator uses injected client instead of creating OpenAI."""
        fake_client = FakeLLMClient(response_text="Test translation")

        config = {
            'model': 'test-model',
            'api_url': 'http://fake.com/v1',
        }

        translator = LocalLLMTranslator(config, client=fake_client)

        result = translator.translate("Hello", target_language="French")

        assert result == "Test translation"
        assert fake_client.call_count == 1
        assert fake_client.last_model == 'test-model'
        assert fake_client.last_temperature == 0.3  # default temperature

    def test_translator_with_custom_temperature(self):
        """Test that temperature is passed to injected client."""
        fake_client = FakeLLMClient(response_text="Response")

        config = {
            'model': 'test-model',
            'temperature': 0.8,
        }

        translator = LocalLLMTranslator(config, client=fake_client)

        translator.translate("Hello", target_language="French")

        assert fake_client.last_temperature == 0.8

    def test_translator_with_custom_max_tokens(self):
        """Test that max_tokens is passed to injected client."""
        fake_client = FakeLLMClient(response_text="Response")

        config = {
            'model': 'test-model',
            'max_tokens': 2000,
        }

        translator = LocalLLMTranslator(config, client=fake_client)

        translator.translate("Hello", target_language="French")

        assert fake_client.last_max_tokens == 2000

    def test_translator_batch_uses_injected_client(self):
        """Test that batch translation uses injected client."""
        fake_client = FakeLLMClient(response_text="Translated")

        config = {'model': 'test-model'}

        translator = LocalLLMTranslator(config, client=fake_client)

        results = translator.translate_batch(["Hello", "World"], target_language="French")

        assert len(results) == 2
        assert all(r == "Translated" for r in results)
        assert fake_client.call_count == 2

    def test_translator_srt_uses_injected_client(self):
        """Test that SRT translation uses injected client."""
        fake_client = FakeLLMClient(response_text="- id: 1\n  step1: 直译\n  step2: 意译")

        config = {'model': 'test-model'}

        translator = LocalLLMTranslator(config, client=fake_client)

        srt_data = [
            {'text': 'Hello', 'time': '00:00:01,000 --> 00:00:02,000', 'line': 1},
        ]

        result = translator.translate_srt(srt_data, target_language="zh-cn")

        assert len(result) == 1
        assert result[0]['text'] == '意译'
        assert result[0]['time'] == '00:00:01,000 --> 00:00:02,000'
        assert fake_client.call_count == 1

    def test_translator_backward_compatibility(self):
        """Test that translator still works without injected client."""
        # This should create an OpenAIClient internally
        config = {
            'model': 'test-model',
            'api_url': 'http://localhost:8080/v1',
        }

        # Should not raise an error
        translator = LocalLLMTranslator(config)

        # The translator should have created an internal client
        assert translator._injected_client is None
        assert translator._openai_client is None  # Not created yet (lazy)
        assert translator._client is None  # Legacy client also lazy


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
