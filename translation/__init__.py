"""
Translation module for llama-manager.

Provides translation functionality using various translation backends
including local LLM models via OpenAI-compatible APIs.
"""

from .local_llm_translator import LocalLLMTranslator

__all__ = ['LocalLLMTranslator']
