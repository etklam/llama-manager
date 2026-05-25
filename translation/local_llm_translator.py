# -*- coding: utf-8 -*-
"""
Local-LLM Translator Module

Provides translation functionality using local LLM models via OpenAI-compatible API endpoints.
This module is adapted from the pyvideotrans project and simplified for use in llama-manager.

Uses a two-step translation approach (inspired by ImmersiveTranslate Paraphrase Expert):
  Step 1: Literal translation (直译)
  Step 2: Paraphrase/free translation (意译) - this is the final output
"""

import logging
import re
import time
from pathlib import Path
from typing import Callable, List, Dict, Optional

import httpx
from openai import OpenAI, LengthFinishReasonError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants
RETRY_NUMS = 3
RETRY_DELAY = 1  # Initial delay for exponential backoff
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.3
DEFAULT_API_URL = "http://localhost:8080/v1"
DEFAULT_BATCH_SIZE = 5  # Small batch for two-step translation (output is 2x)

# Load prompt template
_PROMPT_DIR = Path(__file__).parent / "prompts"
_PROMPT_TEMPLATE_PATH = _PROMPT_DIR / "srt_translation.txt"


def _load_prompt_template() -> str:
    """Load the SRT translation prompt template from file."""
    if _PROMPT_TEMPLATE_PATH.exists():
        return _PROMPT_TEMPLATE_PATH.read_text(encoding='utf-8')
    # Fallback simple prompt if template file is missing
    logger.warning(f"Prompt template not found at {_PROMPT_TEMPLATE_PATH}, using fallback")
    return "Translate the following text to {target_language}:\n\n{text}"


SRT_TRANSLATION_PROMPT = _load_prompt_template()


class LocalLLMTranslator:
    """
    Translator using local LLM models via OpenAI-compatible API.

    This translator supports:
    - Single text translation (two-step: literal + paraphrase)
    - Batch text translation with YAML format
    - SRT subtitle format translation (preserves timestamps)
    - Configurable retry logic with exponential backoff
    - Proxy support via httpx
    """

    def __init__(self, config: dict):
        """
        Initialize the Local-LLM translator.

        Args:
            config: Dictionary containing configuration keys:
                - api_url: API endpoint URL (default: http://localhost:8080/v1)
                - model: Model name (required)
                - max_tokens: Maximum tokens for generation (default: 4096)
                - temperature: Sampling temperature 0-2 (default: 0.3)
                - api_key: API key if required (default: empty string)
                - proxy: Proxy URL for HTTP client (default: None)
        """
        # Validate required config
        if 'model' not in config:
            raise ValueError("Configuration must include 'model' parameter")

        # Store config for later use
        self._config = config

        # Initialize configuration
        self._api_model = config['model']
        self.api_url = config.get('api_url', DEFAULT_API_URL)

        if 'api_url' not in config:
            self.model = config['model'].replace('.', '-')
        else:
            self.model = config['model']
        self.max_tokens = config.get('max_tokens', DEFAULT_MAX_TOKENS)
        self.temperature = config.get('temperature', DEFAULT_TEMPERATURE)
        self.api_key = config.get('api_key', '')
        self.proxy = config.get('proxy', None)

        # Validate and clamp temperature to valid range [0, 2]
        if not isinstance(self.temperature, (int, float)) or self.temperature < 0:
            self.temperature = 0.0
        elif self.temperature > 2.0:
            self.temperature = 2.0

        # Validate max_tokens
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            self.max_tokens = DEFAULT_MAX_TOKENS

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
                'timeout': 180.0,  # 3 minutes per request
            }

            if self.proxy:
                client_kwargs['http_client'] = httpx.Client(
                    proxy=self.proxy,
                    timeout=httpx.Timeout(180.0, connect=30.0)
                )

            self._client = OpenAI(**client_kwargs)

        return self._client

    def _build_system_prompt(self, target_language: str) -> str:
        """Build system prompt based on target language (simplified vs traditional Chinese)."""
        if target_language in ('zh-tw', 'Traditional Chinese'):
            return (
                "您是一位精通繁體中文的專業翻譯，"
                "您負責將它翻譯成中文，不要有任何解釋。"
            )
        return (
            "你是一位精通专业翻译的专家，"
            "你负责将它翻译成中文，不要有任何解释。"
        )

    def _build_single_prompt(
        self,
        text: str,
        target_language: str,
        context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build a two-step prompt for single-text translation.

        Args:
            text: Text to translate
            target_language: Target language name
            context: Optional context information

        Returns:
            List of message dictionaries for the API call
        """
        is_traditional = target_language in ('zh-tw', 'Traditional Chinese')

        system_message = {
            'role': 'system',
            'content': self._build_system_prompt(target_language)
        }

        if is_traditional:
            user_content = (
                f"請根據以下要求完成翻譯任務：\n"
                f"1. 將下面 YAML 對象裡的 source 字段直接翻譯為 {target_language}，"
                f"保留原文特定的術語或媒體名稱（如有）。"
                f"將本次翻譯的結果放入 YAML 陣列中的 step1 字段。\n"
                f"2. 根據第一次翻譯的結果進行意譯，力求信達雅，"
                f"但還是要保留特定的術語或媒體名稱（如有），"
                f"在遵守原意的前提下讓文本更通俗易懂，符合中文的表達習慣，"
                f"將第二次翻譯的結果放入 YAML 陣列中的 step2 字段。\n\n"
                f"示例格式:\n"
                f"  示例請求:\n"
                f"    - id: 1\n"
                f"      source: Source\n"
                f"  示例結果:\n"
                f"    - id: 1\n"
                f"      step1: 直譯結果\n"
                f"      step2: 意譯結果\n\n"
            )
        else:
            user_content = (
                f"请根据以下要求完成翻译任务：\n"
                f"1. 将下面 YAML 对象里的 source 字段直接翻译为 {target_language}，"
                f"保留原文特定的术语或媒体名称（如有）。"
                f"将本次翻译的结果放入 YAML 数组中的 step1 字段。\n"
                f"2. 根据第一次翻译的结果进行意译，力求信达雅，"
                f"但还是要保留特定的术语或媒体名称（如有），"
                f"在遵守原意的前提下让文本更通俗易懂，符合中文的表达习惯，"
                f"将第二次翻译的结果放入 YAML 数组中的 step2 字段。\n\n"
                f"示例格式:\n"
                f"  示例请求:\n"
                f"    - id: 1\n"
                f"      source: Source\n"
                f"  示例结果:\n"
                f"    - id: 1\n"
                f"      step1: 直译结果\n"
                f"      step2: 意译结果\n\n"
            )

        if context:
            user_content += f"Context: {context}\n\n"

        # Build YAML input
        yaml_input = f"- id: 1\n  source: {text}"
        user_content += f"开始翻译:\n\n{yaml_input}"

        return [
            system_message,
            {'role': 'user', 'content': user_content}
        ]

    def _build_srt_prompt(
        self,
        text: str,
        target_language: str,
        context: Optional[str] = None,
        glossary: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build the full SRT translation prompt using the loaded template.

        Args:
            text: Formatted SRT batch text (YAML format) to translate
            target_language: Target language name
            context: Optional context information
            glossary: Optional glossary/terminology

        Returns:
            List of message dictionaries for the API call
        """
        is_traditional = target_language in ('zh-tw', 'Traditional Chinese')

        system_message = {
            'role': 'system',
            'content': self._build_system_prompt(target_language)
        }

        context_block = f"Context: {context}" if context else ""
        glossary_block = f"Glossary: {glossary}" if glossary else ""
        system_prompt_extra = ""

        user_content = SRT_TRANSLATION_PROMPT.format(
            target_language=target_language,
            context_block=context_block,
            glossary_block=glossary_block,
            batch_input=text,
            system_prompt_extra=system_prompt_extra,
        )

        return [
            system_message,
            {'role': 'user', 'content': user_content}
        ]

    @retry(
        stop=stop_after_attempt(RETRY_NUMS),
        wait=wait_exponential(multiplier=1, min=RETRY_DELAY, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the OpenAI-compatible API with retry logic.

        Args:
            messages: List of message dictionaries

        Returns:
            Raw response text

        Raises:
            RuntimeError: If the API call fails or returns invalid response
            LengthFinishReasonError: If the response was truncated due to length
        """
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self._api_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                frequency_penalty=0,
                messages=messages
            )

            logger.debug(f'[LocalLLM] Response: {response}')

        except Exception as e:
            logger.error(f'[LocalLLM] API call failed: {e}')
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
                f"[LocalLLM] None content - finish_reason: "
                f"{response.choices[0].finish_reason}"
            )

        if not content or not content.strip():
            return ''

        return content.strip()

    @staticmethod
    def _parse_yaml_result(raw: str) -> List[Dict[str, str]]:
        """
        Parse YAML-formatted translation result with step1 and step2 fields.

        Expected format:
            - id: 1
              step1: 直译结果
              step2: 意译结果
            - id: 2
              step1: ...
              step2: ...

        Returns:
            List of dicts with keys: id, step1, step2
        """
        results = []
        # Try to extract YAML block from various wrapper formats
        text = raw.strip()

        # Remove possible markdown code fences
        text = re.sub(r'^```ya?ml\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```\s*$', '', text)

        # Try to extract from <TRANSLATE_TEXT> tags if present
        match = re.search(
            r'<TRANSLATE_TEXT>(.*?)</TRANSLATE_TEXT>',
            text, re.DOTALL | re.IGNORECASE
        )
        if match:
            text = match.group(1).strip()

        # Parse YAML-like blocks: each starts with "- id: N"
        # Split into individual items
        items = re.split(r'\n\s*-\s+id:', text)

        for item in items:
            item = item.strip()
            if not item:
                continue

            # Re-add "id:" prefix that was consumed by split
            item = "id:" + item

            # Extract id
            id_match = re.search(r'id:\s*(\d+)', item)
            item_id = int(id_match.group(1)) if id_match else len(results) + 1

            # Extract step1 - handle multi-line with proper indentation
            step1_match = re.search(
                r'step1:\s*(.+?)(?=\s+step2:|\s+id:|\s*$)',
                item, re.DOTALL
            )
            step1 = step1_match.group(1).strip() if step1_match else ""

            # Extract step2 - handle multi-line
            step2_match = re.search(
                r'step2:\s*(.+?)(?=\s+id:|\s*$)',
                item, re.DOTALL
            )
            step2 = step2_match.group(1).strip() if step2_match else ""

            results.append({
                'id': item_id,
                'step1': step1,
                'step2': step2,
            })

        return results

    def translate(
        self,
        text: str,
        target_language: str,
        context: Optional[str] = None
    ) -> str:
        """
        Translate a single text string using two-step translation.

        Step 1: Literal translation (直译)
        Step 2: Paraphrase/free translation (意译) - returned as final result

        Args:
            text: Text to translate
            target_language: Target language name (e.g., 'Simplified Chinese', 'English')
            context: Optional context information for better translation

        Returns:
            Paraphrased (意译) translation result

        Raises:
            ValueError: If text is empty or target_language is None
            RuntimeError: If translation fails after retries
        """
        if not text or not text.strip():
            return ''

        if target_language is None:
            raise ValueError("target_language cannot be None")

        messages = self._build_single_prompt(text, target_language, context)

        try:
            raw_result = self._call_api(messages)
        except RetryError as e:
            if e.last_attempt.exception():
                raise e.last_attempt.exception()
            raise

        # Parse YAML result
        parsed = self._parse_yaml_result(raw_result)

        if parsed and parsed[0].get('step2'):
            return parsed[0]['step2']

        # Fallback: if YAML parsing fails, try to extract any useful content
        # from the response (e.g., the model didn't follow YAML format)
        logger.warning(f"[LocalLLM] YAML parse failed for single translation, "
                       f"attempting fallback extraction")

        # Try extracting from TRANSLATE_TEXT tags
        match = re.search(
            r'<TRANSLATE_TEXT>(.*?)</TRANSLATE_TEXT>',
            raw_result, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()

        # Last resort: return raw result
        return raw_result

    def translate_batch(
        self,
        texts: List[str],
        target_language: str
    ) -> List[str]:
        """
        Translate a batch of texts using two-step YAML format.

        Args:
            texts: List of texts to translate
            target_language: Target language name

        Returns:
            List of paraphrased (意译) translations in the same order

        Raises:
            ValueError: If texts is not a list
        """
        if not isinstance(texts, list):
            raise ValueError("texts must be a list")

        if not texts:
            return []

        results = []
        for text in texts:
            if text and text.strip():
                result = self.translate(text, target_language)
                results.append(result)
            else:
                results.append(text or '')

        return results

    def translate_srt(
        self,
        srt_data: List[Dict],
        target_language: str,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """
        Translate SRT subtitle format while preserving timestamps.

        Uses two-step translation:
          Step 1: Literal translation (直译)
          Step 2: Paraphrase/free translation (意译)

        Translates in batches, with automatic fallback to individual
        translation on failure.

        Args:
            srt_data: List of subtitle dictionaries
            target_language: Target language name
            progress_callback: Optional fn(current, total, status)
            log_callback: Optional fn(level, message)

        Returns:
            List of subtitle dictionaries with translated text (step2/意译)
        """
        if not isinstance(srt_data, list):
            raise ValueError("srt_data must be a list")

        if not srt_data:
            return []

        def _log(level, msg):
            if log_callback:
                log_callback(level, msg)

        batch_size = max(1, self._config.get('batch_size', DEFAULT_BATCH_SIZE))

        # Use individual translation for tiny files
        if len(srt_data) <= 3:
            _log("INFO", f"Short file ({len(srt_data)} lines), translating individually")
            return self._translate_srt_simple(
                srt_data, target_language, progress_callback, log_callback
            )

        # Split into batches
        batches = [
            srt_data[i:i + batch_size]
            for i in range(0, len(srt_data), batch_size)
        ]
        total_batches = len(batches)
        _log("INFO", f"Batching {len(srt_data)} lines in "
             f"{total_batches} groups of {batch_size}")

        all_translated = []
        overall_start = time.time()
        for batch_idx, batch in enumerate(batches):
            batch_start_global = batch_idx * batch_size + 1
            batch_end_global = batch_start_global + len(batch) - 1

            if progress_callback:
                progress_callback(
                    batch_start_global, len(srt_data),
                    f"batch {batch_idx+1}/{total_batches} L{batch_start_global}-{batch_end_global}"
                )

            _log("INFO", f"Batch {batch_idx+1}/{total_batches}: "
                 f"lines {batch_start_global}-{batch_end_global} ({len(batch)} lines)")

            batch_t0 = time.time()
            try:
                batch_result = self._translate_batch_lines(
                    batch, target_language, log_callback
                )
                elapsed = time.time() - batch_t0
                all_translated.extend(batch_result)
                # Estimate remaining time
                done_count = batch_idx + 1
                avg_batch_time = (time.time() - overall_start) / done_count
                remaining_batches = total_batches - done_count
                eta = avg_batch_time * remaining_batches
                _log("SUCCESS", f"Batch {batch_idx+1} done ({elapsed:.1f}s, "
                     f"ETA: {eta/60:.1f}min remaining)")
            except Exception as e:
                _log("WARNING", f"Batch {batch_idx+1} failed ({e}), "
                     "falling back to individual translation")
                # Fall back one at a time for this batch
                for i, entry in enumerate(batch):
                    line_num = batch_start_global + i
                    if progress_callback:
                        progress_callback(
                            line_num, len(srt_data),
                            f"fallback L{line_num}"
                        )
                    text = entry.get('text', '').strip().replace("\n", " ")
                    if text:
                        try:
                            translated = self.translate(text, target_language)
                        except Exception:
                            translated = text  # Keep original on failure
                    else:
                        translated = text
                    _log("INFO", f"  L{line_num}: \"{text[:60]}\" -> \"{translated[:60]}\"")
                    all_translated.append({
                        'text': translated,
                        'time': entry.get('time', ''),
                        'line': entry.get('line', line_num)
                    })

        total_elapsed = time.time() - overall_start
        _log("INFO", f"Translation done: {len(all_translated)} lines in "
             f"{total_elapsed:.1f}s ({len(all_translated)/total_elapsed:.1f} lines/s)")
        return all_translated

    def _build_batch_yaml(self, batch: List[Dict]) -> str:
        """
        Build YAML input for batch translation from subtitle entries.

        Format:
            - id: 1
              source: First line text
            - id: 2
              source: Second line text
        """
        yaml_lines = []
        for j, entry in enumerate(batch):
            text = entry.get('text', '').strip().replace("\n", " ")
            yaml_lines.append(f"- id: {j+1}")
            yaml_lines.append(f"  source: {text}")
        return "\n".join(yaml_lines)

    def _translate_batch_lines(
        self,
        batch: List[Dict],
        target_language: str,
        log_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """Translate a batch of subtitle lines using YAML two-step format."""
        def _log(level, msg):
            if log_callback:
                log_callback(level, msg)

        # Build YAML input
        yaml_input = self._build_batch_yaml(batch)

        messages = self._build_batch_prompt(yaml_input, target_language)

        try:
            raw_result = self._call_api(messages)
        except LengthFinishReasonError:
            raise  # Let caller fall back to individual
        except RetryError as e:
            if e.last_attempt.exception():
                raise e.last_attempt.exception()
            raise

        # Parse YAML result
        parsed = self._parse_yaml_result(raw_result)

        if len(parsed) < len(batch) // 2:
            # Fallback: try numbered line parsing
            _log("WARNING", "YAML parse got too few results, trying numbered fallback")
            translations = self._parse_numbered_result(raw_result, len(batch))
            return self._reconstruct_batch(batch, translations, _log)

        # Log per-line results (step1 -> step2)
        for j, entry in enumerate(batch):
            src = entry.get('text', '').strip().replace("\n", " ")
            if j < len(parsed):
                s1 = parsed[j].get('step1', '')[:60]
                s2 = parsed[j].get('step2', '')[:60]
                _log("INFO", f"  L{entry.get('line', '?')}: "
                     f"直译=\"{s1}\" -> 意译=\"{s2}\"")
            else:
                _log("WARNING", f"  L{entry.get('line', '?')}: missing parsed result")

        # Reconstruct entries using step2 (意译) as final text
        result = []
        for j, entry in enumerate(batch):
            if j < len(parsed):
                final_text = parsed[j].get('step2', '') or parsed[j].get('step1', '')
                if not final_text:
                    final_text = entry.get('text', '')
            else:
                final_text = entry.get('text', '')
            result.append({
                'text': final_text,
                'time': entry.get('time', ''),
                'line': entry.get('line', 0)
            })
        return result

    def _reconstruct_batch(
        self,
        batch: List[Dict],
        translations: List[str],
        log_fn: Callable
    ) -> List[Dict]:
        """Reconstruct batch results from simple numbered translations."""
        for j, entry in enumerate(batch):
            src = entry.get('text', '').strip().replace("\n", " ")
            tgt = translations[j] if j < len(translations) else ''
            log_fn("INFO", f"  L{entry.get('line', '?')}: \"{src[:60]}\" -> \"{tgt[:60]}\"")

        result = []
        for j, entry in enumerate(batch):
            result.append({
                'text': translations[j] if j < len(translations) else entry.get('text', ''),
                'time': entry.get('time', ''),
                'line': entry.get('line', 0)
            })
        return result

    def _build_batch_prompt(
        self, yaml_text: str, target_language: str
    ) -> List[Dict[str, str]]:
        """Build a batch translation prompt using YAML format."""
        is_traditional = target_language in ('zh-tw', 'Traditional Chinese')

        system_message = {
            'role': 'system',
            'content': self._build_system_prompt(target_language)
        }

        if is_traditional:
            user_content = (
                f"請根據以下要求完成翻譯任務：\n"
                f"1. 將下面 YAML 對象裡的 source 字段直接翻譯為 {target_language}，"
                f"保留原文特定的術語或媒體名稱（如有）。"
                f"將本次翻譯的結果放入 YAML 陣列中的 step1 字段。\n"
                f"2. 根據第一次翻譯的結果進行意譯，力求信達雅，"
                f"但還是要保留特定的術語或媒體名稱（如有），"
                f"在遵守原意的前提下讓文本更通俗易懂，符合中文的表達習慣，"
                f"將第二次翻譯的結果放入 YAML 陣列中的 step2 字段。\n\n"
                f"示例格式:\n"
                f"  示例請求:\n"
                f"    - id: 1\n"
                f"      source: Source\n"
                f"  示例結果:\n"
                f"    - id: 1\n"
                f"      step1: 直譯結果\n"
                f"      step2: 意譯結果\n\n"
                f"開始翻譯:\n\n{yaml_text}"
            )
        else:
            user_content = (
                f"请根据以下要求完成翻译任务：\n"
                f"1. 将下面 YAML 对象里的 source 字段直接翻译为 {target_language}，"
                f"保留原文特定的术语或媒体名称（如有）。"
                f"将本次翻译的结果放入 YAML 数组中的 step1 字段。\n"
                f"2. 根据第一次翻译的结果进行意译，力求信达雅，"
                f"但还是要保留特定的术语或媒体名称（如有），"
                f"在遵守原意的前提下让文本更通俗易懂，符合中文的表达习惯，"
                f"将第二次翻译的结果放入 YAML 数组中的 step2 字段。\n\n"
                f"示例格式:\n"
                f"  示例请求:\n"
                f"    - id: 1\n"
                f"      source: Source\n"
                f"  示例结果:\n"
                f"    - id: 1\n"
                f"      step1: 直译结果\n"
                f"      step2: 意译结果\n\n"
                f"开始翻译:\n\n{yaml_text}"
            )

        return [
            system_message,
            {'role': 'user', 'content': user_content}
        ]

    def _translate_srt_simple(
        self,
        srt_data: List[Dict],
        target_language: str,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """Translate short SRT files line by line using single-text translation."""
        def _log(level, msg):
            if log_callback:
                log_callback(level, msg)

        results = []
        for i, entry in enumerate(srt_data):
            if progress_callback:
                progress_callback(i + 1, len(srt_data), f"line {i+1}")

            text = entry.get('text', '').strip().replace("\n", " ")
            if text:
                try:
                    translated = self.translate(text, target_language)
                except Exception as e:
                    _log("WARNING", f"Line {i+1} failed: {e}, keeping original")
                    translated = text
            else:
                translated = text

            _log("INFO", f"  L{i+1}: \"{text[:60]}\" -> \"{translated[:60]}\"")
            results.append({
                'text': translated,
                'time': entry.get('time', ''),
                'line': entry.get('line', i + 1)
            })

        return results

    @staticmethod
    def _parse_numbered_result(raw: str, expected_count: int) -> List[str]:
        """Parse numbered translation response as fallback."""
        # Match "N. text" or "N) text" patterns
        pattern = re.compile(r'^\s*(\d+)[\.\)\)]\s*(.+)', re.MULTILINE)
        matches = pattern.findall(raw)

        if matches and len(matches) >= expected_count // 2:
            parsed = {}
            for num_str, text in matches:
                idx = int(num_str)
                if 1 <= idx <= expected_count:
                    parsed[idx] = text.strip()
            if len(parsed) >= expected_count // 2:
                return [parsed.get(i + 1, '') for i in range(expected_count)]

        # Fallback: clean split by newlines
        lines = [l.strip() for l in raw.split('\n') if l.strip()]
        cleaned = []
        for line in lines:
            cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            if cleaned_line:
                cleaned.append(cleaned_line)
        if len(cleaned) >= expected_count:
            return cleaned[:expected_count]
        return cleaned + [''] * (expected_count - len(cleaned))

    def __repr__(self) -> str:
        """String representation of the translator."""
        return (
            f"LocalLLMTranslator("
            f"model='{self.model}', "
            f"api_url='{self.api_url}', "
            f"max_tokens={self.max_tokens}, "
            f"temperature={self.temperature})"
        )
