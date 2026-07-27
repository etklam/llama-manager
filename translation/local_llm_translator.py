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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Dict, Optional, TYPE_CHECKING

from openai import LengthFinishReasonError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)

# Import LLMClient port for type checking
if TYPE_CHECKING:
    from translation.llm_client import LLMClient

from translation.prompt_builder import build_translation_prompt
from utils.srt_parser import collapse_repeats

logger = logging.getLogger(__name__)

# Constants
RETRY_NUMS = 3
RETRY_DELAY = 1  # Initial delay for exponential backoff
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.3
DEFAULT_API_URL = "http://localhost:8080/v1"
DEFAULT_BATCH_SIZE = 15  # Larger batch reduces API call overhead
DEFAULT_MAX_WORKERS = 3

# Dynamic max_tokens budgeting. Subtitle lines are short, so a fixed
# max_tokens=16384 reserves far more KV than a batch ever needs and, on the
# rare runaway generation, lets the model ramble much longer before hitting
# the length stop. Instead we size the request to the batch: a per-line token
# budget times the line count, floored so tiny batches still have headroom and
# capped by the user's configured max_tokens.
PER_LINE_TOKEN_BUDGET = 160  # conservative tokens per subtitle line (one field)
MIN_DYNAMIC_MAX_TOKENS = 512


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

    def __init__(self, config: dict, client: Optional['LLMClient'] = None):
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
            client: Optional LLMClient instance for dependency injection.
                    If not provided, an OpenAIClient will be created from config.

        Raises:
            ValueError: If 'model' is not in config and client is not provided
        """
        # ponytail: config_helpers owns defaults; here we only validate the
        # 'model' presence and clamp the numeric trust-boundary inputs.
        if 'model' not in config and client is None:
            raise ValueError("Configuration must include 'model' parameter")

        self._config = config
        self._api_model = config['model']
        self.api_url = config['api_url']
        self.model = config['model']
        self.max_tokens = config['max_tokens']
        self.temperature = config['temperature']
        self.api_key = config.get('api_key', '')
        self.proxy = config.get('proxy', None)

        # Clamp temperature to [0, 2] at the trust boundary
        if not isinstance(self.temperature, (int, float)) or self.temperature < 0:
            self.temperature = 0.0
        elif self.temperature > 2.0:
            self.temperature = 2.0

        # Reject non-positive max_tokens at the trust boundary
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            self.max_tokens = DEFAULT_MAX_TOKENS

        self.single_step = bool(config['single_step'])
        self.max_workers = max(1, int(config['max_workers']))

        # Store injected client or create OpenAIClient lazily
        self._injected_client = client
        self._openai_client = None  # OpenAIClient instance (will be created lazily if needed)
        self._client = None

    def _get_llm_client(self) -> 'LLMClient':
        """
        Get the LLM client (injected or created from config).

        Returns:
            LLMClient instance (either injected or OpenAIClient created from config)
        """
        if self._injected_client is not None:
            return self._injected_client

        # Create OpenAIClient from config
        if self._openai_client is None:
            from translation.openai_client import OpenAIClient
            self._openai_client = OpenAIClient(
                api_url=self.api_url,
                model=self._api_model,
                api_key=self.api_key,
                proxy=self.proxy,
                timeout=180.0
            )

        return self._openai_client

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
        Build a prompt for single-text translation.

        In single-step mode: only requests 意译 (translation field).
        In two-step mode: requests step1 (直译) + step2 (意译).

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

        mode = 'single_step' if self.single_step else 'two_step'
        locale = 'traditional' if is_traditional else 'simplified'

        # Build YAML input
        yaml_input = f"- id: 1\n  source: {text}"

        user_content = build_translation_prompt(
            mode=mode,
            locale=locale,
            target_language=target_language,
            yaml_text=yaml_input,
        )

        if context:
            # Insert context before the "开始翻译" / "開始翻譯" line
            start_marker = '開始翻譯:' if is_traditional else '开始翻译:'
            user_content = user_content.replace(
                start_marker,
                f"Context: {context}\n\n{start_marker}"
            )

        return [
            system_message,
            {'role': 'user', 'content': user_content}
        ]

    def _dynamic_max_tokens(self, line_count: int) -> int:
        """Size max_tokens to the request instead of using a fixed ceiling.

        Budget = per-line tokens x line count, doubled in two-step mode
        (the model emits both step1 直译 and step2 意译). The result is floored
        at MIN_DYNAMIC_MAX_TOKENS so short batches keep headroom and capped at
        the user-configured self.max_tokens so this never raises the ceiling.
        """
        line_count = max(1, line_count)
        per_line = PER_LINE_TOKEN_BUDGET * (1 if self.single_step else 2)
        budget = line_count * per_line
        budget = max(MIN_DYNAMIC_MAX_TOKENS, budget)
        return min(budget, self.max_tokens)

    def _call_api(self, messages: List[Dict[str, str]],
                  max_tokens: Optional[int] = None) -> str:
        """
        Call the LLM API via the client port with retry logic.

        Args:
            messages: List of message dictionaries
            max_tokens: Per-request token cap; falls back to self.max_tokens
                (the user-configured ceiling) when not supplied.

        Returns:
            Raw response text

        Raises:
            RuntimeError: If the API call fails or returns invalid response
            LengthFinishReasonError: If the response was truncated due to length
        """
        # Use the LLMClient port (injected or created)
        client = self._get_llm_client()

        try:
            response = client.complete(
                messages=messages,
                model=self._api_model,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                temperature=self.temperature
            )

            logger.debug(f'[LocalLLM] Response: {response}')

        except RetryError as e:
            # Tenacity wraps exceptions in RetryError
            if e.last_attempt.exception():
                logger.error(f'[LocalLLM] API call failed after retries: {e.last_attempt.exception()}')
                raise e.last_attempt.exception()
            raise
        except Exception as e:
            logger.error(f'[LocalLLM] API call failed: {e}')
            raise

        # Validate response - response should be a string
        if not isinstance(response, str):
            raise RuntimeError(f'Invalid response type: {type(response)}')

        if not response or not response.strip():
            return ''

        return response.strip()

    @staticmethod
    def _parse_yaml_result(raw: str) -> List[Dict[str, str]]:
        """
        Parse YAML-formatted translation result.

        Supports both two-step (step1/step2) and single-step (translation) formats:
            - id: 1
              step1: 直译结果
              step2: 意译结果
            OR:
            - id: 1
              translation: 意译结果

        Returns:
            List of dicts with keys: id, step1, step2
            For single-step format, 'translation' is mapped to 'step2'.
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

        # Split into individual YAML list items on the "- " list marker.
        # We split on the list dash rather than "- id:" because models often
        # omit the id field; splitting on the dash keeps each entry isolated so
        # a value can't bleed into the next item's marker. Prepend a newline so
        # a leading "- " at the very start also becomes a split boundary.
        items = re.split(r'\n\s*-\s+', '\n' + text)

        for item in items:
            item = item.strip()
            if not item:
                continue

            # Extract id (optional - the model may omit it)
            id_match = re.search(r'id:\s*(\d+)', item)
            item_id = int(id_match.group(1)) if id_match else len(results) + 1

            # Extract step1 - stop at the next field label or end of item.
            step1_match = re.search(
                r'step1:\s*(.+?)(?=\n\s*(?:step2|translation|id)\s*:|$)',
                item, re.DOTALL
            )
            step1 = LocalLLMTranslator._clean_field_value(step1_match.group(1)) if step1_match else ""

            # Extract step2 - stop at the next field label or end of item.
            step2_match = re.search(
                r'step2:\s*(.+?)(?=\n\s*(?:id|translation)\s*:|$)',
                item, re.DOTALL
            )
            step2 = LocalLLMTranslator._clean_field_value(step2_match.group(1)) if step2_match else ""

            # Extract translation field (single-step mode)
            translation_match = re.search(
                r'translation:\s*(.+?)(?=\n\s*(?:id|step1|step2)\s*:|$)',
                item, re.DOTALL
            )
            translation = LocalLLMTranslator._clean_field_value(translation_match.group(1)) if translation_match else ""

            # For single-step responses, map translation -> step2
            if not step2 and translation:
                step2 = translation

            results.append({
                'id': item_id,
                'step1': step1,
                'step2': step2,
            })

        return results

    @staticmethod
    def _clean_field_value(value: str) -> str:
        """Strip leaked YAML markers from an extracted field value.

        Models sometimes emit malformed output where one entry's value bleeds
        into the next list item (e.g. "结果\n- translation: 下一句") or where a
        value is prefixed with a stray marker. This truncates at the first
        subsequent list/field marker and strips any leading marker or wrapping
        quotes so raw tokens like "- translation:" never reach the subtitle.
        """
        if not value:
            return ""

        # Truncate at the first subsequent list item or field marker that
        # appears on a new line (a sign the next entry leaked in).
        value = re.split(
            r'\n\s*-\s+|\n\s*(?:id|step1|step2|translation)\s*:',
            value, maxsplit=1
        )[0]

        # Leading markers and wrapping quotes can be nested in either order
        # (e.g. '"- translation: 你好"' vs '- "你好"'), so peel them off
        # iteratively until the value stops shrinking.
        while True:
            before = value
            value = value.strip()

            # Strip a leading list marker / field label left over from a bad split.
            value = re.sub(
                r'^\s*(?:-\s*)?(?:id\s*:\s*\d+\s*)?(?:step1|step2|translation)\s*:\s*',
                '', value
            )

            # Strip matching wrapping quotes the model may have added.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]

            if value == before:
                break

        return value.strip()

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
            raw_result = self._call_api(
                messages, max_tokens=self._dynamic_max_tokens(1)
            )
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
        log_callback: Optional[Callable] = None,
        _deduplicate: bool = True
    ) -> List[Dict]:
        """
        Translate SRT subtitle format while preserving timestamps.

        Supports two modes via config:
          - Two-step (default): step1 (直译) + step2 (意译)
          - Single-step (single_step=True): translation only (意译)

        Translates in batches with concurrent execution (ThreadPoolExecutor),
        with automatic fallback to individual translation on batch failure.

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

        # Collapse pathological repetition (e.g. "あ、あ、あ、..." x100) in each
        # cue's source before translating. Such runs burn tokens and can hang
        # the model. We copy each entry so the caller's dicts stay untouched and
        # only the text field is normalized; timestamps/line numbers are kept.
        collapsed_count = 0
        normalized: List[Dict] = []
        for entry in srt_data:
            text = entry.get('text', '')
            new_text = collapse_repeats(text)
            if new_text != text:
                collapsed_count += 1
                entry = {**entry, 'text': new_text}
            normalized.append(entry)
        if collapsed_count:
            _log("INFO", f"Collapsed repeated runs in {collapsed_count} line(s)")
        srt_data = normalized

        # Translate identical cues only once, then fan the result back out to
        # every original timestamp. Exact text after whitespace normalization is
        # the cache key; repetition compression above runs first, so pathological
        # variants such as 100x "あ、" also converge to the same short key.
        if _deduplicate and len(srt_data) > 1:
            unique_entries: List[Dict] = []
            unique_index: Dict[str, int] = {}
            entry_keys: List[str] = []

            for entry in srt_data:
                key = re.sub(r'\s+', ' ', entry.get('text', '')).strip()
                entry_keys.append(key)
                if key not in unique_index:
                    unique_index[key] = len(unique_entries)
                    unique_entries.append(entry)

            duplicate_count = len(srt_data) - len(unique_entries)
            if duplicate_count:
                _log(
                    "INFO",
                    f"Deduplicated {duplicate_count} repeated line(s): "
                    f"{len(srt_data)} -> {len(unique_entries)} LLM inputs"
                )

                def unique_progress(current, total, status):
                    if progress_callback:
                        ratio = min(current, total) / max(1, total)
                        original_current = min(
                            len(srt_data), round(ratio * len(srt_data))
                        )
                        progress_callback(
                            original_current, len(srt_data),
                            f"{status} · {len(unique_entries)} unique lines"
                        )

                unique_translated = self.translate_srt(
                    unique_entries,
                    target_language,
                    progress_callback=unique_progress,
                    log_callback=log_callback,
                    _deduplicate=False,
                )

                translated_by_key = {}
                for idx, entry in enumerate(unique_entries):
                    key = re.sub(r'\s+', ' ', entry.get('text', '')).strip()
                    if idx < len(unique_translated):
                        translated_by_key[key] = unique_translated[idx].get(
                            'text', entry.get('text', '')
                        )
                    else:
                        translated_by_key[key] = entry.get('text', '')

                return [
                    {
                        'text': translated_by_key.get(
                            key, entry.get('text', '')
                        ),
                        'time': entry.get('time', ''),
                        'line': entry.get('line', idx + 1),
                    }
                    for idx, (entry, key) in enumerate(zip(srt_data, entry_keys))
                ]

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
             f"{total_batches} groups of {batch_size} "
             f"(workers={self.max_workers}, single_step={self.single_step})")

        overall_start = time.time()

        # Submit all batches concurrently
        batch_results: Dict[int, List[Dict]] = {}
        batch_errors: Dict[int, Exception] = {}

        if self.max_workers <= 1:
            # Sequential mode (no concurrency)
            for batch_idx, batch in enumerate(batches):
                self._process_single_batch(
                    batch_idx, batch, batches, srt_data, target_language,
                    log_callback, progress_callback, batch_results, batch_errors
                )
        else:
            # Concurrent mode. log_callback is safe to pass through: the UI
            # marshals it via log_bus.emit -> root.after(0, ...) rather than
            # touching Tk directly, so per-line logs stay visible with workers>1.
            done_lines = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for batch_idx, batch in enumerate(batches):
                    future = executor.submit(
                        self._translate_batch_lines,
                        batch, target_language, log_callback
                    )
                    futures[future] = batch_idx

                for future in as_completed(futures):
                    batch_idx = futures[future]
                    batch = batches[batch_idx]
                    batch_start_global = batch_idx * batch_size + 1
                    batch_end_global = batch_start_global + len(batch) - 1
                    try:
                        batch_results[batch_idx] = future.result()
                        _log("SUCCESS", f"Batch {batch_idx+1} done "
                             f"({batch_start_global}-{batch_end_global})")
                    except Exception as e:
                        batch_errors[batch_idx] = e
                        _log("WARNING", f"Batch {batch_idx+1} failed ({e}), "
                             "falling back to individual translation")

                    # Report line-level progress as each batch completes so the
                    # UI progress bar / ETA advances smoothly, not once at the end.
                    done_lines += len(batch)
                    if progress_callback:
                        elapsed = time.time() - overall_start
                        rate = done_lines / elapsed if elapsed > 0 else 0.0
                        progress_callback(
                            done_lines, len(srt_data),
                            f"{rate:.1f} lines/s"
                        )

        # Reconstruct results in order, with fallback for failed batches
        all_translated = []
        for batch_idx, batch in enumerate(batches):
            batch_start_global = batch_idx * batch_size + 1

            if batch_idx in batch_results:
                all_translated.extend(batch_results[batch_idx])
            elif batch_idx in batch_errors:
                # Fall back to individual translation
                for i, entry in enumerate(batch):
                    line_num = batch_start_global + i
                    text = entry.get('text', '').strip().replace("\n", " ")
                    if text:
                        try:
                            translated = self.translate(text, target_language)
                        except Exception:
                            translated = text
                    else:
                        translated = text
                    _log("INFO", f"  L{line_num}: \"{text[:60]}\" -> \"{translated[:60]}\"")
                    all_translated.append({
                        'text': translated,
                        'time': entry.get('time', ''),
                        'line': entry.get('line', line_num)
                    })

        total_elapsed = time.time() - overall_start
        rate = len(all_translated) / total_elapsed if total_elapsed > 0 else float('inf')
        _log("INFO", f"Translation done: {len(all_translated)} lines in "
             f"{total_elapsed:.1f}s ({rate:.1f} lines/s)")
        return all_translated

    def _process_single_batch(
        self,
        batch_idx: int,
        batch: List[Dict],
        batches: List[List[Dict]],
        srt_data: List[Dict],
        target_language: str,
        log_callback: Optional[Callable],
        progress_callback: Optional[Callable],
        batch_results: Dict[int, List[Dict]],
        batch_errors: Dict[int, Exception]
    ):
        """Process a single batch sequentially (used when max_workers <= 1)."""
        batch_size = len(batch)
        batch_start_global = batch_idx * max(1, self._config.get('batch_size', DEFAULT_BATCH_SIZE)) + 1
        batch_end_global = batch_start_global + batch_size - 1

        def _log(level, msg):
            if log_callback:
                log_callback(level, msg)

        if progress_callback:
            progress_callback(
                batch_start_global, len(srt_data),
                f"batch {batch_idx+1}/{len(batches)} L{batch_start_global}-{batch_end_global}"
            )

        _log("INFO", f"Batch {batch_idx+1}/{len(batches)}: "
             f"lines {batch_start_global}-{batch_end_global} ({batch_size} lines)")

        batch_t0 = time.time()
        try:
            batch_result = self._translate_batch_lines(
                batch, target_language, log_callback
            )
            elapsed = time.time() - batch_t0
            batch_results[batch_idx] = batch_result
            _log("SUCCESS", f"Batch {batch_idx+1} done ({elapsed:.1f}s)")
        except Exception as e:
            batch_errors[batch_idx] = e

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
            raw_result = self._call_api(
                messages, max_tokens=self._dynamic_max_tokens(len(batch))
            )
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

        mode = 'single_step' if self.single_step else 'two_step'
        locale = 'traditional' if is_traditional else 'simplified'

        user_content = build_translation_prompt(
            mode=mode,
            locale=locale,
            target_language=target_language,
            yaml_text=yaml_text,
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
