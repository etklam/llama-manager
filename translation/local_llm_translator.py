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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Dict, Optional, TYPE_CHECKING

from tenacity import RetryError

# Import LLMClient port for type checking
if TYPE_CHECKING:
    from translation.llm_client import LLMClient

from translation.prompt_builder import (
    build_translation_prompt, build_system_prompt, build_source_yaml,
)
from translation.story_context import (
    StoryContextAnalyzer,
    StoryContextError,
    build_translation_context,
    request_fits,
)
from utils.srt_parser import Cue, collapse_repeats

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_TOKENS = 4096
DEFAULT_BATCH_SIZE = 15  # Larger batch reduces API call overhead
DEFAULT_MAX_WORKERS = 3

# Dynamic max_tokens budgeting. Subtitle lines are short, so a fixed
# max_tokens=16384 reserves far more KV than a batch ever needs and, on the
# rare runaway generation, lets the model ramble much longer before hitting
# the length stop. Instead we size the request to the batch: a per-line token
# budget times the line count, floored so tiny batches still have headroom and
# capped by the user's configured max_tokens.
PER_LINE_TOKEN_BUDGET = 160  # conservative tokens per subtitle line (one field)
# The floor covers 1-3 line requests, which is also where single-line retries
# and short files land. A chatty local model spends its first few hundred tokens
# on a preamble before emitting the YAML, so a floor that only just fits the
# translation itself truncates the answer before it starts. This is generous for
# one line on purpose: the cost of over-reserving is a slightly larger KV
# allocation, while the cost of under-reserving is a failed line.
MIN_DYNAMIC_MAX_TOKENS = 1024

# Soft ceiling for a single source cue on the translation path. A spoken
# subtitle line rarely runs past a couple hundred characters, so anything well
# beyond that is long-form text on the wrong path. Exceeding it only logs a
# warning — see translate_srt — never rejects or truncates the line.
LONG_SOURCE_WARN_CHARS = 500


class TranslationIncompleteError(RuntimeError):
    """A file has unresolved cues and must not be published as translated."""

    def __init__(self, failures):
        self.failures = failures
        details = '; '.join(f"L{line}: {reason}" for line, reason in failures)
        super().__init__(f"Translation incomplete ({len(failures)} line(s)); "
                         f"output not saved. {details}")


class TranslationCancelledError(RuntimeError):
    """A translation run was cancelled before it could be committed."""


def _rebuild_cue(entry: Cue, text: str, fallback_line: int = 0) -> Cue:
    """Build the translated cue: same timing, new text.

    Every reconstruction site in translate_srt collapses onto this one
    conversion. Only the text differs from the source cue; start/end times
    travel untouched, and the line number falls back to the caller's position
    when the source cue carried none.
    """
    return replace(entry, text=text, line=entry.line or fallback_line)


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
        self.context_mode = config.get('context_mode', 'none')
        if self.context_mode not in ('none', 'story'):
            self.context_mode = 'none'
        context_size = config.get('context_size')
        self.context_size = context_size if isinstance(context_size, int) and context_size > 0 else None

        # Store injected client or create OpenAIClient lazily
        self._injected_client = client
        self._openai_client = None  # OpenAIClient instance (will be created lazily if needed)

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
        return build_system_prompt(target_language)

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
        yaml_input = build_source_yaml([text])

        user_content = build_translation_prompt(
            mode=mode,
            locale=locale,
            target_language=target_language,
            yaml_text=yaml_input,
            context=context,
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
        Call the LLM API via the client port.

        Retry semantics belong to the client adapter: OpenAIClient's tenacity
        decorator owns the retry loop and, once exhausted, raises RetryError.
        This is the one place the translator unwraps that envelope so callers
        of the port see the underlying failure, not tenacity's wrapper.

        Args:
            messages: List of message dictionaries
            max_tokens: Per-request token cap; falls back to self.max_tokens
                (the user-configured ceiling) when not supplied.

        Returns:
            Raw response text

        Raises:
            RuntimeError: If the API call fails or returns invalid response
            LengthFinishReasonError: If the client reports a truncated
                response with no usable content (the adapter does not retry a
                length stop, so this propagates unwrapped)
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
            # Tenacity wraps the final failure in RetryError once the adapter
            # exhausts its retries; surface that underlying exception.
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
        text = re.sub(r'^\s*```[^\n]*$', '', text, flags=re.MULTILINE)

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
            id_match = re.search(r'^[ \t]*id:[ \t]*(\d+)', item, re.MULTILINE)
            item_id = int(id_match.group(1)) if id_match else len(results) + 1

            # Extract step1 - stop at the next field label or end of item.
            step1_match = re.search(
                r'(?:^|\n)[ \t]*step1:[ \t]*(.+?)(?=\n\s*(?:step2|translation|id)\s*:|$)',
                item, re.DOTALL
            )
            step1 = LocalLLMTranslator._clean_field_value(step1_match.group(1)) if step1_match else ""

            # Extract step2 - stop at the next field label or end of item.
            step2_match = re.search(
                r'(?:^|\n)[ \t]*step2:[ \t]*(.+?)(?=\n\s*(?:id|translation)\s*:|$)',
                item, re.DOTALL
            )
            step2 = LocalLLMTranslator._clean_field_value(step2_match.group(1)) if step2_match else ""

            # Extract translation field (single-step mode)
            translation_match = re.search(
                r'(?:^|\n)[ \t]*translation:[ \t]*(.+?)(?=\n\s*(?:id|step1|step2)\s*:|$)',
                item, re.DOTALL
            )
            translation = LocalLLMTranslator._clean_field_value(translation_match.group(1)) if translation_match else ""

            # For single-step responses, map translation -> step2
            if not step2 and translation:
                step2 = translation

            # Commentary before the YAML is not a subtitle. Giving it a
            # synthetic id used to shadow the real id=1 in the result map.
            if not id_match and not step1 and not step2:
                continue

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

        raw_result = self._call_api(
            messages, max_tokens=self._dynamic_max_tokens(1)
        )
        if not raw_result:
            raise RuntimeError("model returned an empty translation")

        # Parse YAML result
        parsed = self._parse_yaml_result(raw_result)

        if parsed and parsed[0].get('step2'):
            return parsed[0]['step2']
        if parsed:
            if parsed[0].get('step1'):
                return parsed[0]['step1']
            raise RuntimeError("model returned translation fields without usable text")

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
        srt_data: List[Cue],
        target_language: str,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
        _deduplicate: bool = True
    ) -> List[Cue]:
        """
        Translate SRT subtitle format while preserving timestamps.

        Supports two modes via config:
          - Two-step (default): step1 (直译) + step2 (意译)
          - Single-step (single_step=True): translation only (意译)

        Translates in batches with concurrent execution (ThreadPoolExecutor),
        with automatic fallback to individual translation on batch failure.

        Args:
            srt_data: List of Cue objects (see utils.srt_parser)
            target_language: Target language name
            progress_callback: Optional fn(current, total, status)
            log_callback: Optional fn(level, message)

        Returns:
            List of Cue objects with translated text (step2/意译)
        """
        if not isinstance(srt_data, list):
            raise ValueError("srt_data must be a list")

        if not srt_data:
            return []

        def _log(level, msg):
            if log_callback:
                log_callback(level, msg)

        def _cancelled():
            return bool(cancel_callback and cancel_callback())

        if _cancelled():
            raise TranslationCancelledError("translation cancelled")

        # Collapse pathological repetition (e.g. "あ、あ、あ、..." x100) in each
        # cue's source before translating. Such runs burn tokens and can hang
        # the model. We copy each Cue so the caller's objects stay untouched and
        # only the text field is normalized; timestamps/line numbers are kept.
        collapsed_count = 0
        normalized: List[Cue] = []
        for entry in srt_data:
            new_text = collapse_repeats(entry.text)
            if new_text != entry.text:
                collapsed_count += 1
                entry = replace(entry, text=new_text)
            normalized.append(entry)
        if collapsed_count:
            _log("INFO", f"Collapsed repeated runs in {collapsed_count} line(s)")
        srt_data = normalized

        # Story context is local to this invocation.  PipelineRunner deliberately
        # reuses translator/client instances across files, so storing it on self
        # would leak one file's facts into the next.
        story_context = None
        if self.context_mode == 'story':
            analysis_start = time.time()
            story_context = StoryContextAnalyzer(
                client=self._get_llm_client(),
                model=self._api_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                context_size=self.context_size,
                cancel_check=_cancelled,
                progress_callback=progress_callback,
                log_callback=log_callback,
            ).create(srt_data, target_language)
            _log("INFO", f"Story analysis done in {time.time() - analysis_start:.1f}s")
            _deduplicate = False
            if _cancelled():
                raise TranslationCancelledError("translation cancelled after story analysis")

        # Soft guard: this path is tuned for subtitle-sized cues, and the
        # translation server preset runs a correspondingly small context. An
        # over-long cue still gets translated — we only warn, because refusing
        # or truncating would silently lose subtitle content — but the warning
        # names the lines to look at when a result comes back mangled or when
        # the request trips the context limit. Long-form input belongs on the
        # chat endpoint with its larger context, not here. The guard is about
        # context size, not dedup: it fires whether or not the dedup pre-pass
        # runs, so it is not gated on _deduplicate.
        long_lines = [
            entry.line or (idx + 1)
            for idx, entry in enumerate(srt_data)
            if len(entry.text) > LONG_SOURCE_WARN_CHARS
        ]
        if long_lines:
            shown = ', '.join(f"L{n}" for n in long_lines[:5])
            if len(long_lines) > 5:
                shown += f", +{len(long_lines) - 5} more"
            _log("WARNING",
                 f"{len(long_lines)} line(s) exceed "
                 f"{LONG_SOURCE_WARN_CHARS} chars ({shown}); this path is "
                 "tuned for short context — use chat mode for long-form text")

        # Translate identical cues only once, then fan the result back out to
        # every original timestamp. Exact text after whitespace normalization is
        # the cache key; repetition compression above runs first, so pathological
        # variants such as 100x "あ、" also converge to the same short key.
        if _deduplicate and len(srt_data) > 1:
            unique_entries: List[Cue] = []
            unique_index: Dict[str, int] = {}
            entry_keys: List[str] = []

            for entry in srt_data:
                key = re.sub(r'\s+', ' ', entry.text).strip()
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
                    cancel_callback=cancel_callback,
                    _deduplicate=False,
                )

                translated_by_key = {}
                for idx, entry in enumerate(unique_entries):
                    key = re.sub(r'\s+', ' ', entry.text).strip()
                    if idx < len(unique_translated):
                        translated_by_key[key] = unique_translated[idx].text
                    else:
                        translated_by_key[key] = entry.text

                return [
                    _rebuild_cue(
                        entry,
                        translated_by_key.get(key, entry.text),
                        fallback_line=idx + 1,
                    )
                    for idx, (entry, key) in enumerate(zip(srt_data, entry_keys))
                ]

        # One batch core feeds every code path. There is no separate simple
        # path for tiny files and no sequential branch for max_workers <= 1:
        # both are just small pools over the same _translate_batch_lines
        # machinery, so batch reconstruction / by-id alignment exists exactly
        # once. A 1-worker pool is sequential by construction.
        workers = 1 if self.max_workers <= 1 else self.max_workers
        batch_size = max(1, self._config.get('batch_size', DEFAULT_BATCH_SIZE))

        # A plan item is (start index, batch, batch context, per-line recovery
        # contexts). Standard mode retains the existing fixed batching.
        if story_context is not None:
            batch_plan = self._plan_story_batches(
                srt_data, target_language, story_context, batch_size
            )
        else:
            batch_plan = [
                (i, srt_data[i:i + batch_size], None, None)
                for i in range(0, len(srt_data), batch_size)
            ]
        batches = [item[1] for item in batch_plan]
        total_batches = len(batches)
        _log("INFO", f"Batching {len(srt_data)} lines in "
             f"{total_batches} groups of {batch_size} "
             f"(workers={workers}, single_step={self.single_step})")

        overall_start = time.time()

        # Keep at most `workers` requests submitted.  This preserves concurrency
        # while giving Stop a request boundary at which no further batch starts.
        batch_results: Dict[int, List[Cue]] = {}
        batch_errors: Dict[int, Exception] = {}
        done_lines = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            next_batch_idx = 0

            def submit_next():
                nonlocal next_batch_idx
                if next_batch_idx >= len(batch_plan) or _cancelled():
                    return False
                batch_idx = next_batch_idx
                _, batch, context, recovery_contexts = batch_plan[batch_idx]
                future = executor.submit(
                    self._translate_batch_lines,
                    batch, target_language, log_callback, context,
                    recovery_contexts, _cancelled,
                )
                futures[future] = batch_idx
                next_batch_idx += 1
                return True

            while len(futures) < workers and submit_next():
                pass

            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    batch_idx = futures.pop(future)
                    batch_start, batch, _, _ = batch_plan[batch_idx]
                    batch_start_global = batch_start + 1
                    batch_end_global = batch_start_global + len(batch) - 1
                    try:
                        batch_results[batch_idx] = future.result()
                        _log("SUCCESS", f"Batch {batch_idx+1} done "
                             f"({batch_start_global}-{batch_end_global})")
                    except Exception as e:
                        batch_errors[batch_idx] = e
                        if not isinstance(e, TranslationIncompleteError):
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
                            f"翻譯 · {rate:.1f} lines/s"
                        )

                    while len(futures) < workers and submit_next():
                        pass

            if _cancelled():
                raise TranslationCancelledError("translation cancelled")

        # Reconstruct results in order, with fallback for failed batches
        all_translated = []
        failures = []
        for batch_idx, batch in enumerate(batches):
            if batch_idx in batch_results:
                all_translated.extend(batch_results[batch_idx])
            elif batch_idx in batch_errors:
                error = batch_errors[batch_idx]
                if isinstance(error, TranslationIncompleteError):
                    # Targeted recovery already ran inside this batch.
                    failures.extend(error.failures)
                    continue
                try:
                    _, _, _, recovery_contexts = batch_plan[batch_idx]
                    all_translated.extend(self._recover_missing(
                        batch, [''] * len(batch), target_language, _log,
                        recovery_contexts, _cancelled))
                except TranslationIncompleteError as error:
                    failures.extend(error.failures)

        if failures:
            error = TranslationIncompleteError(failures)
            _log("ERROR", str(error))
            raise error

        if _cancelled():
            raise TranslationCancelledError("translation cancelled before completion")

        total_elapsed = time.time() - overall_start
        rate = len(all_translated) / total_elapsed if total_elapsed > 0 else float('inf')
        _log("INFO", f"Translation done: {len(all_translated)} lines in "
             f"{total_elapsed:.1f}s ({rate:.1f} lines/s)")
        return all_translated

    def _build_batch_yaml(self, batch: List[Cue]) -> str:
        """
        Build YAML input for batch translation from subtitle cues.

        Format:
            - id: 1
              source: First line text
            - id: 2
              source: Second line text
        """
        return build_source_yaml([entry.text.strip() for entry in batch])

    def _translate_batch_lines(
        self,
        batch: List[Cue],
        target_language: str,
        log_callback: Optional[Callable] = None,
        context: Optional[str] = None,
        recovery_contexts: Optional[List[str]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> List[Cue]:
        """Translate a batch of subtitle lines using YAML two-step format."""
        def _log(level, msg):
            if log_callback:
                log_callback(level, msg)

        # Build YAML input
        yaml_input = self._build_batch_yaml(batch)

        messages = self._build_batch_prompt(yaml_input, target_language, context)

        # Any exception — including a LengthFinishReasonError the client
        # raised because nothing usable came back — propagates to the executor
        # loop, which falls back to per-line translation.
        try:
            raw_result = self._call_api(
                messages, max_tokens=self._dynamic_max_tokens(len(batch))
            )
        except Exception as error:
            if self._is_context_limit_error(error) and len(batch) > 1:
                middle = len(batch) // 2
                left_contexts = recovery_contexts[:middle] if recovery_contexts else None
                right_contexts = recovery_contexts[middle:] if recovery_contexts else None
                return (
                    self._translate_batch_lines(
                        batch[:middle], target_language, log_callback, context,
                        left_contexts, cancel_check,
                    )
                    + self._translate_batch_lines(
                        batch[middle:], target_language, log_callback, context,
                        right_contexts, cancel_check,
                    )
                )
            if self._is_context_limit_error(error):
                line = batch[0].line or 1
                raise TranslationIncompleteError([
                    (line, f"context limit: {error}")
                ]) from error
            raise

        # Parse YAML result
        parsed = self._parse_yaml_result(raw_result)

        if not parsed:
            # Fallback: try numbered line parsing
            _log("WARNING", "YAML parse got too few results, trying numbered fallback")
            translations = self._parse_numbered_result(raw_result, len(batch))
            return self._recover_missing(
                batch, translations, target_language, _log, recovery_contexts,
                cancel_check)

        # Map parsed entries by the id echoed back by the model. The batch YAML
        # numbers sources 1..N (see _build_batch_yaml), so the source at
        # position j carries id j+1. Reconstructing by id — not by list
        # position — means a reordered, dropped, or merged line in the model's
        # response can no longer shift every following translation onto the
        # wrong subtitle. That positional shift was exactly what surfaced as
        # lines "returning the original text": once one line was missing or out
        # of order, the tail ran off the end of `parsed` and fell back to the
        # untranslated source. First id wins so a duplicated id can't clobber a
        # real one.
        by_id: Dict[int, Dict[str, str]] = {}
        for parsed_entry in parsed:
            pid = parsed_entry.get('id')
            if isinstance(pid, int) and pid not in by_id:
                by_id[pid] = parsed_entry

        translations = []
        for j, entry in enumerate(batch):
            matched = by_id.get(j + 1)
            final_text = ''
            if matched is not None:
                final_text = matched.get('step2', '') or matched.get('step1', '')
                s1 = matched.get('step1', '')[:60]
                s2 = matched.get('step2', '')[:60]
                if self.single_step:
                    _log("INFO", f"  L{entry.line or '?'}: translation=\"{final_text[:60]}\"")
                else:
                    _log("INFO", f"  L{entry.line or '?'}: "
                         f"literal=\"{s1}\" -> paraphrase=\"{s2}\"")
            if not final_text:
                # The model returned no usable text for this line. Emitting the
                # source verbatim is precisely the "not translated" symptom, so
                # mark it for a targeted individual retry below.
                _log("WARNING", f"  L{entry.line or '?'}: missing parsed result")
            translations.append(final_text)

        return self._recover_missing(
            batch, translations, target_language, _log, recovery_contexts,
            cancel_check)

    def _recover_missing(self, batch, translations, target_language, log_fn,
                         recovery_contexts=None, cancel_check=None):
        """One recovery policy for YAML, numbered output and failed API batches."""
        result = []
        failures = []
        for j, entry in enumerate(batch):
            src = entry.text.strip().replace("\n", " ")
            text = translations[j] if j < len(translations) else ''
            if src and not text.strip():
                if cancel_check and cancel_check():
                    raise TranslationCancelledError(
                        "translation cancelled before recovery"
                    )
                try:
                    context = (recovery_contexts[j]
                               if recovery_contexts and j < len(recovery_contexts)
                               else None)
                    text = self.translate(src, target_language, context=context)
                    if not text or not text.strip():
                        raise RuntimeError("model returned an empty translation")
                    log_fn("INFO", f"  L{entry.line or j+1}: retried -> \"{text[:60]}\"")
                except Exception as error:
                    reason = f"{type(error).__name__}: {error}"
                    failures.append((entry.line or j + 1, reason))
                    log_fn("ERROR", f"  L{entry.line or j+1}: retry failed ({reason})")
                    continue
            result.append(_rebuild_cue(entry, text, fallback_line=j + 1))
        if failures:
            raise TranslationIncompleteError(failures)
        return result

    def _build_batch_prompt(
        self, yaml_text: str, target_language: str,
        context: Optional[str] = None,
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
            context=context,
        )

        return [
            system_message,
            {'role': 'user', 'content': user_content}
        ]

    def _plan_story_batches(self, srt_data, target_language, story_context,
                            requested_batch_size):
        """Plan batches that fit without ever truncating a requested source."""
        plan = []
        cursor = 0
        while cursor < len(srt_data):
            size = min(requested_batch_size, len(srt_data) - cursor)
            selected = None
            while size >= 1 and selected is None:
                batch = srt_data[cursor:cursor + size]
                for radius in (2, 1, 0):
                    nearby = self._nearby_cues(
                        srt_data, cursor, cursor + size, radius
                    )
                    context = build_translation_context(story_context, nearby)
                    messages = self._build_batch_prompt(
                        self._build_batch_yaml(batch), target_language, context
                    )
                    if request_fits(
                        messages, self.context_size,
                        self._dynamic_max_tokens(len(batch)),
                    ):
                        recovery_contexts = [
                            self._recovery_story_context(
                                srt_data, index, target_language, story_context,
                            )
                            for index in range(cursor, cursor + size)
                        ]
                        selected = (cursor, batch, context, recovery_contexts)
                        break
                if selected is None:
                    size //= 2
            if selected is None:
                line = srt_data[cursor].line or cursor + 1
                raise StoryContextError(
                    f"context window is too small to translate cue {line} without truncation"
                )
            plan.append(selected)
            cursor += len(selected[1])
        return plan

    def _recovery_story_context(self, srt_data, index, target_language,
                                story_context):
        """Choose the largest nearby window that leaves a full retry possible."""
        source = srt_data[index].text.strip().replace("\n", " ")
        for radius in (2, 1, 0):
            context = build_translation_context(
                story_context,
                self._nearby_cues(srt_data, index, index + 1, radius),
            )
            messages = self._build_single_prompt(source, target_language, context)
            if request_fits(
                messages, self.context_size, self._dynamic_max_tokens(1)
            ):
                return context
        line = srt_data[index].line or index + 1
        raise StoryContextError(
            f"context window is too small to recover cue {line} without truncation"
        )

    @staticmethod
    def _nearby_cues(srt_data, start, end, radius):
        if radius <= 0:
            return []
        before = range(max(0, start - radius), start)
        after = range(end, min(len(srt_data), end + radius))
        return [(index + 1, srt_data[index]) for index in (*before, *after)]

    @staticmethod
    def _is_context_limit_error(error):
        text = str(error).lower()
        return any(marker in text for marker in (
            'context limit', 'context length', 'context window',
            'too many tokens', 'n_ctx', 'prompt is too long',
        ))

    @staticmethod
    def _parse_numbered_result(raw: str, expected_count: int) -> List[str]:
        """Parse numbered translation response as fallback."""
        # Match "N. text" or "N) text" patterns
        pattern = re.compile(r'^\s*(\d+)[\.\)\)]\s*(.+)', re.MULTILINE)
        matches = pattern.findall(raw)

        if matches:
            parsed = {}
            for num_str, text in matches:
                idx = int(num_str)
                if 1 <= idx <= expected_count and idx not in parsed:
                    parsed[idx] = text.strip()
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
