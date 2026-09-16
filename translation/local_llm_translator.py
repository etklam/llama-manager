# -*- coding: utf-8 -*-
"""
Local-LLM Translator Module

Provides translation functionality using local LLM models via OpenAI-compatible API endpoints.
This module is adapted from the pyvideotrans project and simplified for use in llama-manager.

Uses a two-step translation approach (inspired by ImmersiveTranslate Paraphrase Expert):
  Step 1: Literal translation (直译)
  Step 2: Paraphrase/free translation (意译) - this is the final output
"""

import json
import logging
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Dict, Optional, TYPE_CHECKING

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
    estimate_messages_tokens,
    request_fits,
)
from translation.transport import (
    ContextLengthError,
    FatalProviderError,
    ProviderUnavailableError,
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


class TxtTooLargeError(RuntimeError):
    """Full-text input cannot fit the context window.

    Oversized TXT is rejected explicitly instead of being silently
    truncated; an unbounded long-document implementation is out of scope
    for this hardening phase.
    """


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
        # The active run's cancellation token, checked before every HTTP
        # dispatch. Set by translate_srt/translate_full_text for the run's
        # lifetime only; None between runs.
        self._run_cancel_check: Optional[Callable[[], bool]] = None

    def close(self) -> None:
        """Close the HTTP client this translator created.

        Injected clients are never closed here: their owner decides their
        lifetime. Idempotent.
        """
        if self._openai_client is not None:
            self._openai_client.close()
            self._openai_client = None

    @contextmanager
    def _cancellation_scope(self, cancel_check: Optional[Callable[[], bool]]):
        """Bind the run's cancellation token to this translator and client.

        The token is checked before every HTTP dispatch and between
        transport retries (the transport policy reads it off the adapter),
        and by the translator itself before batch dispatches, context-limit
        splits, and per-line recovery. Restored to the previous state on
        exit so a reused translator never carries a stale token.
        """
        previous_run_token = self._run_cancel_check
        self._run_cancel_check = cancel_check
        client = self._get_llm_client()
        supports_token = hasattr(client, 'cancel_check')
        previous_client_token = (getattr(client, 'cancel_check', None)
                                 if supports_token else None)
        if supports_token:
            client.cancel_check = cancel_check
        try:
            yield
        finally:
            # Restore, not clear: translate_srt recurses through itself for
            # deduplication, and the outer run's token must survive the
            # inner scope's exit.
            self._run_cancel_check = previous_run_token
            if supports_token:
                client.cancel_check = previous_client_token

    def _check_run_cancelled(self) -> None:
        """Raise if the active run's token fired. No-op between runs."""
        if self._run_cancel_check is not None and self._run_cancel_check():
            raise TranslationCancelledError(
                'translation cancelled before request dispatch')

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

        Retry semantics belong to the client adapter's transport policy
        (translation.transport): bounded, classified, cancel-aware retries
        with no second owner here. The run's cancellation token is checked
        before the dispatch so a Stop that landed between requests prevents
        this one from ever leaving.

        Args:
            messages: List of message dictionaries
            max_tokens: Per-request token cap; falls back to self.max_tokens
                (the user-configured ceiling) when not supplied.

        Returns:
            Raw response text

        Raises:
            TranslationCancelledError: The run token fired before dispatch
            RuntimeError: If the API call fails or returns invalid response
            FatalProviderError / ProviderUnavailableError: The provider
                rejected the request or stayed unreachable after retries
        """
        # Checked on the worker thread right before the request leaves, so
        # cancellation has a boundary at every HTTP dispatch.
        self._check_run_cancelled()

        # Use the LLMClient port (injected or created)
        client = self._get_llm_client()

        try:
            response = client.complete(
                messages=messages,
                model=self._api_model,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                temperature=self.temperature
            )
        except Exception as e:
            logger.error(f'[LocalLLM] API call failed: {type(e).__name__}: {e}')
            raise

        # Log size only: full response text is sensitive source material.
        if response:
            logger.debug('[LocalLLM] Response: %d chars', len(response))

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

            # Extract id (optional - the model may omit it). The fallback id
            # stays positional for single-entry callers, but the flag records
            # whether the model actually echoed one: batch mapping must not
            # trust a position-derived id (see _translate_batch_lines).
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
                'explicit_id': bool(id_match),
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

    def translate_full_text(
        self,
        text: str,
        target_language: str,
        cancel_check: Optional[Callable[[], bool]] = None,
        log_callback: Optional[Callable] = None,
    ) -> str:
        """Translate one full text document (TXT path) in a single request.

        Full-text budgets and completeness are deliberately separate from
        the short-subtitle recovery path: a document is one request, sized
        against the context window up front, and a response that stopped at
        the token limit is incomplete output, never a completed file.

        Raises:
            TxtTooLargeError: the request cannot fit the context window.
                Oversized documents are rejected explicitly; an unbounded
                long-document implementation is out of scope here.
            TranslationIncompleteError: the model stopped at the token
                limit before finishing, so nothing may be published.
        """
        def _log(level, msg):
            if log_callback:
                log_callback(level, msg)

        if not text or not text.strip():
            return ''
        if target_language is None:
            raise ValueError("target_language cannot be None")

        messages = self._build_single_prompt(text, target_language, None)
        # A document is one request with an output budget sized to its
        # source: a translation runs about as long as its input, so the
        # configured ceiling only caps genuinely huge documents. The
        # finish-reason completeness check below is what protects content,
        # not a generous token budget.
        output_tokens = min(
            self.max_tokens,
            max(MIN_DYNAMIC_MAX_TOKENS, estimate_messages_tokens(messages)),
        )
        if not request_fits(messages, self.context_size, output_tokens):
            raise TxtTooLargeError(
                "TXT exceeds the model's context window "
                f"(estimated request + {output_tokens} output tokens does "
                "not fit); long-document translation is not supported — "
                "split the file instead"
            )

        client = self._get_llm_client()

        # Prefer metadata so a token-limit stop can be told apart from a
        # natural finish; fall back to the plain port for other clients.
        metadata_method = getattr(type(client), 'complete_with_metadata', None)
        with self._cancellation_scope(cancel_check):
            self._check_run_cancelled()
            if callable(metadata_method):
                result = client.complete_with_metadata(
                    messages=messages, model=self._api_model,
                    max_tokens=output_tokens, temperature=self.temperature,
                )
                raw_result, finish_reason = result.content, result.finish_reason
            else:
                raw_result = client.complete(
                    messages=messages, model=self._api_model,
                    max_tokens=output_tokens, temperature=self.temperature,
                )
                finish_reason = None

        if not raw_result or not raw_result.strip():
            raise RuntimeError("model returned an empty document translation")

        if finish_reason == 'length':
            # The document stopped mid-generation. Unlike a subtitle batch,
            # there is no per-cue recovery that could salvage the tail, so
            # publishing any part of it would present a truncated document
            # as a complete translation.
            _log("ERROR", "TXT translation stopped at the max-token limit; "
                          "output not saved")
            raise TranslationIncompleteError([
                (0, 'document translation stopped at the max-token limit')
            ])

        parsed = self._parse_yaml_result(raw_result)
        if parsed and parsed[0].get('step2'):
            return parsed[0]['step2']
        if parsed and parsed[0].get('step1'):
            return parsed[0]['step1']
        match = re.search(
            r'<TRANSLATE_TEXT>(.*?)</TRANSLATE_TEXT>',
            raw_result, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
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

        The run's cancellation token is bound to the translator and its HTTP
        client for the duration of the call, so every dispatch, transport
        retry, split and recovery below can observe it.

        Args:
            srt_data: List of Cue objects (see utils.srt_parser)
            target_language: Target language name
            progress_callback: Optional fn(current, total, status)
            log_callback: Optional fn(level, message)

        Returns:
            List of Cue objects with translated text (step2/意译)
        """
        if not srt_data:
            # Return before binding a run scope: no client needs to exist
            # for an empty input.
            return []
        with self._cancellation_scope(cancel_callback):
            return self._translate_srt_run(
                srt_data, target_language, progress_callback, log_callback,
                cancel_callback, _deduplicate)

    def _translate_srt_run(
        self,
        srt_data: List[Cue],
        target_language: str,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
        _deduplicate: bool = True
    ) -> List[Cue]:
        """translate_srt's implementation; see there for the contract."""
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
        # A fatal or persistent provider failure (auth, invalid model,
        # retries exhausted) stops dispatching the same way: expanding it
        # into N per-cue requests would only multiply identical failures.
        batch_results: Dict[int, List[Cue]] = {}
        batch_errors: Dict[int, Exception] = {}
        fatal_error: Optional[Exception] = None
        done_lines = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            next_batch_idx = 0

            def submit_next():
                nonlocal next_batch_idx
                if (next_batch_idx >= len(batch_plan) or _cancelled()
                        or fatal_error is not None):
                    return False
                batch_idx = next_batch_idx
                _, batch, context, recovery_contexts = batch_plan[batch_idx]
                future = executor.submit(
                    self._translate_batch_lines,
                    batch, target_language, log_callback, context,
                    recovery_contexts, _cancelled, progress_callback,
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
                        if isinstance(e, TranslationCancelledError):
                            pass  # the loop exit below converts this into a run cancel
                        elif isinstance(e, (FatalProviderError,
                                            ProviderUnavailableError)):
                            if fatal_error is None:
                                fatal_error = e
                                _log("ERROR", f"Batch {batch_idx+1} provider "
                                     f"failure ({e}); aborting the run")
                        elif not isinstance(e, TranslationIncompleteError):
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

            if fatal_error is not None:
                raise fatal_error
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
                        recovery_contexts, _cancelled, progress_callback))
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
        request_note = self._request_count_note()
        _log("INFO", f"Translation done: {len(all_translated)} lines in "
             f"{total_elapsed:.1f}s ({rate:.1f} lines/s){request_note}")
        return all_translated

    def _request_count_note(self) -> str:
        """Request-count suffix for run summaries, when the client tracks them."""
        stats = getattr(self._get_llm_client(), 'stats', None)
        if stats is None:
            return ''
        return (f" · {stats.logical_requests} logical / "
                f"{stats.http_attempts} HTTP request(s)")

    @staticmethod
    def _map_batch_ids(parsed: List[Dict[str, str]], batch_size: int, log_fn
                       ) -> Dict[int, Dict[str, str]]:
        """Decide which parsed entries can be trusted, and onto which line.

        Rules, applied in order:

        - Explicit ids are kept when unique and within 1..batch_size. A
          duplicate keeps its first entry; later copies are dropped with a
          warning. Out-of-range ids are dropped with a warning.
        - A response with no ids at all is mapped positionally only when it
          has exactly batch_size entries — anything shorter may have dropped
          any line, so guessing would shift translations onto wrong cues.
        - A mixed response (some ids, some not) keeps only its explicit
          entries: an id-less entry's position means nothing once any
          sibling carried an id.
        """
        by_id: Dict[int, Dict[str, str]] = {}
        explicit = [e for e in parsed if e.get('explicit_id')]
        implicit = [e for e in parsed if not e.get('explicit_id')]

        for entry in explicit:
            pid = entry.get('id')
            if not isinstance(pid, int) or not (1 <= pid <= batch_size):
                log_fn("WARNING", f"  model returned out-of-range id {pid}; "
                                 "ignoring that entry")
                continue
            if pid in by_id:
                log_fn("WARNING", f"  model returned duplicate id {pid}; "
                                 "keeping the first entry")
                continue
            by_id[pid] = entry

        if explicit and implicit:
            log_fn("WARNING",
                   f"model returned {len(implicit)} id-less entries mixed with "
                   f"explicit ids; only explicit mappings are trusted")
        elif implicit and not explicit:
            if len(implicit) == batch_size:
                for position, entry in enumerate(implicit, 1):
                    by_id.setdefault(position, entry)
            else:
                log_fn("WARNING",
                       f"model returned {len(implicit)} id-less entries for "
                       f"{batch_size} lines; cannot map positionally, "
                       "recovering individually")
        return by_id

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
        progress_callback: Optional[Callable] = None,
        background_reduced: bool = False,
    ) -> List[Cue]:
        """Translate a batch of subtitle lines using YAML two-step format."""
        def _log(level, msg):
            if log_callback:
                log_callback(level, msg)

        # Covers the initial dispatch and every recursive context-limit
        # split: a cancellation between requests must prevent the split's
        # second HTTP call, not just the next batch's first one.
        if cancel_check and cancel_check():
            raise TranslationCancelledError(
                'translation cancelled before batch dispatch')

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
            if self._is_context_limit_error(error):
                reduced_context = self._without_nearby_context(context)
                if reduced_context != context:
                    return self._translate_batch_lines(
                        batch, target_language, log_callback, reduced_context,
                        recovery_contexts, cancel_check, progress_callback,
                        background_reduced,
                    )
            if self._is_context_limit_error(error) and len(batch) > 1:
                middle = len(batch) // 2
                left_contexts = recovery_contexts[:middle] if recovery_contexts else None
                right_contexts = recovery_contexts[middle:] if recovery_contexts else None
                return (
                    self._translate_batch_lines(
                        batch[:middle], target_language, log_callback, context,
                        left_contexts, cancel_check, progress_callback,
                        background_reduced,
                    )
                    + self._translate_batch_lines(
                        batch[middle:], target_language, log_callback, context,
                        right_contexts, cancel_check, progress_callback,
                        background_reduced,
                    )
                )
            if self._is_context_limit_error(error) and not background_reduced:
                reduced_context = self._compact_story_context(context)
                if reduced_context != context:
                    return self._translate_batch_lines(
                        batch, target_language, log_callback, reduced_context,
                        recovery_contexts, cancel_check, progress_callback, True,
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
                cancel_check, progress_callback)

        # Map parsed entries onto the batch. The batch YAML numbers sources
        # 1..N (see _build_batch_yaml), so the source at position j carries
        # id j+1 — but only ids the model actually echoed are trustworthy.
        # Inventing one from result position would silently accept a
        # reordered or short answer (e.g. ID-less B/C answering A/B/C), so
        # ambiguous output is treated as missing text and sent to targeted
        # recovery instead of being guessed onto the wrong subtitle.
        by_id = self._map_batch_ids(parsed, len(batch), _log)

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
            cancel_check, progress_callback)

    def _recover_missing(self, batch, translations, target_language, log_fn,
                         recovery_contexts=None, cancel_check=None,
                         progress_callback=None):
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
                if progress_callback:
                    progress_callback(
                        -1, 0, f"補譯 L{entry.line or j + 1}"
                    )
                try:
                    context = (recovery_contexts[j]
                               if recovery_contexts and j < len(recovery_contexts)
                               else None)
                    text = self.translate(src, target_language, context=context)
                    if not text or not text.strip():
                        raise RuntimeError("model returned an empty translation")
                    log_fn("INFO", f"  L{entry.line or j+1}: retried -> \"{text[:60]}\"")
                except (FatalProviderError, ProviderUnavailableError):
                    # Retrying the remaining lines against a provider that
                    # just rejected or dropped us would multiply identical
                    # failures — abort the run and let the user fix it.
                    raise
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
        if isinstance(error, ContextLengthError):
            return True
        text = str(error).lower()
        return any(marker in text for marker in (
            'context limit', 'context length', 'context window',
            'too many tokens', 'n_ctx', 'prompt is too long',
        ))

    @staticmethod
    def _without_nearby_context(context):
        if not context:
            return context
        try:
            payload = json.loads(context)
        except (TypeError, json.JSONDecodeError):
            return context
        if not isinstance(payload, dict) or not payload.get('nearby_source'):
            return context
        payload['nearby_source'] = []
        return json.dumps(payload, ensure_ascii=False, separators=(',', ':'))

    @staticmethod
    def _compact_story_context(context):
        """Make one schema-preserving final fallback for a minimum request."""
        if not context:
            return context
        try:
            payload = json.loads(context)
        except (TypeError, json.JSONDecodeError):
            return context
        story = payload.get('story_context') if isinstance(payload, dict) else None
        if not isinstance(story, dict):
            return context
        summary = story.get('summary', '')
        story['summary'] = summary[:128] if summary else ''
        for field in ('characters', 'glossary', 'tone', 'uncertainties'):
            value = story.get(field)
            if isinstance(value, list):
                story[field] = []
        compact = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        return compact if compact != context else context

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
