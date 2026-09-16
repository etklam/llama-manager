"""Single-owner transport retry policy for OpenAI-compatible requests.

Two retry owners used to multiply: the OpenAI SDK retried every request
twice by default (``max_retries=2``) and a tenacity decorator around the
adapter retried the whole thing three more times, so one logical request
could cost nine HTTP attempts. This module is now the only retry owner.
The SDK client is built with ``max_retries=0`` and every transport-level
decision lives here:

- Typed error classification: authentication, permission, configuration,
  invalid-model and unsupported-parameter failures fail immediately;
  connection, timeout, 429 and 5xx failures are retryable.
- Bounded attempts and an elapsed-time budget.
- Cancel-aware backoff with bounded ``Retry-After`` handling.
- Diagnostics per HTTP attempt (status, error type, elapsed) without
  request bodies, credentials, or response payloads.
"""
from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import httpx
import openai

logger = logging.getLogger(__name__)

# One logical request costs at most MAX_ATTEMPTS HTTP attempts, and the
# retries together must finish within RETRY_BUDGET_SECONDS.
MAX_ATTEMPTS = 3
RETRY_BUDGET_SECONDS = 30.0

# Exponential backoff between transport attempts.
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 10.0

# A server-provided Retry-After is honored, but never beyond this cap: a
# provider asking for minutes of silence is a provider that is down.
RETRY_AFTER_MAX_SECONDS = 30.0


class ErrorKind(enum.Enum):
    """What the transport should do about one failure."""

    FATAL = 'fatal'                  # never retried; aborts the run
    RETRYABLE = 'retryable'          # transient; retry within the budget
    CONTEXT_LENGTH = 'context_length'  # request too large; caller must shrink it


# Markers matched against provider error text. llama-server reports an
# oversized prompt as HTTP 500 "prompt is too long..." — a 5xx that would
# otherwise be retried identically and fail identically three times.
CONTEXT_LENGTH_MARKERS = (
    'context limit', 'context length', 'context window',
    'too many tokens', 'n_ctx', 'prompt is too long',
)


class TransportCancelledError(RuntimeError):
    """The run's cancellation token fired before or during a dispatch."""


class FatalProviderError(RuntimeError):
    """The provider rejected the request in a way retries cannot fix.

    Authentication, permission, configuration, invalid-model and
    unsupported-parameter failures carry this type so callers abort the
    run instead of re-asking per subtitle cue.
    """


class ProviderUnavailableError(RuntimeError):
    """Transport retries are exhausted or the elapsed budget is spent."""


class ContextLengthError(RuntimeError):
    """The request exceeds the model's context window.

    Retrying the identical request cannot help; the caller must apply a
    bounded request modification (drop context, split the batch).
    """


def _context_length_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in CONTEXT_LENGTH_MARKERS)


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Read a numeric Retry-After header, ignoring HTTP-date forms."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if seconds < 0:
        return None
    return min(seconds, RETRY_AFTER_MAX_SECONDS)


def classify(exc: BaseException) -> Tuple[ErrorKind, Optional[int], Optional[float]]:
    """Classify one exception into (kind, http status, retry-after seconds).

    Unknown exception types are FATAL by design: a typed policy that
    retried everything unexpected would reintroduce blind amplification.
    """
    status = getattr(exc, 'status_code', None)
    retry_after = None
    response = getattr(exc, 'response', None)
    headers = getattr(response, 'headers', None)
    if headers is not None:
        try:
            retry_after = _parse_retry_after(headers.get('retry-after'))
        except Exception:
            retry_after = None

    # An oversized prompt is never transient, whatever status reported it.
    if isinstance(exc, openai.LengthFinishReasonError) or _context_length_marker(str(exc)):
        return ErrorKind.CONTEXT_LENGTH, status, retry_after
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError,
                        openai.NotFoundError, openai.UnprocessableEntityError,
                        openai.BadRequestError)):
        return ErrorKind.FATAL, status, retry_after
    if isinstance(exc, (openai.RateLimitError, openai.InternalServerError)):
        return ErrorKind.RETRYABLE, status, retry_after
    if isinstance(exc, openai.APIStatusError):
        kind = ErrorKind.RETRYABLE if isinstance(status, int) and status >= 500 else ErrorKind.FATAL
        return kind, status, retry_after
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return ErrorKind.RETRYABLE, None, None
    return ErrorKind.FATAL, status, retry_after


@dataclass
class TransportStats:
    """Request counters shared between the adapter and its run log."""

    logical_requests: int = 0
    http_attempts: int = 0
    retries: int = 0

    def as_log_dict(self) -> Dict[str, int]:
        return {
            'logical_requests': self.logical_requests,
            'http_attempts': self.http_attempts,
            'retries': self.retries,
        }


def _default_note(event: Dict[str, Any]) -> None:
    level = event.pop('level', 'DEBUG')
    logger.log(
        logging.DEBUG if level == 'DEBUG' else logging.WARNING,
        '[transport] %s', event,
    )


def request_with_retry(
    dispatch: Callable[[], Any],
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    stats: Optional[TransportStats] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    max_attempts: int = MAX_ATTEMPTS,
    budget_seconds: float = RETRY_BUDGET_SECONDS,
) -> Any:
    """Run one logical request with bounded, cancel-aware transport retries.

    The cancellation token is checked before every HTTP dispatch and before
    and after every backoff wait, so a Stop that lands mid-backoff prevents
    the next attempt instead of sending it. A synchronous request already
    in flight still returns or times out on its own; that is documented
    behavior, not a promise of instant interruption.

    Raises:
        TransportCancelledError: the token fired before an attempt.
        FatalProviderError: the provider rejected the request for good.
        ContextLengthError: the request does not fit the context window.
        ProviderUnavailableError: attempts or budget exhausted.
    """
    note = on_event or _default_note
    started = clock()

    def cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    attempt = 0
    while True:
        if cancelled():
            raise TransportCancelledError('transport dispatch cancelled')
        attempt += 1
        if stats is not None:
            stats.http_attempts += 1
        try:
            return dispatch()
        except openai.LengthFinishReasonError:
            # The SDK raises before returning a completion whose finish
            # reason is "length"; the adapter normalizes it outside this
            # loop, and it is never a transport failure.
            raise
        except Exception as exc:
            kind, status, retry_after = classify(exc)
            if kind is ErrorKind.CONTEXT_LENGTH:
                note({'level': 'WARNING', 'attempt': attempt,
                      'result': 'context-length rejection', 'status': status})
                raise ContextLengthError(str(exc)) from exc
            if kind is ErrorKind.FATAL:
                note({'level': 'WARNING', 'attempt': attempt,
                      'result': 'fatal provider rejection',
                      'error_type': type(exc).__name__, 'status': status})
                raise FatalProviderError(
                    f'{type(exc).__name__} (status {status}): {exc}'
                ) from exc

            elapsed = clock() - started
            if attempt >= max_attempts:
                note({'level': 'WARNING', 'attempt': attempt,
                      'result': 'retry attempts exhausted',
                      'error_type': type(exc).__name__, 'status': status,
                      'elapsed': round(elapsed, 2)})
                raise ProviderUnavailableError(
                    f'provider unreachable after {attempt} attempt(s) '
                    f'({type(exc).__name__}, status {status})'
                ) from exc

            backoff = min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                          BACKOFF_MAX_SECONDS)
            if retry_after is not None:
                backoff = max(backoff, retry_after)
            if elapsed + backoff >= budget_seconds:
                # Sleeping the whole remaining budget (or more) leaves no
                # time for the next attempt, so the budget is already gone.
                note({'level': 'WARNING', 'attempt': attempt,
                      'result': 'retry budget exhausted',
                      'error_type': type(exc).__name__, 'status': status,
                      'elapsed': round(elapsed, 2)})
                raise ProviderUnavailableError(
                    f'provider retry budget ({budget_seconds:.0f}s) exhausted '
                    f'after {attempt} attempt(s) ({type(exc).__name__})'
                ) from exc

            if stats is not None:
                stats.retries += 1
            note({'level': 'INFO', 'attempt': attempt,
                  'result': 'retryable failure; backing off',
                  'error_type': type(exc).__name__, 'status': status,
                  'backoff_seconds': round(backoff, 2),
                  'elapsed': round(elapsed, 2)})
            sleeper(backoff)
            if cancelled():
                raise TransportCancelledError(
                    'transport dispatch cancelled during backoff')
