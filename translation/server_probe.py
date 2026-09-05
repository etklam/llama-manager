"""Preflight probe for a llama-server instance.

Two problems this solves, both of which used to surface as "the translation is
slow" or "the translation hung":

  1. A translation run against a server that is not up spends its whole life in
     tenacity's retry/backoff loop — three attempts per batch, every batch — so
     a large SRT takes minutes to report a failure that is knowable in
     milliseconds. Probing once up front turns that into an immediate message.

  2. Client concurrency and server slots are configured independently. Asking
     for 3 workers against a server started with `--parallel 1` does not fail;
     the requests simply queue, so throughput matches one worker while the UI
     claims three. Reading the server's real slot count lets the caller clamp
     to it and say so.

The probe hits `/props` rather than `/health` because it answers both questions
at once: reachability, and `total_slots` for the clamp.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import httpx

# Read timeout, for a server that accepted the connection but is busy loading a
# model.
PROBE_TIMEOUT = 3.0

# Connect timeout, kept separate and much shorter. A refused connection is not
# instant on Windows: the stack sits on its retransmit schedule for ~2s, and
# because "localhost" resolves to both ::1 and 127.0.0.1, httpx tries each in
# turn and the total reaches ~4.4s. Anything actually listening on loopback
# accepts in microseconds, so a short connect budget costs a healthy server
# nothing and stops the "server is down" case from being the slow path.
PROBE_CONNECT_TIMEOUT = 1.0

# Fields emitted by llama-server's /props response across current and older
# builds. `total_slots` is deliberately not required by itself: older builds may
# omit it, but still identify themselves through model/generation metadata.
LLAMA_PROPS_MARKERS = frozenset({
    'total_slots',
    'model_path',
    'default_generation_settings',
    'chat_template',
    'modalities',
})


@dataclass(frozen=True)
class ServerInfo:
    """What a probe learned about the server.

    slots is None when the server answered but did not report `total_slots`
    (older llama.cpp builds). Callers must treat that as "unknown", not as 1,
    so an unrecognised build does not silently serialize the client.
    """

    reachable: bool
    slots: Optional[int] = None
    model_path: Optional[str] = None
    context_size: Optional[int] = None
    error: Optional[str] = None
    retryable: bool = False


def props_url(api_url: str) -> str:
    """Map an OpenAI-compatible base URL to the server's /props endpoint.

    Config carries `http://host:port/v1`; `/props` lives at the server root.
    """
    base = api_url.rstrip('/')
    for suffix in ('/v1', '/v1/chat/completions', '/chat/completions'):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base.rstrip('/')}/props"


def _request_url(url: str) -> str:
    """Pin a localhost probe to the IPv4 loopback.

    `localhost` resolves to both ::1 and 127.0.0.1, and httpx tries them in
    turn — so when nothing is listening, the probe pays the connect timeout
    twice. The IPv6 attempt cannot succeed anyway: llama-server is launched with
    `--host 0.0.0.0`, which binds IPv4 only. Addressing 127.0.0.1 directly drops
    the wasted half. Only the hostname is rewritten, so a user pointing at a
    real remote host is unaffected.
    """
    return url.replace('//localhost:', '//127.0.0.1:', 1)


def probe_server(api_url: str, timeout: float = PROBE_TIMEOUT) -> ServerInfo:
    """Ask the server what it is. Never raises — failures come back as data."""
    url = props_url(api_url)

    # Bound the connect phase separately. On Windows a refused connect is not
    # instant: the stack sits on its retransmit schedule for ~2s. An overall
    # timeout does not help, because httpx applies it per connection attempt
    # rather than to the whole call. A local server either completes the TCP
    # handshake immediately or is not there, so a short connect budget costs a
    # healthy server nothing and keeps the "server is down" path fast.
    limits = httpx.Timeout(timeout, connect=PROBE_CONNECT_TIMEOUT)

    try:
        response = httpx.get(_request_url(url), timeout=limits)
    except Exception as exc:  # connection refused, DNS, timeout, ...
        return ServerInfo(
            reachable=False,
            error=f"{type(exc).__name__}: {exc}",
            retryable=isinstance(exc, httpx.TransportError),
        )

    # A model still loading answers 503. That is reachable-but-not-ready, and
    # for the caller's purposes (can I translate now?) it is a failure.
    if response.status_code != 200:
        return ServerInfo(
            reachable=False,
            error=f"HTTP {response.status_code} from {url}",
            retryable=response.status_code == 503,
        )

    try:
        payload = response.json()
    except Exception:
        return ServerInfo(
            reachable=False,
            error=f"Invalid non-JSON response from {url}",
        )

    # Reachability alone is not enough: another local service can occupy the
    # configured port and return 200 from /props. Treat it as llama-server only
    # when its response carries at least one known llama-server property. This
    # avoids declaring preflight success and then falling into per-batch retries
    # against an unrelated HTTP service.
    if not isinstance(payload, dict) or not (payload.keys() & LLAMA_PROPS_MARKERS):
        return ServerInfo(
            reachable=False,
            error=f"Unexpected /props response from {url}",
        )

    slots = payload.get('total_slots')
    if not isinstance(slots, int) or slots < 1:
        slots = None

    generation_settings = payload.get('default_generation_settings')
    context_size = None
    if isinstance(generation_settings, dict):
        n_ctx = generation_settings.get('n_ctx')
        if isinstance(n_ctx, int) and n_ctx > 0:
            context_size = n_ctx

    model_path = payload.get('model_path')
    if not isinstance(model_path, str):
        model_path = None

    return ServerInfo(
        reachable=True,
        slots=slots,
        model_path=model_path,
        context_size=context_size,
    )


def clamp_workers(requested: int, info: ServerInfo) -> Tuple[int, Optional[str]]:
    """Clamp a requested worker count to the server's slot count.

    Returns (workers, note). note is None when nothing was changed, otherwise a
    user-facing line explaining the adjustment — the point of the clamp is that
    the user learns their two settings disagree, not just that we quietly fixed
    it.
    """
    try:
        requested = max(1, int(requested))
    except (TypeError, ValueError):
        requested = 1

    if info.slots is None:
        return requested, None

    if requested <= info.slots:
        return requested, None

    return info.slots, (
        f"workers {requested} → server 只有 {info.slots} "
        f"slot{'s' if info.slots > 1 else ''}，已調整為 {info.slots}"
        "（要真並行請在 Server 分頁提高「並發槽 (-np)」並重啟）"
    )


def unreachable_message(api_url: str, info: ServerInfo) -> str:
    """User-facing text for a failed preflight."""
    detail = info.error or "no response"
    return (
        f"llama-server 未回應 ({props_url(api_url)}): {detail}\n"
        "請先在 Server 分頁啟動伺服器並等模型載入完成。"
    )
