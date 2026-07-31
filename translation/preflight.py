"""The Translation Preflight: the probe→clamp→plan decision before a run.

Two consumers used to hand-roll the same sequence — probe the server, abort
when it is unreachable, clamp the worker count to the server's slot count,
and explain either outcome — and the two copies drifted: the pipeline
re-clamped per file and deduplicated its note, the tab clamped once and had
no dedup at all. This module owns the whole flow so the probe, the clamp,
and the note formatting cannot drift apart again. Callers ask for one plan
per translation run and only present the result: the tab shows its
messagebox and log line, the pipeline its log line.

The HTTP probe stays in server_probe; preflight composes it. probe_server is
the substitutable seam — callers and tests patch it, not httpx.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from translation.server_probe import (
    ServerInfo,
    clamp_workers,
    probe_server,
    unreachable_message,
)


@dataclass(frozen=True)
class PreflightPlan:
    """What one preflight pass decided about a translation run.

    Well-formed for every probe outcome: an unreachable server still carries
    the requested worker count unchanged (there is no server to clamp
    against, and callers must not translate), and `note` is None exactly when
    there is nothing to tell the user.
    """

    reachable: bool
    workers: int
    info: ServerInfo
    note: Optional[str] = None


def run_preflight(api_url: str, requested_workers: int) -> PreflightPlan:
    """Probe the server and decide the workers for one translation run.

    Never raises: probe_server already turns transport failures into data,
    and every outcome becomes a plan. The note — the unreachable message or
    the clamp explanation — is formatted here, once, so every consumer
    presents the same user-facing text.
    """
    info = probe_server(api_url)
    if not info.reachable:
        return PreflightPlan(
            reachable=False,
            workers=requested_workers,
            info=info,
            note=unreachable_message(api_url, info),
        )

    workers, clamp_note = clamp_workers(requested_workers, info)
    return PreflightPlan(
        reachable=True,
        workers=workers,
        info=info,
        note=clamp_note,
    )
