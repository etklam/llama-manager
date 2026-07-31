"""The Chunking Policy: the rule that decides when a Transcription is Chunked.

One deep module owning the whole rule so its pieces cannot drift apart: the
config key the Whisper tab and the pipeline share, the duration threshold, the
probe epsilon, the chunk size, and the checkbox label that renders that size.
Callers ask the policy and execute the ChunkPlan it returns; the bare config
key string never leaves this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from config_manager import ConfigManager


CONFIG_KEY = "whisper.chunk_long_audio"
CHUNK_SECONDS = 30 * 60
_CHUNK_TIME_EPSILON = 0.001


@dataclass(frozen=True)
class ChunkPlan:
    """What the policy decided for one piece of media.

    Well-formed for any duration: unchunked media still carries the policy's
    chunk size and a chunk_count of 1, so callers can execute the plan without
    branching on the decision.
    """

    chunked: bool
    chunk_seconds: float
    chunk_count: int


def decide(duration_seconds: float) -> ChunkPlan:
    """Decide how one media duration is transcribed.

    Long media (over CHUNK_SECONDS, with the probe's epsilon so a float that
    rounds just past the boundary is not spuriously chunked) is Chunked;
    everything else stays a single Transcription. The chunk count is chosen so
    rounding never produces a zero-length final chunk.
    """
    if duration_seconds > CHUNK_SECONDS + _CHUNK_TIME_EPSILON:
        chunk_count = max(
            1,
            math.ceil(
                (duration_seconds - _CHUNK_TIME_EPSILON) / CHUNK_SECONDS
            ),
        )
        return ChunkPlan(True, CHUNK_SECONDS, chunk_count)
    return ChunkPlan(False, CHUNK_SECONDS, 1)


def chunking_enabled(config_manager: ConfigManager) -> bool:
    """Read the chunking flag from config; off when it was never set."""
    return bool(config_manager.get(CONFIG_KEY, False))


def set_chunking_enabled(config_manager: ConfigManager, enabled: bool) -> None:
    """Persist the chunking flag for both the Whisper tab and the pipeline."""
    config_manager.set(CONFIG_KEY, bool(enabled))


def chunking_label() -> str:
    """Checkbox label derived from CHUNK_SECONDS.

    Rendered from the constant so the UI text cannot drift from the chunk size
    the policy enforces.
    """
    return f"Chunk long audio ({CHUNK_SECONDS // 60} min, anti-repeat)"
