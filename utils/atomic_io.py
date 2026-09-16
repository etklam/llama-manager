"""Atomic file publication and the race-aware cancellation/commit boundary.

Direct ``Path.write_text`` on an output file is a destructive write: a
crash or power loss midway leaves a truncated file where a complete one
used to be, and with "Replace original" that file is the user's input.
This module mirrors the transcription side's candidate/commit pattern
(see whisper_transcription.WhisperTranscriber): write a sibling
temporary file completely, fsync it, close every handle, and only then
``os.replace`` it over the destination — on Windows the replace itself
is atomic once no handle to the temp file remains open.
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional


def atomic_write_text(path: os.PathLike | str, text: str,
                      encoding: str = 'utf-8') -> None:
    """Publish `text` at `path` atomically.

    The candidate is a temp file in the destination's own directory (same
    volume, so the replace is a rename, not a copy), fully written,
    flushed, fsynced and closed before the replace. Any failure before the
    replace leaves the destination's previous bytes untouched; the temp
    file is removed. A failure of the replace itself also leaves the
    original intact (``os.replace`` does not truncate first).

    Raises whatever the underlying I/O raised; the destination is never
    left holding a partial write.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f'.{destination.name}.',
        suffix='.tmp',
    )
    try:
        with os.fdopen(fd, 'w', encoding=encoding, newline='') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        # The handle is closed by the with-block before the replace: on
        # Windows an open handle would make os.replace fail.
        os.replace(temp_name, str(destination))
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

    # Best-effort directory fsync so the rename itself survives power
    # loss; not available on Windows, where it is skipped rather than
    # turning every successful write into an error.
    if os.name == 'posix':
        try:
            dir_fd = os.open(str(destination.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass


class CommitGate:
    """The race-aware boundary between a cancellation token and a commit.

    "Cancel" and "publish" can race: the user presses Stop at the same
    instant the last line finishes. Checking the token, then committing,
    is not enough — the token can flip between the two. Serializing both
    under one lock (mirroring whisper_transcription's
    CancellationToken._commit) makes the outcome unambiguous: either the
    commit wins and is published, or cancellation wins and nothing is.

    Unlike a plain cancel flag, ``cancel()`` blocks until any in-flight
    commit finished, so a cancelled run can never observe a commit that
    its own cancellation was checked against.
    """

    def __init__(self, cancel_check: Optional[Callable[[], bool]] = None):
        self._lock = threading.Lock()
        self._cancelled = False
        self._cancel_check = cancel_check or (lambda: False)

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled or self._cancel_check()

    def commit(self, action: Callable[[], None]) -> bool:
        """Run `action` only if cancellation has not won the race.

        Returns True when the commit ran, False when it was suppressed.
        The action runs under the lock, so a `cancel()` arriving during a
        commit waits for the commit to finish instead of splitting the
        outcome.
        """
        with self._lock:
            if self._cancelled or self._cancel_check():
                return False
            action()
            return True
