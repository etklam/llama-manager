"""Fault-injection tests for the shared atomic writer and commit gate.

Every test uses temporary files only. Failures are injected at the three
points that matter: mid-write (fsync), at the replacement, and against the
cancellation/commit boundary.
"""
import os
from pathlib import Path

import pytest

from utils.atomic_io import CommitGate, atomic_write_text


@pytest.fixture
def target(tmp_path):
    path = tmp_path / 'output.srt'
    path.write_text('previous complete output', encoding='utf-8')
    return path


class TestAtomicWriteText:

    def test_successful_write_replaces_content_and_leaves_no_temp(
            self, target, tmp_path):
        atomic_write_text(target, 'new content')
        assert target.read_text(encoding='utf-8') == 'new content'
        leftovers = [p for p in tmp_path.iterdir()
                     if p.name.endswith('.tmp')]
        assert leftovers == []

    def test_write_failure_preserves_previous_bytes(self, target, monkeypatch):
        def broken_fsync(fd):
            raise OSError('disk full during fsync')

        monkeypatch.setattr(os, 'fsync', broken_fsync)
        with pytest.raises(OSError, match='disk full'):
            atomic_write_text(target, 'half-written content')
        assert target.read_text(encoding='utf-8') == 'previous complete output'

    def test_failed_replacement_preserves_previous_bytes(
            self, target, monkeypatch, tmp_path):
        def broken_replace(src, dst):
            raise OSError('replace blocked')

        monkeypatch.setattr(os, 'replace', broken_replace)
        with pytest.raises(OSError, match='replace blocked'):
            atomic_write_text(target, 'content that never lands')
        assert target.read_text(encoding='utf-8') == 'previous complete output'
        leftovers = [p for p in tmp_path.iterdir() if p.name.endswith('.tmp')]
        assert leftovers == []

    def test_temp_file_lives_in_the_destination_directory(self, target,
                                                          monkeypatch):
        seen_dirs = []
        import utils.atomic_io as atomic_io
        real = atomic_io.tempfile.mkstemp

        def spy_mkstemp(*args, **kwargs):
            seen_dirs.append(kwargs.get('dir') or (args[0] if args else None))
            return real(*args, **kwargs)

        monkeypatch.setattr(atomic_io.tempfile, 'mkstemp', spy_mkstemp)
        atomic_write_text(target, 'same volume')
        assert seen_dirs == [str(target.parent)]

    def test_creates_missing_parent_directories(self, tmp_path):
        nested = tmp_path / 'deep' / 'sub' / 'out.srt'
        atomic_write_text(nested, 'content')
        assert nested.read_text(encoding='utf-8') == 'content'


class TestCommitGate:

    def test_commit_runs_when_not_cancelled(self):
        gate = CommitGate()
        ran = []
        assert gate.commit(lambda: ran.append(1)) is True
        assert ran == [1]

    def test_cancel_before_commit_suppresses_the_action(self):
        gate = CommitGate()
        gate.cancel()
        ran = []
        assert gate.commit(lambda: ran.append(1)) is False
        assert ran == []

    def test_external_cancel_check_is_honored(self):
        stop = {'requested': False}
        gate = CommitGate(cancel_check=lambda: stop['requested'])
        stop['requested'] = True
        ran = []
        assert gate.commit(lambda: ran.append(1)) is False
        assert ran == []

    def test_commit_then_cancel_publishes_then_cancels(self):
        gate = CommitGate()
        published = gate.commit(lambda: None)
        gate.cancel()
        assert published is True
        assert gate.cancelled is True

    def test_cancel_during_a_commit_waits_for_the_commit(self):
        """The serialized outcome: the commit finishes, cancel lands after."""
        import threading
        gate = CommitGate()
        order = []
        release = threading.Event()
        started = threading.Event()

        def slow_commit():
            started.set()
            release.wait(timeout=5)
            order.append('committed')

        worker = threading.Thread(
            target=lambda: gate.commit(slow_commit))
        worker.start()
        started.wait(timeout=5)

        canceller = threading.Thread(
            target=lambda: (order.append('cancel requested'),
                            gate.cancel(), order.append('cancelled')))
        canceller.start()
        # Canceller is blocked on the lock the commit holds.
        assert order == ['cancel requested']

        release.set()
        worker.join(timeout=5)
        canceller.join(timeout=5)
        assert order == ['cancel requested', 'committed', 'cancelled']
