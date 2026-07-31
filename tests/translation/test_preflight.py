"""Tests for the Translation Preflight module: probe → clamp → one note.

run_preflight is the whole probe→clamp→plan flow in one call, so these tests
patch the probe — the substitutable seam — and assert what the plan carries
for each probe outcome. The probe itself (httpx, /props, loopback pinning)
is tested in test_server_probe.py.
"""
from unittest.mock import patch

from translation.preflight import PreflightPlan, run_preflight
from translation.server_probe import ServerInfo


class TestRunPreflight:

    def test_unreachable_probe_aborts_with_message(self):
        with patch('translation.preflight.probe_server',
                   return_value=ServerInfo(
                       reachable=False, error="ConnectError: refused")):
            plan = run_preflight("http://localhost:8080/v1", 3)

        assert plan.reachable is False
        assert plan.workers == 3  # untouched: nothing to clamp against
        assert plan.note is not None
        assert "http://localhost:8080/props" in plan.note
        assert "ConnectError: refused" in plan.note

    def test_clamps_workers_to_slot_count(self):
        with patch('translation.preflight.probe_server',
                   return_value=ServerInfo(reachable=True, slots=1)):
            plan = run_preflight("http://localhost:8080/v1", 3)

        assert plan.reachable is True
        assert plan.workers == 1
        assert "3" in plan.note and "1" in plan.note

    def test_no_note_when_workers_fit(self):
        with patch('translation.preflight.probe_server',
                   return_value=ServerInfo(reachable=True, slots=4)):
            plan = run_preflight("http://localhost:8080/v1", 3)

        assert plan.workers == 3
        assert plan.note is None

    def test_unknown_slots_leaves_workers_alone(self):
        """An older build reporting no total_slots must not serialize the client."""
        with patch('translation.preflight.probe_server',
                   return_value=ServerInfo(reachable=True, slots=None)):
            plan = run_preflight("http://localhost:8080/v1", 3)

        assert plan.workers == 3
        assert plan.note is None

    def test_probe_receives_the_api_url(self):
        with patch('translation.preflight.probe_server',
                   return_value=ServerInfo(reachable=True, slots=4)) as probe:
            run_preflight("http://localhost:8080/v1", 3)

        probe.assert_called_once_with("http://localhost:8080/v1")

    def test_plan_carries_the_probe_info(self):
        info = ServerInfo(reachable=True, slots=2)
        with patch('translation.preflight.probe_server', return_value=info):
            plan = run_preflight("http://localhost:8080/v1", 3)

        assert plan.info is info

    def test_plan_is_frozen(self):
        import dataclasses

        assert dataclasses.is_dataclass(PreflightPlan)
        assert PreflightPlan.__dataclass_params__.frozen
