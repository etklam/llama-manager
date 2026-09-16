"""Tests for the Translation Preflight module: probe → clamp → one note.

run_preflight is the whole probe→clamp→plan flow in one call, so these tests
patch the probe — the substitutable seam — and assert what the plan carries
for each probe outcome. The probe itself (httpx, /props, loopback pinning)
is tested in test_server_probe.py.
"""
from unittest.mock import patch

from translation.preflight import PreflightPlan, plan_for_target, run_preflight
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


class TestPlanForTarget:
    """Backend dispatch: local targets keep the /props flow, remote never probe.

    Remote endpoints are OpenAI-compatible APIs, not llama.cpp servers: /props
    does not exist there, slot counts are meaningless, and there is no model
    load to wait out. The plan validates what is knowable locally and leaves
    live failures to the client.
    """

    def _remote_target(self, **overrides):
        from llm_target import LLMTarget
        fields = dict(
            mode='remote', name='OpenRouter',
            api_url='https://openrouter.ai/api/v1',
            model='provider/model', api_key='sk-x', max_workers=3)
        fields.update(overrides)
        return LLMTarget(**fields)

    def test_local_target_uses_the_props_probe(self):
        with patch('translation.preflight.probe_server',
                   return_value=ServerInfo(reachable=True, slots=1)) as probe:
            from llm_target import LLMTarget
            plan = plan_for_target(LLMTarget(
                mode='local', name='llama-server (local)',
                api_url='http://localhost:8080/v1', model='m'), 3)

        probe.assert_called_once_with('http://localhost:8080/v1')
        assert plan.reachable is True
        assert plan.workers == 1  # local keeps the slot clamp

    def test_remote_target_never_probes_props(self):
        with patch('translation.preflight.probe_server') as probe:
            plan = plan_for_target(self._remote_target(), 3)

        probe.assert_not_called()
        assert plan.reachable is True
        assert plan.workers == 3

    def test_remote_target_does_not_clamp_workers(self):
        """No llama-server slots exist remotely; the profile's count stands."""
        with patch('translation.preflight.probe_server',
                   return_value=ServerInfo(reachable=True, slots=1)):
            plan = plan_for_target(self._remote_target(), 8)

        assert plan.workers == 8

    def test_missing_fields_are_reported_with_the_profile_name(self):
        target = self._remote_target(api_url='', model='', api_key='',
                                     key_env='OPENROUTER_API_KEY')
        plan = plan_for_target(target, 3)

        assert plan.reachable is False
        assert plan.workers == 3  # untouched: nothing to clamp against
        assert 'OpenRouter' in plan.note
        assert 'OPENROUTER_API_KEY' in plan.note

    def test_no_key_env_means_no_key_required(self):
        """A keyless endpoint (e.g. LAN vLLM) must not be flagged incomplete."""
        plan = plan_for_target(self._remote_target(api_key='', key_env=''), 3)

        assert plan.reachable is True
        assert plan.note is None

    def test_remote_plan_carries_no_context_size(self):
        """Context sizing is llama-server knowledge; the story planner treats
        None as its own default instead of a server-reported cap."""
        plan = plan_for_target(self._remote_target(), 3)

        assert plan.info.context_size is None
        assert plan.info.slots is None
