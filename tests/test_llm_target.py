"""Tests for llm_target: the LLM endpoint resolver and profile store."""
import json

import pytest

from config_manager import ConfigManager
from llm_target import (
    LLMTarget, SESSION_KEYS, active_profile, current_mode, delete_profile,
    list_profiles, normalize_base_url, resolve_llm_target, set_active_profile,
    set_mode, upsert_profile,
)


@pytest.fixture
def cm(tmp_path):
    manager = ConfigManager(str(tmp_path / "cfg.json"))
    manager.load()
    return manager


@pytest.fixture(autouse=True)
def clean_session_keys():
    SESSION_KEYS.clear()
    yield
    SESSION_KEYS.clear()


PROFILE = {
    'id': '',
    'name': 'OpenRouter',
    'base_url': 'https://openrouter.ai/api/v1',
    'model': 'provider/model-name',
    'api_key_env': 'OPENROUTER_API_KEY',
    'max_workers': 3,
    'proxy': '',
}


class TestNormalizeBaseUrl:

    def test_trims_whitespace_and_trailing_slash(self):
        assert normalize_base_url('  https://api.x.com/v1/ ') == 'https://api.x.com/v1'

    def test_strips_pasted_chat_completions_path(self):
        assert normalize_base_url(
            'https://api.x.com/v1/chat/completions') == 'https://api.x.com/v1'

    def test_pasted_bare_chat_completions_gains_v1(self):
        assert normalize_base_url(
            'https://api.x.com/chat/completions') == 'https://api.x.com/v1'

    def test_bare_host_is_left_alone(self):
        assert normalize_base_url('http://192.168.1.5:8000') == 'http://192.168.1.5:8000'

    def test_empty_url_stays_empty(self):
        assert normalize_base_url('') == ''
        assert normalize_base_url(None) == ''


class TestProfilePersistence:

    def test_upsert_assigns_and_returns_a_stable_id(self, cm):
        first_id = upsert_profile(cm, dict(PROFILE))
        stored = list_profiles(cm)[0]
        assert stored['id'] == first_id

        edited = dict(PROFILE, id=first_id, name='OpenRouter 2')
        assert upsert_profile(cm, edited) == first_id  # same profile, same id
        assert len(list_profiles(cm)) == 1
        assert list_profiles(cm)[0]['name'] == 'OpenRouter 2'

    def test_profiles_survive_a_restart(self, tmp_path, cm):
        upsert_profile(cm, dict(PROFILE))
        path = str(tmp_path / "cfg.json")
        reloaded = ConfigManager(path)
        reloaded.load()
        assert [p['name'] for p in list_profiles(reloaded)] == ['OpenRouter']

    def test_unknown_profile_keys_survive_an_edit(self, cm):
        pid = upsert_profile(cm, dict(PROFILE, custom_flag=True))
        upsert_profile(cm, dict(PROFILE, id=pid, name='edited'))
        assert list_profiles(cm)[0].get('custom_flag') is True

    def test_plaintext_keys_never_reach_config_json(self, tmp_path, cm):
        upsert_profile(cm, dict(PROFILE, api_key='sk-secret'))
        raw = json.loads((tmp_path / "cfg.json").read_text(encoding='utf-8'))
        dumped = json.dumps(raw)
        assert 'sk-secret' not in dumped
        for profile in raw['llm']['profiles']:
            assert 'api_key' not in profile

    def test_delete_clears_active_pointer_and_session_key(self, cm):
        pid = upsert_profile(cm, dict(PROFILE))
        set_active_profile(cm, pid)
        SESSION_KEYS.set(pid, 'sk-session')

        delete_profile(cm, pid)

        assert list_profiles(cm) == []
        assert cm.get('llm.active_profile_id') == ''
        assert SESSION_KEYS.get(pid) == ''

    def test_delete_of_another_profile_keeps_the_active_pointer(self, cm):
        keep = upsert_profile(cm, dict(PROFILE))
        other = upsert_profile(cm, dict(PROFILE, name='Other'))
        set_active_profile(cm, keep)

        delete_profile(cm, other)

        assert active_profile(cm)['id'] == keep


class TestModes:

    def test_config_without_llm_section_resolves_local(self, cm):
        assert current_mode(cm) == 'local'

    def test_invalid_mode_falls_back_to_local(self, cm):
        cm.set('llm.mode', 'telepathy')
        assert current_mode(cm) == 'local'

    def test_set_mode_rejects_unknown_values(self, cm):
        set_mode(cm, 'telepathy')
        assert current_mode(cm) == 'local'


class TestLocalResolution:

    def test_local_target_points_at_the_llama_server_port(self, cm):
        cm.set('ui.max_workers', 5)
        target = resolve_llm_target(cm, lambda: 8081, lambda: 'gemma')
        assert target.mode == 'local'
        assert target.api_url == 'http://localhost:8081/v1'
        assert target.model == 'gemma'
        assert target.api_key == ''
        assert target.proxy is None
        assert target.max_workers == 5

    def test_local_target_without_a_model_resolves_empty(self, cm):
        target = resolve_llm_target(cm, lambda: 8080, lambda: '')
        assert target.model == ''  # the preflight/runner reports it, not the resolver


class TestRemoteResolution:

    def _remote_cm(self, cm, **profile_overrides):
        profile = dict(PROFILE, **profile_overrides)
        pid = upsert_profile(cm, profile)
        set_mode(cm, 'remote')
        set_active_profile(cm, pid)
        return pid

    def test_remote_target_uses_the_active_profile(self, cm):
        self._remote_cm(cm)
        target = resolve_llm_target(cm, lambda: 8080, lambda: 'ignored')
        assert target.mode == 'remote'
        assert target.name == 'OpenRouter'
        assert target.api_url == 'https://openrouter.ai/api/v1'
        assert target.model == 'provider/model-name'
        assert target.max_workers == 3

    def test_api_key_comes_from_the_environment_variable(self, monkeypatch, cm):
        self._remote_cm(cm)
        monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-from-env')
        target = resolve_llm_target(cm, lambda: 8080, lambda: '')
        assert target.api_key == 'sk-from-env'

    def test_session_key_wins_over_the_environment(self, monkeypatch, cm):
        pid = self._remote_cm(cm)
        monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-from-env')
        SESSION_KEYS.set(pid, 'sk-from-session')
        target = resolve_llm_target(cm, lambda: 8080, lambda: '')
        assert target.api_key == 'sk-from-session'

    def test_profile_without_key_env_needs_no_key(self, cm):
        self._remote_cm(cm, api_key_env='')
        target = resolve_llm_target(cm, lambda: 8080, lambda: '')
        assert target.api_key == ''

    def test_proxy_and_base_url_normalization_apply(self, cm):
        self._remote_cm(cm,base_url=' https://api.x.com/v1/ ',
                        proxy=' http://127.0.0.1:7890 ')
        target = resolve_llm_target(cm, lambda: 8080, lambda: '')
        assert target.api_url == 'https://api.x.com/v1'
        assert target.proxy == 'http://127.0.0.1:7890'

    def test_remote_mode_with_no_active_profile_resolves_to_empty_fields(self, cm):
        set_mode(cm, 'remote')
        target = resolve_llm_target(cm, lambda: 8080, lambda: '')
        assert target.mode == 'remote'
        assert target.api_url == ''
        assert target.model == ''


class TestNoSecretsInRepr:

    def test_repr_masks_the_api_key(self):
        target = LLMTarget(mode='remote', name='p', api_url='u', model='m',
                           api_key='sk-super-secret')
        assert 'sk-super-secret' not in repr(target)
        assert '***' in repr(target)
