"""Where translation LLM requests go: local llama-server or a remote API.

Backend selection used to be implicit — every consumer built
``http://localhost:{port}/v1`` itself, so pointing anywhere else meant
scattering ``if remote`` checks through every translator call. This module is
the single resolver: consumers ask for an :class:`LLMTarget` and hand it to
``build_translation_config``; whether that target is a local llama-server or a
saved OpenAI-compatible profile is decided here and nowhere else.

Remote profiles persist under ``llm.profiles`` in config.json. Keys never do:
the persistent credential lives in an environment variable named by
``api_key_env``; a key typed into the UI lives in :data:`SESSION_KEYS` for the
current process only. The resolver prefers the session key, then the
environment variable.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

MODE_LOCAL = 'local'
MODE_REMOTE = 'remote'
MODES = (MODE_LOCAL, MODE_REMOTE)


def api_url_for_port(port):
    """Base URL of the local llama-server's OpenAI-compatible API.

    Shared so the preflight probe and the translator cannot end up pointing at
    different endpoints.
    """
    return f'http://localhost:{port}/v1'


class SessionKeyStore:
    """API keys typed into the UI. Memory-only: never written to config.json."""

    def __init__(self):
        self._keys: Dict[str, str] = {}

    def set(self, profile_id: str, key: str) -> None:
        if key:
            self._keys[profile_id] = key
        else:
            self._keys.pop(profile_id, None)

    def get(self, profile_id: str) -> str:
        return self._keys.get(profile_id, '')

    def is_set(self, profile_id: str) -> bool:
        return profile_id in self._keys

    def clear(self) -> None:
        self._keys.clear()


# The one session-wide store; the settings dialog writes, the resolver reads.
SESSION_KEYS = SessionKeyStore()


@dataclass(frozen=True, repr=False)
class LLMTarget:
    """The effective LLM endpoint for one translation run.

    ``api_key`` is the resolved credential (session key or environment
    variable), not the environment variable name — that is ``key_env``, kept
    alongside so a missing key can be reported with the name the user must
    set.
    """

    mode: str
    name: str
    api_url: str
    model: str
    api_key: str = ''
    proxy: Optional[str] = None
    max_workers: int = 3
    key_env: str = ''

    def __repr__(self) -> str:
        # repr travels into logs and tracebacks; the key must never ride along.
        return (
            f"LLMTarget(mode='{self.mode}', name='{self.name}', "
            f"api_url='{self.api_url}', model='{self.model}', "
            f"api_key={'***' if self.api_key else ''}, "
            f"proxy={self.proxy!r}, max_workers={self.max_workers}, "
            f"key_env='{self.key_env}')"
        )


def normalize_base_url(url: str) -> str:
    """Coerce a pasted endpoint into an OpenAI SDK base URL.

    Trims whitespace and trailing slashes, and strips a pasted
    ``/chat/completions`` or ``/v1/chat/completions`` path back to the base
    (appending ``/v1`` when the paste omitted it). A bare host is left alone:
    some providers serve the API at the root, so guessing ``/v1`` would break
    them.
    """
    base = (url or '').strip().rstrip('/')
    if base.lower().endswith('/chat/completions'):
        base = base[: -len('/chat/completions')]
        if not base.lower().endswith('/v1'):
            base += '/v1'
    return base


# ---------------------------------------------------------------------------
# Profile persistence (config.json → llm.profiles)
# ---------------------------------------------------------------------------

# The one field that must never reach config.json. Everything else a profile
# dict carries is copied through, so unknown keys survive edits.
SECRET_FIELDS = frozenset({'api_key', 'session_api_key'})


def _get(cm, key: str, default: Any) -> Any:
    return cm.get(key, default) if cm is not None else default


def current_mode(cm) -> str:
    mode = _get(cm, 'llm.mode', MODE_LOCAL)
    return mode if mode in MODES else MODE_LOCAL


def list_profiles(cm) -> List[Dict[str, Any]]:
    profiles = _get(cm, 'llm.profiles', [])
    return profiles if isinstance(profiles, list) else []


def active_profile(cm) -> Optional[Dict[str, Any]]:
    active_id = _get(cm, 'llm.active_profile_id', '')
    for profile in list_profiles(cm):
        if profile.get('id') == active_id:
            return profile
    return None


def new_profile_id() -> str:
    return uuid.uuid4().hex[:12]


def sanitize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """One shape for every profile before it is persisted.

    Strips credentials, trims strings, normalizes the base URL, and clamps
    max_workers — callers (dialog, resolver, tests) cannot each re-derive
    these rules without drifting.
    """
    sanitized = {
        key: value for key, value in (profile or {}).items()
        if key not in SECRET_FIELDS
    }
    sanitized['id'] = sanitized.get('id') or new_profile_id()
    for field in ('name', 'model', 'api_key_env', 'proxy'):
        value = sanitized.get(field)
        sanitized[field] = value.strip() if isinstance(value, str) else ''
    sanitized['base_url'] = normalize_base_url(sanitized.get('base_url') or '')
    try:
        sanitized['max_workers'] = max(1, int(sanitized.get('max_workers', 3)))
    except (TypeError, ValueError):
        sanitized['max_workers'] = 3
    return sanitized


def upsert_profile(cm, profile: Dict[str, Any]) -> str:
    """Insert or update a profile by id and return the stable id.

    A new profile gets an id here; an edited one keeps the id it came with,
    because ``llm.active_profile_id`` and the session key store both point at
    it by id. The update merges over what is stored, so a key this app does
    not know about survives an edit from the settings dialog, which rebuilds
    a profile from its own fields.
    """
    sanitized = sanitize_profile(profile)
    profiles = [dict(p) for p in list_profiles(cm)]
    for index, existing in enumerate(profiles):
        if existing.get('id') == sanitized['id']:
            merged = dict(existing)
            merged.update(sanitized)
            profiles[index] = sanitize_profile(merged)
            break
    else:
        profiles.append(sanitized)
    cm.set('llm.profiles', profiles)
    return sanitized['id']


def delete_profile(cm, profile_id: str) -> None:
    profiles = [p for p in list_profiles(cm) if p.get('id') != profile_id]
    cm.set('llm.profiles', profiles)
    if _get(cm, 'llm.active_profile_id', '') == profile_id:
        cm.set('llm.active_profile_id', '')
    SESSION_KEYS.set(profile_id, '')


def set_mode(cm, mode: str) -> None:
    if mode in MODES:
        cm.set('llm.mode', mode)


def set_active_profile(cm, profile_id: str) -> None:
    cm.set('llm.active_profile_id', profile_id)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _local_target(cm, get_port: Callable[[], int],
                  get_model: Callable[[], str]) -> LLMTarget:
    try:
        max_workers = max(1, int(_get(cm, 'ui.max_workers', 3)))
    except (TypeError, ValueError):
        max_workers = 3
    return LLMTarget(
        mode=MODE_LOCAL,
        name='llama-server (local)',
        api_url=api_url_for_port(get_port()),
        model=(get_model() or ''),
        max_workers=max_workers,
    )


def resolve_llm_target(cm, get_port: Callable[[], int],
                       get_local_model: Callable[[], str]) -> LLMTarget:
    """Turn application state into the one endpoint a translation run uses.

    Never raises: a remote mode with a missing or half-configured profile
    resolves to a target with empty fields, which the preflight turns into a
    user-facing message naming the profile — a run that cannot work should
    fail with an explanation, not a traceback.
    """
    if current_mode(cm) == MODE_REMOTE:
        return _remote_target(cm)
    return _local_target(cm, get_port, get_local_model)


def _remote_target(cm) -> LLMTarget:
    profile = active_profile(cm) or {}
    profile_id = _get(cm, 'llm.active_profile_id', '')
    api_key_env = profile.get('api_key_env') or ''
    # Session key first: an explicitly typed key wins over a possibly stale
    # environment value for exactly as long as the user keeps it in the UI.
    api_key = SESSION_KEYS.get(profile_id)
    if not api_key and api_key_env:
        api_key = os.environ.get(api_key_env, '')
    try:
        max_workers = max(1, int(profile.get('max_workers', 3)))
    except (TypeError, ValueError):
        max_workers = 3
    return LLMTarget(
        mode=MODE_REMOTE,
        name=profile.get('name') or profile_id or 'remote profile',
        api_url=normalize_base_url(profile.get('base_url') or ''),
        model=(profile.get('model') or '').strip(),
        api_key=api_key,
        proxy=(profile.get('proxy') or '').strip() or None,
        max_workers=max_workers,
        key_env=api_key_env,
    )
