"""Tests for config_helpers.build_translation_config."""
import pytest

from config_manager import ConfigManager
from config_helpers import build_translation_config
from llm_target import LLMTarget, api_url_for_port


@pytest.fixture
def cm(tmp_path):
    manager = ConfigManager(str(tmp_path / "cfg.json"))
    manager.load()
    return manager


def local_target(port=8080, model="llama-3.2", workers=3):
    return LLMTarget(
        mode='local', name='llama-server (local)',
        api_url=api_url_for_port(port), model=model, max_workers=workers)


def remote_target(url="https://openrouter.ai/api/v1", model="provider/model",
                  api_key="sk-test", proxy=None, workers=3):
    return LLMTarget(
        mode='remote', name='OpenRouter', api_url=url, model=model,
        api_key=api_key, proxy=proxy, max_workers=workers)


class TestBuildTranslationConfigDefaults:

    def test_returns_dict_with_all_keys(self, cm):
        cfg = build_translation_config(cm, local_target())
        for key in ("api_url", "model", "api_key", "proxy", "max_tokens",
                    "temperature", "batch_size", "max_workers", "single_step",
                    "context_mode", "context_size"):
            assert key in cfg

    def test_default_values(self, cm):
        cfg = build_translation_config(cm, local_target())
        assert cfg["batch_size"] == 15
        assert cfg["temperature"] == 0.2
        assert cfg["max_tokens"] == 16384
        assert cfg["max_workers"] == 3
        assert cfg["single_step"] is True
        assert cfg["context_mode"] == "none"
        assert cfg["context_size"] is None

    def test_endpoint_comes_from_the_target(self, cm):
        cfg = build_translation_config(cm, local_target(port=9999, model="qwen-7b"))
        assert cfg["api_url"] == "http://localhost:9999/v1"
        assert cfg["model"] == "qwen-7b"

    def test_local_target_carries_no_key_or_proxy(self, cm):
        cfg = build_translation_config(cm, local_target())
        assert cfg["api_key"] == ''
        assert cfg["proxy"] is None

    def test_remote_target_spreads_endpoint_fields(self, cm):
        cfg = build_translation_config(
            cm, remote_target(proxy="http://127.0.0.1:7890", workers=5))
        assert cfg["api_url"] == "https://openrouter.ai/api/v1"
        assert cfg["model"] == "provider/model"
        assert cfg["api_key"] == "sk-test"
        assert cfg["proxy"] == "http://127.0.0.1:7890"
        assert cfg["max_workers"] == 5


class TestBuildTranslationConfigOverrides:

    def test_applies_cm_overrides(self, cm):
        cm.set("ui.batch_size", 25)
        cm.set("ui.temperature", 0.5)
        cm.set("ui.max_tokens", 8192)
        cm.set("ui.single_step", False)
        cm.set("ui.context_mode", "story")
        cfg = build_translation_config(cm, local_target())
        assert cfg["batch_size"] == 25
        assert cfg["temperature"] == 0.5
        assert cfg["max_tokens"] == 8192
        assert cfg["single_step"] is False
        assert cfg["context_mode"] == "story"

    def test_missing_cm_keys_fall_back_to_defaults(self, cm):
        cfg = build_translation_config(cm, local_target())
        assert cfg["batch_size"] == 15
        assert cfg["temperature"] == 0.2
        assert cfg["max_tokens"] == 16384
        assert cfg["single_step"] is True
