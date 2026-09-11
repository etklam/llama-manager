"""Tests for config_helpers.build_translation_config."""
import pytest

from config_manager import ConfigManager
from config_helpers import build_translation_config


@pytest.fixture
def cm(tmp_path):
    manager = ConfigManager(str(tmp_path / "cfg.json"))
    manager.load()
    return manager


class TestBuildTranslationConfigDefaults:

    def test_returns_dict_with_all_keys(self, cm):
        cfg = build_translation_config(cm, 8080, "llama-3.2")
        for key in ("api_url", "model", "max_tokens", "temperature",
                    "batch_size", "max_workers", "single_step",
                    "context_mode", "context_size"):
            assert key in cfg

    def test_default_values(self, cm):
        cfg = build_translation_config(cm, 8080, "llama-3.2")
        assert cfg["batch_size"] == 15
        assert cfg["temperature"] == 0.2
        assert cfg["max_tokens"] == 16384
        assert cfg["max_workers"] == 3
        assert cfg["single_step"] is True
        assert cfg["context_mode"] == "none"
        assert cfg["context_size"] is None

    def test_api_url_built_from_port(self, cm):
        cfg = build_translation_config(cm, 9999, "llama-3.2")
        assert cfg["api_url"] == "http://localhost:9999/v1"

    def test_model_passed_through(self, cm):
        cfg = build_translation_config(cm, 8080, "qwen-7b")
        assert cfg["model"] == "qwen-7b"


class TestBuildTranslationConfigOverrides:

    def test_applies_cm_overrides(self, cm):
        cm.set("ui.batch_size", 25)
        cm.set("ui.temperature", 0.5)
        cm.set("ui.max_tokens", 8192)
        cm.set("ui.max_workers", 5)
        cm.set("ui.single_step", False)
        cm.set("ui.context_mode", "story")
        cfg = build_translation_config(cm, 8080, "llama-3.2")
        assert cfg["batch_size"] == 25
        assert cfg["temperature"] == 0.5
        assert cfg["max_tokens"] == 8192
        assert cfg["max_workers"] == 5
        assert cfg["single_step"] is False
        assert cfg["context_mode"] == "story"

    def test_missing_cm_keys_fall_back_to_defaults(self, cm):
        cfg = build_translation_config(cm, 8080, "llama-3.2")
        assert cfg["batch_size"] == 15
        assert cfg["temperature"] == 0.2
        assert cfg["max_tokens"] == 16384
        assert cfg["max_workers"] == 3
        assert cfg["single_step"] is True
