"""Tests for the slimmed LocalLLMTranslator.__init__ contract.

After the cleanup, LocalLLMTranslator trusts the config dict (built by
config_helpers.build_translation_config) and no longer re-defaults missing
keys. Only 'model' presence is validated, and temperature/max_tokens get
defensive clamping at the trust boundary.
"""
import pytest

from translation.local_llm_translator import LocalLLMTranslator


def _full_config(**overrides):
    cfg = {
        "api_url": "http://localhost:8080/v1",
        "model": "llama-3.2",
        "max_tokens": 16384,
        "temperature": 0.2,
        "batch_size": 15,
        "max_workers": 3,
        "single_step": True,
    }
    cfg.update(overrides)
    return cfg


class TestTranslatorInitTrustsDict:

    def test_explicit_values_kept(self):
        cfg = _full_config(
            temperature=0.7, max_tokens=2000, max_workers=5,
            single_step=False, batch_size=20,
        )
        t = LocalLLMTranslator(cfg)
        assert t.temperature == 0.7
        assert t.max_tokens == 2000
        assert t.max_workers == 5
        assert t.single_step is False

    def test_no_redefault_temperature(self):
        # ponytail: if config_helpers is the only default layer, omitting
        # temperature must NOT silently become 0.3 inside the translator.
        cfg = _full_config()
        del cfg["temperature"]
        with pytest.raises(KeyError):
            LocalLLMTranslator(cfg)

    def test_no_redefault_max_tokens(self):
        cfg = _full_config()
        del cfg["max_tokens"]
        with pytest.raises(KeyError):
            LocalLLMTranslator(cfg)

    def test_no_redefault_max_workers(self):
        cfg = _full_config()
        del cfg["max_workers"]
        with pytest.raises(KeyError):
            LocalLLMTranslator(cfg)

    def test_no_redefault_single_step(self):
        cfg = _full_config()
        del cfg["single_step"]
        with pytest.raises(KeyError):
            LocalLLMTranslator(cfg)


class TestTranslatorInitValidation:

    def test_missing_model_raises(self):
        cfg = _full_config()
        del cfg["model"]
        with pytest.raises(ValueError):
            LocalLLMTranslator(cfg)

    def test_temperature_clamped_high(self):
        cfg = _full_config(temperature=3.0)
        t = LocalLLMTranslator(cfg)
        assert t.temperature == 2.0

    def test_temperature_clamped_negative(self):
        cfg = _full_config(temperature=-1.0)
        t = LocalLLMTranslator(cfg)
        assert t.temperature == 0.0

    def test_max_tokens_zero_rejected(self):
        # ponytail: defensive check at trust boundary stays.
        cfg = _full_config(max_tokens=0)
        t = LocalLLMTranslator(cfg)
        # Invalid max_tokens falls back to a safe positive value
        assert t.max_tokens > 0
