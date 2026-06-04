"""
Comprehensive test suite for WhisperModelRegistry module.

This test file follows TDD (Test-Driven Development) principles.
Tests are written one at a time, following red-green-refactor cycle.

Run with: pytest tests/test_whisper_model_registry.py -v --no-header --tb=short
"""

import pytest
from pathlib import Path
from typing import Optional

from whisper_model_registry import WhisperModelRegistry
from config_manager import ConfigManager


@pytest.fixture
def temp_config_file(tmp_path):
    return tmp_path / "test_config.json"


@pytest.fixture
def temp_scan_dir(tmp_path):
    scan_dir = tmp_path / "whisper_models"
    scan_dir.mkdir()
    return scan_dir


@pytest.fixture
def config_manager(temp_config_file):
    return ConfigManager(str(temp_config_file))


@pytest.fixture
def whisper_registry(temp_scan_dir, config_manager):
    return WhisperModelRegistry(temp_scan_dir, config_manager)


class TestParseModelName:

    def test_parse_tiny(self):
        result = WhisperModelRegistry.parse_model_name("ggml-tiny.bin")
        assert result == "tiny", f"Expected 'tiny', got '{result}'"

    def test_parse_base(self):
        result = WhisperModelRegistry.parse_model_name("ggml-base.bin")
        assert result == "base", f"Expected 'base', got '{result}'"

    def test_parse_small_en(self):
        result = WhisperModelRegistry.parse_model_name("ggml-small.en.bin")
        assert result == "small (en)", f"Expected 'small (en)', got '{result}'"

    def test_parse_large_v3_turbo(self):
        result = WhisperModelRegistry.parse_model_name("ggml-large-v3-turbo.bin")
        assert result == "large-v3-turbo", f"Expected 'large-v3-turbo', got '{result}'"

    def test_parse_large_v3(self):
        result = WhisperModelRegistry.parse_model_name("ggml-large-v3.bin")
        assert result == "large-v3", f"Expected 'large-v3', got '{result}'"

    def test_parse_medium_en(self):
        result = WhisperModelRegistry.parse_model_name("ggml-medium.en.bin")
        assert result == "medium (en)", f"Expected 'medium (en)', got '{result}'"

    def test_parse_non_ggml_prefix(self):
        result = WhisperModelRegistry.parse_model_name("random-file.bin")
        assert result == "random-file", f"Expected 'random-file', got '{result}'"


class TestListModelsEmpty:

    def test_list_models_returns_empty_when_config_empty(self, whisper_registry):
        models = whisper_registry.list_models()
        assert isinstance(models, list), "Should return a list"
        assert len(models) == 0, "Should return empty list when no models"

    def test_list_models_returns_empty_when_models_section_missing(self, whisper_registry):
        models = whisper_registry.list_models()
        assert isinstance(models, list), "Should return a list"
        assert len(models) == 0, "Should return empty list"


class TestScanDiscovers:

    def test_scan_discovers_ggml_bin_files(self, whisper_registry, temp_scan_dir):
        (temp_scan_dir / "ggml-tiny.bin").touch()
        (temp_scan_dir / "ggml-base.bin").touch()

        models, added, removed = whisper_registry.scan()

        assert len(models) == 2, f"Should discover 2 models, found {len(models)}"
        assert added == 2, f"Should add 2 models, added count was {added}"
        assert removed == 0, f"Should remove 0 models, removed count was {removed}"

    def test_scan_ignores_non_bin_files(self, whisper_registry, temp_scan_dir):
        (temp_scan_dir / "ggml-tiny.bin").touch()
        (temp_scan_dir / "readme.txt").touch()
        (temp_scan_dir / "config.json").touch()

        models, added, removed = whisper_registry.scan()

        assert len(models) == 1, f"Should only discover 1 .bin file, found {len(models)}"
        assert added == 1, f"Should add 1 model, added count was {added}"

    def test_scan_returns_model_info_dicts(self, whisper_registry, temp_scan_dir):
        test_file = temp_scan_dir / "ggml-tiny.bin"
        test_file.touch()
        test_file.write_text("x" * (1024 * 1024 * 75))

        models, added, removed = whisper_registry.scan()

        assert len(models) == 1, "Should discover 1 model"
        model = models[0]

        assert "name" in model, "Model dict should have 'name' key"
        assert "path" in model, "Model dict should have 'path' key"
        assert "size" in model, "Model dict should have 'size' key"
        assert "format" in model, "Model dict should have 'format' key"

        assert model["name"] == "tiny", f"Name should be parsed, got {model['name']}"
        assert model["size"].endswith("MB"), f"Size should be formatted in MB, got {model['size']}"

    def test_scan_discovers_any_bin_files(self, whisper_registry, temp_scan_dir):
        (temp_scan_dir / "ggml-whisper-tiny.bin").touch()
        (temp_scan_dir / "ggml-tiny.bin").touch()
        (temp_scan_dir / "whisper-base.bin").touch()

        models, added, removed = whisper_registry.scan()

        assert len(models) == 3, f"Should discover all .bin files, found {len(models)}"


class TestScanRemove:

    def test_scan_removes_models_with_missing_files(self, whisper_registry, temp_scan_dir):
        test_file = temp_scan_dir / "ggml-tiny.bin"
        test_file.write_text("x" * 1024)

        whisper_registry.scan()

        test_file.unlink()

        models, added, removed = whisper_registry.scan()

        assert len(models) == 0, f"Should have 0 models after file deletion, found {len(models)}"
        assert removed == 1, f"Should remove 1 model, removed count was {removed}"
        assert added == 0, f"Should add 0 models, added count was {added}"


class TestScanPreserve:

    def test_scan_preserves_existing_models(self, whisper_registry, temp_scan_dir):
        (temp_scan_dir / "ggml-tiny.bin").write_text("x" * 1024)
        (temp_scan_dir / "ggml-base.bin").write_text("x" * 1024)

        whisper_registry.scan()

        models, added, removed = whisper_registry.scan()

        assert len(models) == 2, f"Should preserve 2 models, found {len(models)}"
        assert added == 0, f"Should add 0 models, added count was {added}"
        assert removed == 0, f"Should remove 0 models, removed count was {removed}"


class TestScanCounts:

    def test_scan_counts_added_and_removed(self, whisper_registry, temp_scan_dir):
        (temp_scan_dir / "ggml-tiny.bin").write_text("x" * 1024)
        (temp_scan_dir / "ggml-base.bin").write_text("x" * 1024)

        whisper_registry.scan()

        (temp_scan_dir / "ggml-base.bin").unlink()
        (temp_scan_dir / "ggml-small.bin").write_text("x" * 1024)
        (temp_scan_dir / "ggml-medium.bin").write_text("x" * 1024)

        models, added, removed = whisper_registry.scan()

        assert len(models) == 3, f"Should have 3 models total, found {len(models)}"
        assert added == 2, f"Should add 2 models, added count was {added}"
        assert removed == 1, f"Should remove 1 model, removed count was {removed}"


class TestAddModel:

    def test_add_model_adds_to_registry(self, whisper_registry, temp_scan_dir):
        test_file = temp_scan_dir / "ggml-large-v3.bin"
        test_file.write_text("x" * (1024 * 1024 * 3))

        model_info = whisper_registry.add_model(str(test_file))

        assert model_info is not None, "Should return model info dict"
        assert model_info["name"] == "large-v3", f"Name should be parsed, got {model_info['name']}"
        assert model_info["path"] == str(test_file), f"Path should match, got {model_info['path']}"

        models = whisper_registry.list_models()
        assert len(models) == 1, f"Should have 1 model in list, found {len(models)}"
        assert models[0]["name"] == "large-v3", "Model should be in list"

    def test_add_model_persists_to_config(self, whisper_registry, temp_scan_dir):
        test_file = temp_scan_dir / "ggml-tiny.bin"
        test_file.write_text("x" * 1024)

        whisper_registry.add_model(str(test_file))

        new_registry = WhisperModelRegistry(temp_scan_dir, whisper_registry.config_manager)
        models = new_registry.list_models()

        assert len(models) == 1, "Model should be persisted to config"
        assert models[0]["name"] == "tiny", "Persisted model name should match"


class TestGetModelPath:

    def test_get_model_path_returns_path_for_known_model(self, whisper_registry, temp_scan_dir):
        test_file = temp_scan_dir / "ggml-tiny.bin"
        test_file.write_text("x" * 1024)

        whisper_registry.add_model(str(test_file))

        path = whisper_registry.get_model_path("tiny")

        assert path == str(test_file), f"Path should match, expected {str(test_file)}, got {path}"

    def test_get_model_path_returns_none_for_unknown_model(self, whisper_registry):
        path = whisper_registry.get_model_path("non-existent-model")
        assert path is None, f"Should return None for unknown model, got {path}"


class TestScanMissingDirectory:

    def test_scan_handles_missing_directory(self, tmp_path, config_manager):
        missing_dir = tmp_path / "non-existent-dir"
        registry = WhisperModelRegistry(missing_dir, config_manager)

        models, added, removed = registry.scan()

        assert models == [], "Should return empty list for missing directory"
        assert added == 0, "Should have 0 added for missing directory"
        assert removed == 0, "Should have 0 removed for missing directory"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header", "--tb=short"])
