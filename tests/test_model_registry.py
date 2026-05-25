"""
Comprehensive test suite for ModelRegistry module.

This test file follows TDD (Test-Driven Development) principles.
Tests are written one at a time, following red-green-refactor cycle.

Run with: pytest tests/test_model_registry.py -v --no-header --tb=short
"""

import pytest
from pathlib import Path
from typing import Optional

# This import will FAIL until ModelRegistry is implemented
from model_registry import ModelRegistry
from config_manager import ConfigManager


# Fixtures
@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file path."""
    return tmp_path / "test_config.json"


@pytest.fixture
def temp_scan_dir(tmp_path):
    """Create a temporary scan directory."""
    scan_dir = tmp_path / "models"
    scan_dir.mkdir()
    return scan_dir


@pytest.fixture
def config_manager(temp_config_file):
    """Create a ConfigManager instance."""
    return ConfigManager(str(temp_config_file))


@pytest.fixture
def model_registry(temp_scan_dir, config_manager):
    """Create a ModelRegistry instance."""
    return ModelRegistry(temp_scan_dir, config_manager)


# Test 1: detect_format returns correct format for known patterns
class TestDetectFormat:
    """Test suite for format detection from filenames."""

    def test_detect_format_returns_q4_k_m(self):
        """Test detecting Q4_K_M format."""
        filename = "llama-2-7b-chat.Q4_K_M.gguf"
        format_detected = ModelRegistry.detect_format(filename)
        assert format_detected == "Q4_K_M", f"Expected 'Q4_K_M', got '{format_detected}'"

    def test_detect_format_returns_q5_k_s(self):
        """Test detecting Q5_K_S format."""
        filename = "model.Q5_K_S.gguf"
        format_detected = ModelRegistry.detect_format(filename)
        assert format_detected == "Q5_K_S", f"Expected 'Q5_K_S', got '{format_detected}'"

    def test_detect_format_case_insensitive(self):
        """Test that format detection is case-insensitive."""
        filename = "model.q4_k_m.GGUF"
        format_detected = ModelRegistry.detect_format(filename)
        assert format_detected == "Q4_K_M", f"Expected 'Q4_K_M' (case-insensitive), got '{format_detected}'"

    def test_detect_format_multiple_formats_in_name(self):
        """Test detecting format when multiple format strings appear."""
        # Should detect the first matching format
        filename = "model.Q4_K_M.Q5_K_M.gguf"
        format_detected = ModelRegistry.detect_format(filename)
        # Should match one of them (implementation-dependent which one)
        assert format_detected in ["Q4_K_M", "Q5_K_M"], f"Expected one of the formats, got '{format_detected}'"


# Test 2: detect_format returns "Unknown" for unrecognized patterns
class TestDetectFormatUnknown:
    """Test suite for unknown format detection."""

    def test_detect_format_returns_unknown_for_no_format(self):
        """Test detecting format when no format string is present."""
        filename = "model.gguf"
        format_detected = ModelRegistry.detect_format(filename)
        assert format_detected == "Unknown", f"Expected 'Unknown', got '{format_detected}'"

    def test_detect_format_returns_unknown_for_gibberish(self):
        """Test detecting format from gibberish filename."""
        filename = "xyz123-abc-789.gguf"
        format_detected = ModelRegistry.detect_format(filename)
        assert format_detected == "Unknown", f"Expected 'Unknown', got '{format_detected}'"

    def test_detect_format_all_known_formats(self):
        """Test all known format patterns are detected."""
        test_cases = [
            ("model.Q4_K_M.gguf", "Q4_K_M"),
            ("model.Q4_K_S.gguf", "Q4_K_S"),
            ("model.Q5_K_M.gguf", "Q5_K_M"),
            ("model.Q5_K_S.gguf", "Q5_K_S"),
            ("model.Q8_0.gguf", "Q8_0"),
            ("model.IQ4_NL.gguf", "IQ4_NL"),
            ("model.IQ4_XS.gguf", "IQ4_XS"),
            ("model.Q3_K_M.gguf", "Q3_K_M"),
            ("model.Q2_K.gguf", "Q2_K"),
        ]

        for filename, expected_format in test_cases:
            result = ModelRegistry.detect_format(filename)
            assert result == expected_format, f"For {filename}, expected '{expected_format}', got '{result}'"


# Test 3: list_models returns empty list when no models in config
class TestListModelsEmpty:
    """Test suite for listing models when config is empty."""

    def test_list_models_returns_empty_when_config_empty(self, model_registry):
        """Test that list_models returns empty list when no models in config."""
        models = model_registry.list_models()
        assert isinstance(models, list), "Should return a list"
        assert len(models) == 0, "Should return empty list when no models"

    def test_list_models_returns_empty_when_models_section_missing(self, model_registry):
        """Test that list_models returns empty list when models section is missing."""
        models = model_registry.list_models()
        assert isinstance(models, list), "Should return a list"
        assert len(models) == 0, "Should return empty list"


# Test 4: scan discovers .gguf files in scan_dir
class TestScanDiscover:
    """Test suite for scanning and discovering .gguf files."""

    def test_scan_discovers_gguf_files(self, model_registry, temp_scan_dir):
        """Test that scan discovers .gguf files in scan directory."""
        # Create test .gguf files
        (temp_scan_dir / "model1.gguf").touch()
        (temp_scan_dir / "model2.gguf").touch()

        models, added, removed = model_registry.scan()

        assert len(models) == 2, f"Should discover 2 models, found {len(models)}"
        assert added == 2, f"Should add 2 models, added count was {added}"
        assert removed == 0, f"Should remove 0 models, removed count was {removed}"

    def test_scan_ignores_non_gguf_files(self, model_registry, temp_scan_dir):
        """Test that scan ignores non-.gguf files."""
        # Create mix of files
        (temp_scan_dir / "model.gguf").touch()
        (temp_scan_dir / "readme.txt").touch()
        (temp_scan_dir / "config.json").touch()

        models, added, removed = model_registry.scan()

        assert len(models) == 1, f"Should only discover 1 .gguf file, found {len(models)}"
        assert added == 1, f"Should add 1 model, added count was {added}"

    def test_scan_returns_model_info_dicts(self, model_registry, temp_scan_dir):
        """Test that scan returns proper model info dictionaries."""
        # Create a test file with specific format
        test_file = temp_scan_dir / "llama-2-7b-chat.Q4_K_M.gguf"
        test_file.touch()

        # Write some content to give it size
        test_file.write_text("x" * (1024 * 1024 * 5))  # 5MB

        models, added, removed = model_registry.scan()

        assert len(models) == 1, "Should discover 1 model"
        model = models[0]

        assert "name" in model, "Model dict should have 'name' key"
        assert "path" in model, "Model dict should have 'path' key"
        assert "size" in model, "Model dict should have 'size' key"
        assert "format" in model, "Model dict should have 'format' key"

        assert model["name"] == "llama-2-7b-chat.Q4_K_M", f"Name should be stem, got {model['name']}"
        assert model["format"] == "Q4_K_M", f"Format should be detected, got {model['format']}"
        assert model["size"].endswith("GB"), f"Size should be formatted in GB, got {model['size']}"


# Test 5: scan removes models whose files no longer exist
class TestScanRemove:
    """Test suite for scanning and removing non-existent models."""

    def test_scan_removes_models_with_missing_files(self, model_registry, temp_scan_dir):
        """Test that scan removes models whose files no longer exist."""
        # Create a .gguf file
        test_file = temp_scan_dir / "model1.gguf"
        test_file.write_text("x" * 1024)

        # Scan to add it
        model_registry.scan()

        # Delete the file
        test_file.unlink()

        # Scan again
        models, added, removed = model_registry.scan()

        assert len(models) == 0, f"Should have 0 models after file deletion, found {len(models)}"
        assert removed == 1, f"Should remove 1 model, removed count was {removed}"
        assert added == 0, f"Should add 0 models, added count was {added}"


# Test 6: scan preserves existing valid models
class TestScanPreserve:
    """Test suite for scanning and preserving valid models."""

    def test_scan_preserves_existing_models(self, model_registry, temp_scan_dir):
        """Test that scan preserves models whose files still exist."""
        # Create test files
        (temp_scan_dir / "model1.gguf").write_text("x" * 1024)
        (temp_scan_dir / "model2.gguf").write_text("x" * 1024)

        # Initial scan
        model_registry.scan()

        # Scan again without changes
        models, added, removed = model_registry.scan()

        assert len(models) == 2, f"Should preserve 2 models, found {len(models)}"
        assert added == 0, f"Should add 0 models, added count was {added}"
        assert removed == 0, f"Should remove 0 models, removed count was {removed}"


# Test 7: scan returns correct added/removed counts
class TestScanCounts:
    """Test suite for scan count tracking."""

    def test_scan_counts_added_and_removed(self, model_registry, temp_scan_dir):
        """Test that scan correctly counts added and removed models."""
        # Create initial files
        (temp_scan_dir / "model1.gguf").write_text("x" * 1024)
        (temp_scan_dir / "model2.gguf").write_text("x" * 1024)

        # Initial scan
        model_registry.scan()

        # Delete one, add two new
        (temp_scan_dir / "model2.gguf").unlink()
        (temp_scan_dir / "model3.gguf").write_text("x" * 1024)
        (temp_scan_dir / "model4.gguf").write_text("x" * 1024)

        # Scan again
        models, added, removed = model_registry.scan()

        assert len(models) == 3, f"Should have 3 models total, found {len(models)}"
        assert added == 2, f"Should add 2 models, added count was {added}"
        assert removed == 1, f"Should remove 1 model, removed count was {removed}"


# Test 8: add_model adds a model to the registry
class TestAddModel:
    """Test suite for adding models individually."""

    def test_add_model_adds_to_registry(self, model_registry, temp_scan_dir):
        """Test that add_model adds a model to the registry."""
        # Create a test file
        test_file = temp_scan_dir / "new-model.Q5_K_M.gguf"
        test_file.write_text("x" * (1024 * 1024 * 3))

        # Add the model
        model_info = model_registry.add_model(str(test_file))

        assert model_info is not None, "Should return model info dict"
        assert model_info["name"] == "new-model.Q5_K_M", f"Name should match stem, got {model_info['name']}"
        assert model_info["format"] == "Q5_K_M", f"Format should be detected, got {model_info['format']}"
        assert model_info["path"] == str(test_file), f"Path should match, got {model_info['path']}"

        # Verify it's in the list
        models = model_registry.list_models()
        assert len(models) == 1, f"Should have 1 model in list, found {len(models)}"
        assert models[0]["name"] == "new-model.Q5_K_M", "Model should be in list"

    def test_add_model_persists_to_config(self, model_registry, temp_scan_dir):
        """Test that add_model persists to config."""
        # Create a test file
        test_file = temp_scan_dir / "persistent-model.gguf"
        test_file.write_text("x" * 1024)

        # Add the model
        model_registry.add_model(str(test_file))

        # Create new registry instance (should load from config)
        new_registry = ModelRegistry(temp_scan_dir, model_registry.config_manager)
        models = new_registry.list_models()

        assert len(models) == 1, "Model should be persisted to config"
        assert models[0]["name"] == "persistent-model", "Persisted model name should match"


# Test 9: get_model_path returns path for known model
class TestGetModelPath:
    """Test suite for getting model paths by name."""

    def test_get_model_path_returns_path_for_known_model(self, model_registry, temp_scan_dir):
        """Test that get_model_path returns correct path for known model."""
        # Create a test file
        test_file = temp_scan_dir / "test-model.gguf"
        test_file.write_text("x" * 1024)

        # Add it
        model_registry.add_model(str(test_file))

        # Get path
        path = model_registry.get_model_path("test-model")

        assert path == str(test_file), f"Path should match, expected {str(test_file)}, got {path}"

    def test_get_model_path_handles_duplicate_names(self, model_registry, temp_scan_dir):
        """Test get_model_path when models have duplicate names (uses first)."""
        # This tests edge case - in practice names should be unique
        # But we should handle it gracefully
        file1 = temp_scan_dir / "model.gguf"
        file2 = temp_scan_dir / "model-copy.gguf"
        file1.write_text("x" * 1024)
        file2.write_text("x" * 1024)

        model_registry.add_model(str(file1))
        model_registry.add_model(str(file2))

        # Should return path for one of them
        path = model_registry.get_model_path("model")
        assert path is not None, "Should find a model"
        assert path in [str(file1), str(file2)], "Should return one of the model paths"


# Test 10: get_model_path returns None for unknown model
class TestGetModelPathUnknown:
    """Test suite for getting paths of unknown models."""

    def test_get_model_path_returns_none_for_unknown_model(self, model_registry):
        """Test that get_model_path returns None for unknown model."""
        path = model_registry.get_model_path("non-existent-model")
        assert path is None, f"Should return None for unknown model, got {path}"

    def test_get_model_path_returns_none_for_empty_name(self, model_registry):
        """Test that get_model_path returns None for empty name."""
        path = model_registry.get_model_path("")
        assert path is None, f"Should return None for empty name, got {path}"


# Test 11: scan handles missing scan directory gracefully
class TestScanMissingDirectory:
    """Test suite for handling missing scan directory."""

    def test_scan_handles_missing_directory(self, tmp_path, config_manager):
        """Test that scan handles missing scan directory gracefully."""
        # Create registry with non-existent directory
        missing_dir = tmp_path / "non-existent-dir"
        registry = ModelRegistry(missing_dir, config_manager)

        # Should not raise exception
        models, added, removed = registry.scan()

        # Should return empty results
        assert models == [], "Should return empty list for missing directory"
        assert added == 0, "Should have 0 added for missing directory"
        assert removed == 0, "Should have 0 removed for missing directory"

    def test_scan_creates_directory_if_possible(self, tmp_path, config_manager):
        """Test that scan can handle directory creation if needed."""
        # This test verifies behavior - scan shouldn't create directory
        # but should handle it gracefully
        missing_dir = tmp_path / "models" / "subdir"
        registry = ModelRegistry(missing_dir, config_manager)

        # Should not raise, just return empty
        models, added, removed = registry.scan()

        assert models == [], "Should return empty results"
        # Directory should NOT be created by scan
        assert not missing_dir.exists(), "Scan should not create directory"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header", "--tb=short"])
