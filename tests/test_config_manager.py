"""
Comprehensive test suite for ConfigManager module.

This test file follows TDD (Test-Driven Development) principles.
All tests are designed to FAIL initially since the ConfigManager implementation doesn't exist yet.

Run with: pytest tests/test_config_manager.py -v
"""

import pytest
import json
import os
from pathlib import Path
from typing import Any, Dict

# This import will FAIL until ConfigManager is implemented
from config_manager import ConfigManager, DEFAULT_CONFIG


# Fixtures
@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file path."""
    return tmp_path / "test_config.json"


@pytest.fixture
def default_config():
    """Default configuration structure."""
    return {
        "server": {
            "port": 8080,
            "host": "0.0.0.0",
            "gpu_layers": 99,
            "context_size": 131072,
            "batch_size": 256,
            "threads": -1
        },
        "ui": {
            "theme": "default",
            "auto_scroll": True
        },
        "translation": {
            "enabled": False,
            "source_lang": "auto",
            "target_lang": "en",
            "provider": "mock"
        }
    }


@pytest.fixture
def sample_config_file(temp_config_file, default_config):
    """Create a sample config file with data."""
    config_data = {
        "server": {
            "port": 9090,
            "host": "localhost",
            "gpu_layers": 50,
            "context_size": 65536,
            "batch_size": 128,
            "threads": 4
        },
        "ui": {
            "theme": "dark",
            "auto_scroll": False
        },
        "translation": {
            "enabled": True,
            "source_lang": "es",
            "target_lang": "en",
            "provider": "google"
        }
    }

    with open(temp_config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

    return temp_config_file


@pytest.fixture
def config_manager(sample_config_file):
    """Create a ConfigManager instance with sample config."""
    return ConfigManager(str(sample_config_file))


@pytest.fixture
def config_manager_with_defaults(temp_config_file):
    """Create a ConfigManager instance with defaults (no existing file)."""
    return ConfigManager(str(temp_config_file))


class TestServerDefaults:
    def test_dflash_defaults(self):
        for key, expected in {
            "dflash_enabled": False,
            "dflash_model_path": "",
            "dflash_n_max": 6,
            "dflash_gpu_layers": "all",
            "dflash_device": "Vulkan0",
            "mmproj_path": "",
        }.items():
            assert DEFAULT_CONFIG["server"][key] == expected


# Test 1: Test loading config file
class TestLoadConfigFile:
    """Test suite for loading configuration files."""

    def test_load_existing_config(self, config_manager, sample_config_file):
        """Test loading an existing config file returns a dict."""
        config = config_manager.load()

        assert isinstance(config, dict), "Loaded config should be a dictionary"
        assert "server" in config, "Config should contain 'server' section"
        assert "ui" in config, "Config should contain 'ui' section"
        assert config["server"]["port"] == 9090, "Should load correct port value"
        assert config["ui"]["theme"] == "dark", "Should load correct theme value"

    def test_load_missing_config_creates_defaults(self, temp_config_file, default_config):
        """Test that missing config file is created with default values."""
        # Ensure file doesn't exist
        if temp_config_file.exists():
            temp_config_file.unlink()

        manager = ConfigManager(str(temp_config_file))
        config = manager.load()

        # Verify file was created
        assert temp_config_file.exists(), "Config file should be created if missing"

        # Verify default structure
        assert isinstance(config, dict), "Created config should be a dictionary"
        assert "server" in config, "Default config should contain 'server' section"
        assert config["server"]["port"] == 8080, "Default port should be 8080"

    def test_load_invalid_json_raises_error(self, temp_config_file):
        """Test loading invalid JSON file raises appropriate error."""
        # Create file with invalid JSON
        with open(temp_config_file, 'w') as f:
            f.write("{ invalid json content")

        manager = ConfigManager(str(temp_config_file))

        with pytest.raises((json.JSONDecodeError, ValueError)):
            manager.load()

    def test_load_empty_file_returns_empty_dict(self, temp_config_file):
        """Test loading an empty JSON file returns defaults."""
        with open(temp_config_file, 'w') as f:
            f.write("{}")

        manager = ConfigManager(str(temp_config_file))
        config = manager.load()

        # Empty JSON should be merged with defaults
        assert isinstance(config, dict), "Loaded config should be a dictionary"
        assert "server" in config, "Should have default server section"


# Test 2: Test saving config file
class TestSaveConfigFile:
    """Test suite for saving configuration files."""

    def test_save_config_to_file(self, config_manager, temp_config_file):
        """Test saving config updates the file."""
        new_config = {
            "server": {
                "port": 9999,
                "host": "192.168.1.1"
            },
            "ui": {
                "theme": "light"
            }
        }

        config_manager.save(new_config)

        # Verify file was updated
        with open(temp_config_file, 'r') as f:
            saved_config = json.load(f)

        assert saved_config == new_config, "Saved config should match input"
        assert saved_config["server"]["port"] == 9999, "Port should be updated"

    def test_save_creates_file_if_missing(self, temp_config_file):
        """Test saving creates file if it doesn't exist."""
        if temp_config_file.exists():
            temp_config_file.unlink()

        manager = ConfigManager(str(temp_config_file))
        config = {"test": "value"}

        manager.save(config)

        assert temp_config_file.exists(), "File should be created after save"

        with open(temp_config_file, 'r') as f:
            saved_config = json.load(f)

        assert saved_config == config, "Saved content should match"

    def test_save_preserves_json_formatting(self, config_manager, temp_config_file):
        """Test that saved config has proper JSON formatting."""
        config = {
            "server": {"port": 8080},
            "ui": {"theme": "dark"}
        }

        config_manager.save(config)

        # Read file content
        with open(temp_config_file, 'r') as f:
            content = f.read()

        # Verify it's properly formatted JSON
        parsed = json.loads(content)
        assert parsed == config, "Parsed JSON should match original"

        # Verify indentation (should be readable)
        assert '\n' in content or len(content) < 100, "JSON should be formatted with newlines"

    def test_save_creates_parent_directories(self, tmp_path):
        """Test saving creates parent directories if needed."""
        nested_path = tmp_path / "subdir" / "nested" / "config.json"

        manager = ConfigManager(str(nested_path))
        manager.save({"test": "value"})

        assert nested_path.exists(), "Nested directories should be created"
        assert nested_path.parent.exists(), "Parent directories should exist"


# Test 3: Test getting config values
class TestGetConfigValue:
    """Test suite for getting configuration values."""

    def test_get_existing_simple_value(self, config_manager):
        """Test getting an existing simple value."""
        port = config_manager.get("server.port")
        assert port == 9090, "Should return correct port value"

    def test_get_with_default_fallback(self, config_manager):
        """Test getting non-existent value with default."""
        value = config_manager.get("nonexistent.key", "default_value")
        assert value == "default_value", "Should return default value for missing key"

    def test_get_nested_value(self, config_manager):
        """Test getting nested values using dot notation."""
        host = config_manager.get("server.host")
        assert host == "localhost", "Should return nested value"

        theme = config_manager.get("ui.theme")
        assert theme == "dark", "Should return nested ui value"

    def test_get_deeply_nested_value(self, config_manager):
        """Test getting deeply nested values."""
        # Test with translation config
        provider = config_manager.get("translation.provider")
        assert provider == "google", "Should return deeply nested value"

    def test_get_without_default_returns_none(self, config_manager):
        """Test getting missing value without default returns None."""
        value = config_manager.get("missing.key")
        assert value is None, "Should return None for missing key without default"

    def test_get_entire_section(self, config_manager):
        """Test getting an entire config section."""
        server_config = config_manager.get("server")

        assert isinstance(server_config, dict), "Section should be a dict"
        assert "port" in server_config, "Section should contain port"
        assert "host" in server_config, "Section should contain host"
        assert server_config["port"] == 9090, "Section should have correct values"

    def test_get_with_none_default(self, config_manager):
        """Test getting with None as explicit default."""
        value = config_manager.get("missing.key", None)
        assert value is None, "Should return None when specified as default"


# Test 4: Test setting config values
class TestSetConfigValue:
    """Test suite for setting configuration values."""

    def test_set_simple_value(self, config_manager):
        """Test setting a simple top-level value."""
        config_manager.set("new_key", "new_value")

        value = config_manager.get("new_key")
        assert value == "new_value", "Should retrieve newly set value"

    def test_set_nested_value(self, config_manager):
        """Test setting a nested value."""
        config_manager.set("server.port", 8888)

        port = config_manager.get("server.port")
        assert port == 8888, "Should update nested value"

    def test_set_creates_nested_structure(self, config_manager):
        """Test setting creates nested structure if it doesn't exist."""
        config_manager.set("new.nested.value", 123)

        value = config_manager.get("new.nested.value")
        assert value == 123, "Should create and set nested value"

    def test_set_with_auto_save(self, config_manager, temp_config_file):
        """Test that set auto-saves by default."""
        config_manager.set("server.port", 7777)

        # Verify it was saved to file
        with open(temp_config_file, 'r') as f:
            saved_config = json.load(f)

        assert saved_config["server"]["port"] == 7777, "Value should be saved to file"

    def test_set_without_auto_save(self, config_manager, temp_config_file):
        """Test setting without auto-save."""
        original_port = config_manager.get("server.port")

        config_manager.set("server.port", 6666, auto_save=False)

        # In-memory value should change
        assert config_manager.get("server.port") == 6666, "In-memory value should update"

        # File should not change
        with open(temp_config_file, 'r') as f:
            saved_config = json.load(f)

        assert saved_config["server"]["port"] == original_port, "File should not change without auto-save"

    def test_set_multiple_values(self, config_manager):
        """Test setting multiple values in sequence."""
        config_manager.set("server.host", "0.0.0.0")
        config_manager.set("ui.theme", "light")
        config_manager.set("new.value", 42)

        assert config_manager.get("server.host") == "0.0.0.0"
        assert config_manager.get("ui.theme") == "light"
        assert config_manager.get("new.value") == 42

    def test_set_overwrites_existing_value(self, config_manager):
        """Test that set overwrites existing values."""
        config_manager.set("server.port", 5555)
        config_manager.set("server.port", 4444)

        assert config_manager.get("server.port") == 4444, "Should overwrite existing value"


# Test 5: Test translation config section via generic get/set
class TestTranslationConfigViaGetSet:
    """Test suite verifying get('translation') and set('translation', ...) work
    equivalently to the removed get_translation_config / set_translation_config."""

    def test_get_translation_config_via_get(self, config_manager):
        """Test getting the translation config section using get('translation')."""
        trans_config = config_manager.get("translation")

        assert isinstance(trans_config, dict), "Translation config should be a dict"
        assert "enabled" in trans_config, "Should contain 'enabled' key"
        assert "source_lang" in trans_config, "Should contain 'source_lang' key"
        assert "target_lang" in trans_config, "Should contain 'target_lang' key"
        assert "provider" in trans_config, "Should contain 'provider' key"

    def test_get_translation_config_values(self, config_manager):
        """Test translation config has correct values."""
        trans_config = config_manager.get("translation")

        assert trans_config["enabled"] is True, "Should have correct enabled value"
        assert trans_config["source_lang"] == "es", "Should have correct source_lang"
        assert trans_config["target_lang"] == "en", "Should have correct target_lang"
        assert trans_config["provider"] == "google", "Should have correct provider"

    def test_set_translation_config_via_set(self, config_manager):
        """Test setting the entire translation config using set('translation', ...)."""
        new_trans_config = {
            "enabled": False,
            "source_lang": "fr",
            "target_lang": "de",
            "provider": "deepl"
        }

        config_manager.set("translation", new_trans_config)

        retrieved = config_manager.get("translation")
        assert retrieved == new_trans_config, "Translation config should be updated"

    def test_set_translation_config_partial_via_set(self, config_manager):
        """Test updating individual translation keys using set('translation.key', ...)."""
        config_manager.set("translation.enabled", False)

        assert config_manager.get("translation.enabled") is False, "Should update specified key"
        assert config_manager.get("translation.provider") == "google", "Should preserve other keys"

    def test_get_translation_config_with_defaults(self, temp_config_file):
        """Test getting translation config when missing returns default from merge."""
        # Create config without translation section
        with open(temp_config_file, 'w') as f:
            json.dump({"server": {"port": 8080}}, f)

        manager = ConfigManager(str(temp_config_file))
        manager.load()
        trans_config = manager.get("translation")

        # Should return default translation config from merge
        assert isinstance(trans_config, dict), "Should return default dict"
        assert "enabled" in trans_config, "Should have default keys"
        assert "provider" in trans_config, "Should have default provider"

    def test_set_translation_creates_section(self, temp_config_file):
        """Test setting translation config creates section if missing."""
        with open(temp_config_file, 'w') as f:
            json.dump({"server": {"port": 8080}}, f)

        manager = ConfigManager(str(temp_config_file))
        manager.set("translation.enabled", True)

        assert manager.get("translation.enabled") is True, "Should create and set translation config"


# Test 6: Test default config merge
class TestDefaultConfigMerge:
    """Test suite for merging user config with defaults."""

    def test_merge_with_defaults(self, temp_config_file, default_config):
        """Test that partial user config merges with defaults."""
        # Create minimal user config
        user_config = {
            "server": {
                "port": 9999  # Only override port
            }
        }

        with open(temp_config_file, 'w') as f:
            json.dump(user_config, f)

        manager = ConfigManager(str(temp_config_file))
        config = manager.load()

        # User value should be preserved
        assert config["server"]["port"] == 9999, "User value should override default"

        # Default values should be filled in
        assert "host" in config["server"], "Default keys should be added"
        assert "gpu_layers" in config["server"], "Default keys should be added"
        assert config["server"]["gpu_layers"] == 99, "Should use default value"

    def test_user_values_override_defaults(self, sample_config_file):
        """Test that user values always take precedence."""
        manager = ConfigManager(str(sample_config_file))
        config = manager.load()

        # All user values should be preserved
        assert config["server"]["port"] == 9090, "User port should override default"
        assert config["ui"]["theme"] == "dark", "User theme should override default"
        assert config["translation"]["enabled"] == True, "User translation should override"

    def test_missing_keys_get_defaults(self, temp_config_file):
        """Test that missing keys get default values."""
        # Create config with missing sections
        partial_config = {
            "server": {
                "port": 8080
            }
        }

        with open(temp_config_file, 'w') as f:
            json.dump(partial_config, f)

        manager = ConfigManager(str(temp_config_file))
        config = manager.load()

        # Missing sections should be added
        assert "ui" in config, "Missing ui section should be added"
        assert "translation" in config, "Missing translation section should be added"

        # Missing keys in existing sections should be added
        assert "host" in config["server"], "Missing server keys should be added"

    def test_empty_config_gets_all_defaults(self, temp_config_file):
        """Test that empty config file gets all default values."""
        with open(temp_config_file, 'w') as f:
            json.dump({}, f)

        manager = ConfigManager(str(temp_config_file))
        config = manager.load()

        # Should have all default sections
        assert "server" in config, "Should have server section"
        assert "ui" in config, "Should have ui section"
        assert "translation" in config, "Should have translation section"

    def test_merge_preserves_extra_keys(self, sample_config_file):
        """Test that merge preserves keys not in defaults."""
        manager = ConfigManager(str(sample_config_file))
        config = manager.load()

        # Add a custom key
        config["custom_section"] = {"custom_key": "custom_value"}
        manager.save(config)

        # Reload and verify
        manager2 = ConfigManager(str(sample_config_file))
        reloaded = manager2.load()

        assert "custom_section" in reloaded, "Custom sections should be preserved"
        assert reloaded["custom_section"]["custom_key"] == "custom_value", "Custom values should be preserved"


# Test 7: Test config validation
class TestConfigValidation:
    """Test suite for configuration validation."""

    def test_validate_required_keys_exist(self, config_manager):
        """Test validation checks for required keys."""
        errors = config_manager.validate()

        assert isinstance(errors, list), "Validation should return list of errors"
        # Current config is valid, so should be empty or no errors
        assert len(errors) == 0, "Valid config should have no validation errors"

    def test_validate_missing_required_key(self, temp_config_file):
        """Test validation detects missing required keys after merge."""
        # Create config without 'server' section
        with open(temp_config_file, 'w') as f:
            json.dump({"ui": {"theme": "dark"}}, f)

        manager = ConfigManager(str(temp_config_file))
        # load() merges with defaults, so server IS present
        errors = manager.validate()

        # After merge with defaults, there should be no errors
        assert len(errors) == 0, "Merged config should be valid since defaults fill missing sections"

    def test_validate_port_is_integer(self, temp_config_file):
        """Test validation checks port type."""
        # Create config with invalid port type
        with open(temp_config_file, 'w') as f:
            json.dump({"server": {"port": "not_a_number"}}, f)

        manager = ConfigManager(str(temp_config_file))
        errors = manager.validate()

        assert len(errors) > 0, "Should have validation errors for invalid port type"
        assert any("port" in str(e).lower() for e in errors), "Should error about port"

    def test_validate_port_in_valid_range(self, temp_config_file):
        """Test validation checks port is in valid range."""
        # Create config with out-of-range port
        with open(temp_config_file, 'w') as f:
            json.dump({"server": {"port": 99999}}, f)

        manager = ConfigManager(str(temp_config_file))
        errors = manager.validate()

        assert len(errors) > 0, "Should have validation errors for invalid port range"
        assert any("port" in str(e).lower() for e in errors), "Should error about port range"

    def test_validate_boolean_fields(self, temp_config_file):
        """Test validation checks boolean field types."""
        # Create config with invalid boolean
        with open(temp_config_file, 'w') as f:
            json.dump({
                "server": {"port": 8080},
                "ui": {"auto_scroll": "not_boolean"}
            }, f)

        manager = ConfigManager(str(temp_config_file))
        errors = manager.validate()

        assert len(errors) > 0, "Should have validation errors for invalid boolean"
        assert any("auto_scroll" in str(e).lower() or "boolean" in str(e).lower() for e in errors)

    def test_validate_returns_all_errors(self, temp_config_file):
        """Test validation returns all errors, not just first one."""
        # Create config with multiple errors
        with open(temp_config_file, 'w') as f:
            json.dump({
                "server": {"port": "invalid", "host": 123},  # Two errors
                "ui": {"auto_scroll": "yes"}  # Another error
            }, f)

        manager = ConfigManager(str(temp_config_file))
        errors = manager.validate()

        assert len(errors) >= 3, "Should return multiple validation errors"


# Test 8: Test config file creation
class TestConfigFileCreation:
    """Test suite for config file creation and initialization."""

    def test_create_new_config_if_missing(self, temp_config_file):
        """Test that missing config.json is created with defaults."""
        # Ensure file doesn't exist
        if temp_config_file.exists():
            temp_config_file.unlink()

        manager = ConfigManager(str(temp_config_file))
        manager.load()

        assert temp_config_file.exists(), "Config file should be created"

        with open(temp_config_file, 'r') as f:
            config = json.load(f)

        assert isinstance(config, dict), "Created config should be a dict"
        assert "server" in config, "Should have default server section"

    def test_created_config_has_default_values(self, temp_config_file):
        """Test that created config has correct default values."""
        if temp_config_file.exists():
            temp_config_file.unlink()

        manager = ConfigManager(str(temp_config_file))
        config = manager.load()

        # Check default values
        assert config["server"]["port"] == 8080, "Should have default port"
        assert config["server"]["host"] == "0.0.0.0", "Should have default host"
        assert config["ui"]["theme"] == "default", "Should have default theme"

    def test_create_parent_directories_if_needed(self, tmp_path):
        """Test that parent directories are created."""
        nested_path = tmp_path / "level1" / "level2" / "level3" / "config.json"

        manager = ConfigManager(str(nested_path))
        manager.load()

        assert nested_path.exists(), "Config file should be created"
        assert nested_path.parent.exists(), "Parent directories should be created"

    def test_create_with_custom_defaults(self, temp_config_file):
        """Test creating config with custom default values."""
        if temp_config_file.exists():
            temp_config_file.unlink()

        custom_defaults = {
            "server": {"port": 3000},
            "custom": {"key": "value"}
        }

        manager = ConfigManager(str(temp_config_file), default_config=custom_defaults)
        config = manager.load()

        assert config["server"]["port"] == 3000, "Should use custom default"
        assert "custom" in config, "Should have custom section"

    def test_create_without_overwriting_existing(self, sample_config_file):
        """Test that existing config is not overwritten."""
        manager = ConfigManager(str(sample_config_file))
        config = manager.load()

        # Should preserve existing values, not create defaults
        assert config["server"]["port"] == 9090, "Should preserve existing port"
        assert config["ui"]["theme"] == "dark", "Should preserve existing theme"

    def test_create_file_with_proper_permissions(self, temp_config_file):
        """Test that created file is readable and writable."""
        if temp_config_file.exists():
            temp_config_file.unlink()

        manager = ConfigManager(str(temp_config_file))
        manager.load()

        # Verify file is readable
        assert os.access(temp_config_file, os.R_OK), "File should be readable"

        # Verify file is writable
        assert os.access(temp_config_file, os.W_OK), "File should be writable"


# Edge case tests
class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_handle_concurrent_reads(self, config_manager):
        """Test that concurrent reads are handled safely."""
        config1 = config_manager.load()
        config2 = config_manager.load()

        assert config1 == config2, "Concurrent reads should return same data"

    def test_handle_large_config_values(self, temp_config_file):
        """Test handling of large config values."""
        large_string = "x" * 10000

        with open(temp_config_file, 'w') as f:
            json.dump({"large_key": large_string}, f)

        manager = ConfigManager(str(temp_config_file))
        value = manager.get("large_key")

        assert value == large_string, "Should handle large values"

    def test_handle_special_characters_in_keys(self, temp_config_file):
        """Test handling of special characters in keys."""
        config = {
            "key-with-dash": "value1",
            "key_with_underscore": "value2",
            "key.with.dots": "value3"
        }

        with open(temp_config_file, 'w') as f:
            json.dump(config, f)

        manager = ConfigManager(str(temp_config_file))

        assert manager.get("key-with-dash") == "value1"
        assert manager.get("key_with_underscore") == "value2"
        assert manager.get("key.with.dots") == "value3"

    def test_handle_unicode_values(self, temp_config_file):
        """Test handling of unicode characters in values."""
        unicode_config = {
            "emoji": "😀🎉",
            "chinese": "中文",
            "arabic": "العربية",
            "russian": "Русский"
        }

        with open(temp_config_file, 'w', encoding='utf-8') as f:
            json.dump(unicode_config, f, ensure_ascii=False)

        manager = ConfigManager(str(temp_config_file))

        assert manager.get("emoji") == "😀🎉"
        assert manager.get("chinese") == "中文"
        assert manager.get("arabic") == "العربية"
        assert manager.get("russian") == "Русский"

    def test_handle_none_values(self, temp_config_file):
        """Test handling of None values in config."""
        config = {
            "null_key": None,
            "server": {
                "port": 8080,
                "optional": None
            }
        }

        with open(temp_config_file, 'w') as f:
            json.dump(config, f)

        manager = ConfigManager(str(temp_config_file))

        assert manager.get("null_key") is None
        assert manager.get("server.optional") is None

    def test_reload_config_after_external_change(self, config_manager, temp_config_file):
        """Test reloading config after external modification."""
        # Get initial value
        initial_port = config_manager.get("server.port")

        # Externally modify file
        with open(temp_config_file, 'r') as f:
            config = json.load(f)
        config["server"]["port"] = 1111
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)

        # Reload and verify
        config_manager.load()
        new_port = config_manager.get("server.port")

        assert new_port == 1111, "Should reflect external changes after reload"
        assert new_port != initial_port, "Value should be different from initial"


class TestAtomicConfigPersistence:
    """Config commits share the atomic writer: failures never truncate."""

    def test_failed_save_preserves_previous_file_bytes(self, tmp_path, monkeypatch):
        import utils.atomic_io as atomic_io
        config_file = tmp_path / 'config.json'
        manager = ConfigManager(str(config_file))
        manager.load()
        original = config_file.read_text(encoding='utf-8')

        def broken_replace(src, dst):
            raise OSError('replace blocked')

        monkeypatch.setattr(atomic_io.os, 'replace', broken_replace)
        with pytest.raises(OSError, match='replace blocked'):
            manager.set('server.port', 9999)

        assert config_file.read_text(encoding='utf-8') == original

    def test_unknown_keys_survive_saves(self, tmp_path):
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps({
            'server': {'port': 8080},
            'future_section': {'unknown_setting': 'keep me'},
        }), encoding='utf-8')

        manager = ConfigManager(str(config_file))
        manager.load()
        manager.set('server.port', 9090)

        with open(config_file, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        assert saved['future_section'] == {'unknown_setting': 'keep me'}
        assert saved['server']['port'] == 9090

    def test_save_failure_leaves_in_memory_state_documented(self, tmp_path, monkeypatch):
        """On save failure the file keeps old bytes; in-memory holds the new.

        This is the documented contract (see ConfigManager._save_to_file):
        the discrepancy lasts until the next successful save or a reload.
        """
        import utils.atomic_io as atomic_io
        config_file = tmp_path / 'config.json'
        manager = ConfigManager(str(config_file))
        manager.load()

        monkeypatch.setattr(atomic_io.os, 'fsync',
                            lambda fd: (_ for _ in ()).throw(OSError('disk full')))
        with pytest.raises(OSError, match='disk full'):
            manager.set('server.port', 1234)

        # In-memory value took the change even though the file did not.
        assert manager.get('server.port') == 1234
        with open(config_file, 'r', encoding='utf-8') as f:
            assert json.load(f)['server']['port'] == 8080


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
