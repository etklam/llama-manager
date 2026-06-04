"""
Simple test script to verify ConfigManager works correctly.
"""
import json
import os
import tempfile
from pathlib import Path
from config_manager import ConfigManager

def test_basic_functionality():
    """Test basic ConfigManager functionality."""
    print("Testing ConfigManager...")

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.json"

        # Test 1: Create new config with defaults
        print("\n1. Testing config creation with defaults...")
        manager = ConfigManager(str(config_path))
        config = manager.load()

        assert config["server"]["port"] == 8080, "Default port should be 8080"
        assert config["server"]["host"] == "0.0.0.0", "Default host should be 0.0.0.0"
        assert config["ui"]["theme"] == "default", "Default theme should be 'default'"
        assert config_path.exists(), "Config file should be created"
        print("[OK] Config creation with defaults works")

        # Test 2: Get values
        print("\n2. Testing get operations...")
        port = manager.get("server.port")
        assert port == 8080, "Should get correct port"
        theme = manager.get("ui.theme")
        assert theme == "default", "Should get correct theme"
        missing = manager.get("missing.key", "default")
        assert missing == "default", "Should return default for missing key"
        print("[OK] Get operations work")

        # Test 3: Set values with auto-save
        print("\n3. Testing set operations...")
        manager.set("server.port", 9999)
        assert manager.get("server.port") == 9999, "Port should be updated"

        # Verify it was saved to file
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        assert saved_config["server"]["port"] == 9999, "Value should be saved to file"
        print("[OK] Set operations with auto-save work")

        # Test 4: Set without auto-save
        print("\n4. Testing set without auto-save...")
        manager.set("server.port", 8888, auto_save=False)
        assert manager.get("server.port") == 8888, "In-memory value should change"

        # File should still have old value
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        assert saved_config["server"]["port"] == 9999, "File should not change without auto-save"
        print("[OK] Set without auto-save works")

        # Test 5: Nested key creation
        print("\n5. Testing nested key creation...")
        manager.set("new.nested.value", 123)
        assert manager.get("new.nested.value") == 123, "Should create nested structure"
        print("[OK] Nested key creation works")

        # Test 6: Translation config via generic get/set
        print("\n6. Testing translation config via get/set...")
        trans_config = manager.get("translation")
        assert "enabled" in trans_config, "Should have enabled key"
        assert "provider" in trans_config, "Should have provider key"

        manager.set("translation.enabled", True)
        manager.set("translation.provider", "test")
        assert manager.get("translation.enabled") is True, "Should update translation config"
        assert manager.get("translation.provider") == "test", "Should update provider"
        print("[OK] Translation config operations work")

        # Test 7: Validation
        print("\n7. Testing validation...")
        manager.set("server.port", 8080)
        errors = manager.validate()
        assert len(errors) == 0, "Valid config should have no errors"

        manager.set("server.port", 99999)  # Invalid port
        errors = manager.validate()
        assert len(errors) > 0, "Invalid config should have errors"
        assert any("port" in str(e).lower() for e in errors), "Should error about port"
        print("[OK] Validation works")

        # Test 8: Save and load
        print("\n8. Testing save and load...")
        new_config = {
            "server": {"port": 7777, "host": "localhost"},
            "ui": {"theme": "dark"}
        }
        manager.save(new_config)

        manager2 = ConfigManager(str(config_path))
        loaded_config = manager2.load()
        assert loaded_config["server"]["port"] == 7777, "Should load saved config"
        assert loaded_config["ui"]["theme"] == "dark", "Should load saved config"
        print("[OK] Save and load work")

        # Test 9: Config merging
        print("\n9. Testing config merging...")
        config_path2 = Path(tmpdir) / "test_config2.json"
        with open(config_path2, 'w') as f:
            json.dump({"server": {"port": 9090}}, f)

        manager3 = ConfigManager(str(config_path2))
        merged_config = manager3.load()
        assert merged_config["server"]["port"] == 9090, "User value should override default"
        assert "host" in merged_config["server"], "Should merge defaults"
        assert merged_config["server"]["host"] == "0.0.0.0", "Should use default host"
        print("[OK] Config merging works")

        # Test 10: Unicode and special characters
        print("\n10. Testing unicode and special characters...")
        manager.set("emoji", "😀")
        manager.set("chinese", "中文")
        manager.set("arabic", "العربية")
        assert manager.get("emoji") == "😀", "Should handle emoji"
        assert manager.get("chinese") == "中文", "Should handle chinese"
        assert manager.get("arabic") == "العربية", "Should handle arabic"
        print("[OK] Unicode and special characters work")

        # Test 11: Empty config
        print("\n11. Testing empty config file...")
        config_path3 = Path(tmpdir) / "test_config3.json"
        with open(config_path3, 'w') as f:
            json.dump({}, f)

        manager4 = ConfigManager(str(config_path3))
        empty_config = manager4.load()
        assert "server" in empty_config, "Should add default sections"
        assert "ui" in empty_config, "Should add default sections"
        assert "translation" in empty_config, "Should add default sections"
        print("[OK] Empty config gets defaults")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)

if __name__ == "__main__":
    test_basic_functionality()
