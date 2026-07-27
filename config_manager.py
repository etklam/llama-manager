"""
Configuration Manager Module

Provides comprehensive configuration management with JSON persistence,
nested key access, validation, and default config merging.
"""

import json
import copy
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


# Default configuration structure
DEFAULT_CONFIG = {
    'server': {
        'port': 8080,
        'host': '0.0.0.0',
        'gpu_layers': 99,
        'context_size': 16384,
        'batch_size': 512,
        'threads': -1,
        'parallel': 3,
        'flash_attn': True,
        'cont_batching': True,
        'cache_type_k': 'q8_0',
        'cache_type_v': 'q8_0'
    },
    'ui': {
        'theme': 'default',
        'auto_scroll': True,
        'last_model': '',
        'last_source_lang': 'auto',
        'last_target_lang': 'zh-cn'
    },
    'translation': {
        'enabled': False,
        'source_lang': 'auto',
        'target_lang': 'en',
        'provider': 'mock'
    },
    'whisper': {
        'cli_path': '',
        'model_dir': '',
        'last_model': '',
        'language': 'auto',
        'threads': 8
    }
}


class ConfigManager:
    """
    Manages application configuration with JSON persistence.

    Features:
    - Load/save JSON configuration files
    - Nested key access with dot notation (e.g., 'server.port')
    - Auto-save functionality
    - Default config merging
    - Translation config section management via generic get/set
    - Config validation
    - Thread-safe operations
    """

    def __init__(self, config_path: str = 'config.json', default_config: Optional[dict] = None):
        """
        Initialize ConfigManager.

        Args:
            config_path: Path to configuration file (default: 'config.json')
            default_config: Custom default config (optional, uses DEFAULT_CONFIG if None)
        """
        self.config_path = Path(config_path)
        self.default_config = default_config if default_config is not None else copy.deepcopy(DEFAULT_CONFIG)
        self._config: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file.

        Creates the file with defaults if missing.
        Merges user config with default config.

        Returns:
            Loaded and merged configuration dictionary

        Raises:
            json.JSONDecodeError: If file contains invalid JSON
            ValueError: If file content is invalid
        """
        with self._lock:
            # Create parent directories if they don't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # If file doesn't exist, create it with defaults
            if not self.config_path.exists():
                self._config = copy.deepcopy(self.default_config)
                self._save_to_file(self._config)
                return self._config

            # Load existing file
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON in config file {self.config_path}: {e.msg}",
                    e.doc, e.pos
                )

            # Ensure loaded config is a dictionary
            if not isinstance(loaded_config, dict):
                raise ValueError(f"Config file must contain a JSON object, got {type(loaded_config).__name__}")

            # Merge with defaults
            self._config = self._merge_configs(loaded_config, self.default_config)

            return self._config

    def save(self, config: Dict[str, Any]) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary to save
        """
        with self._lock:
            self._config = copy.deepcopy(config)
            self._save_to_file(self._config)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Supports nested keys using dot notation (e.g., 'server.port').
        If nested path fails, tries exact key match to support keys with dots.

        Args:
            key: Configuration key (supports dot notation for nested access)
            default: Default value to return if key is not found

        Returns:
            Configuration value or default if not found
        """
        # Load config if not already loaded
        if not self._config:
            try:
                self.load()
            except (json.JSONDecodeError, ValueError):
                # If loading fails, work with defaults
                self._config = copy.deepcopy(self.default_config)

        # Try nested path first
        if '.' in key:
            value = self._config
            keys = key.split('.')
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    # Nested path failed, try exact match
                    return self._config.get(key, default)
            return value

        # Simple key lookup
        return self._config.get(key, default)

    def set(self, key: str, value: Any, auto_save: bool = True) -> None:
        """
        Set a configuration value by key.

        Supports nested keys using dot notation (e.g., 'server.port').
        Creates nested structure if it doesn't exist.

        Args:
            key: Configuration key (supports dot notation for nested access)
            value: Value to set
            auto_save: If True, automatically save to file after setting
        """
        with self._lock:
            # Load config if not already loaded
            if not self._config:
                self.load()

            # Navigate through nested structure, creating if needed
            keys = key.split('.')
            current = self._config

            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                elif not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]

            # Set the final value
            current[keys[-1]] = value

            # Auto-save if requested
            if auto_save:
                self._save_to_file(self._config)

    def validate(self) -> List[str]:
        """
        Validate the current configuration.

        Returns a list of validation error messages.
        Empty list means configuration is valid.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Load config if not already loaded
        if not self._config:
            try:
                self.load()
            except (json.JSONDecodeError, ValueError):
                return ["Unable to load configuration file"]

        # Validate required sections exist
        if 'server' not in self._config:
            errors.append("Missing required section: 'server'")
        else:
            server_config = self._config['server']

            # Validate server.port
            if 'port' not in server_config:
                errors.append("Missing required key: 'server.port'")
            else:
                port = server_config['port']
                if not isinstance(port, int):
                    errors.append(f"'server.port' must be an integer, got {type(port).__name__}")
                elif port < 1 or port > 65535:
                    errors.append(f"'server.port' must be between 1 and 65535, got {port}")

            # Validate server.host
            if 'host' in server_config and not isinstance(server_config['host'], str):
                errors.append(f"'server.host' must be a string, got {type(server_config['host']).__name__}")

        # Validate ui section if it exists
        if 'ui' in self._config:
            ui_config = self._config['ui']

            # Validate ui.auto_scroll is boolean if present
            if 'auto_scroll' in ui_config:
                if not isinstance(ui_config['auto_scroll'], bool):
                    errors.append(f"'ui.auto_scroll' must be a boolean, got {type(ui_config['auto_scroll']).__name__}")

        # Validate translation section if it exists
        if 'translation' in self._config:
            trans_config = self._config['translation']

            # Validate enabled is boolean if present
            if 'enabled' in trans_config:
                if not isinstance(trans_config['enabled'], bool):
                    errors.append(f"'translation.enabled' must be a boolean, got {type(trans_config['enabled']).__name__}")

        return errors

    def _merge_configs(self, user_config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge user config with default config.

        User values take precedence. Missing values are filled with defaults.
        Preserves extra keys not in defaults.

        Args:
            user_config: User-provided configuration
            default_config: Default configuration

        Returns:
            Merged configuration dictionary
        """
        result = copy.deepcopy(default_config)

        for key, value in user_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._merge_configs(value, result[key])
            else:
                # User value overrides default (including new keys)
                result[key] = copy.deepcopy(value)

        return result

    def _save_to_file(self, config: Dict[str, Any]) -> None:
        """
        Internal method to save configuration to file.

        Args:
            config: Configuration dictionary to save
        """
        # Ensure parent directories exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file with proper formatting
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
