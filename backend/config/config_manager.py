"""
Configuration Manager for RAG Application.

This module handles loading, validating, and providing access to application configuration.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Manages application configuration from JSON file."""

    def __init__(self, config_path: str = "backend/config/config.json"):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration JSON file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            json.JSONDecodeError: If config file is invalid JSON.
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self.load_config()
        self.validate_config()

    def load_config(self) -> None:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self._config = json.load(f)

    def validate_config(self) -> None:
        """
        Validate that required configuration sections exist.

        Raises:
            ValueError: If required configuration is missing.
        """
        required_sections = ['llm', 'embeddings', 'vector_db', 'chunking', 'retrieval', 'logging']

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate LLM config
        if 'provider' not in self._config['llm']:
            raise ValueError("LLM provider not specified in configuration")

        # Validate embeddings config
        if 'provider' not in self._config['embeddings']:
            raise ValueError("Embeddings provider not specified in configuration")

        # Validate vector DB config
        if 'type' not in self._config['vector_db']:
            raise ValueError("Vector DB type not specified in configuration")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation).

        Args:
            key: Configuration key (e.g., 'llm.provider' or 'chunking.chunk_size')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self._config.get('llm', {})

    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration."""
        return self._config.get('embeddings', {})

    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration."""
        return self._config.get('vector_db', {})

    def get_chunking_config(self) -> Dict[str, Any]:
        """Get chunking configuration."""
        return self._config.get('chunking', {})

    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration."""
        return self._config.get('retrieval', {})

    def get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return self._config.get('system_prompt', 'You are a helpful assistant.')

    def get_auth_config(self) -> Dict[str, Any]:
        """Get authentication configuration."""
        return self._config.get('authentication', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration values and save to file.

        Args:
            updates: Dictionary of configuration updates (supports nested updates)
        """
        def update_nested(base: dict, updates: dict) -> dict:
            """Recursively update nested dictionary."""
            for key, value in updates.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    base[key] = update_nested(base[key], value)
                else:
                    base[key] = value
            return base

        self._config = update_nested(self._config, updates)
        self.save_config()

    def save_config(self) -> None:
        """Save current configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary."""
        return self._config.copy()


# Global configuration instance
_config_instance: Optional[ConfigManager] = None


def get_config(config_path: str = "backend/config/config.json") -> ConfigManager:
    """
    Get or create the global configuration instance.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigManager instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance
