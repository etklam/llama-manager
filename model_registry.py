"""
Model Registry Module

Manages model discovery, storage, and retrieval for GGUF model files.
Provides a clean interface for scanning directories and managing model metadata.
"""

from pathlib import Path
from typing import Optional


class ModelRegistry:
    """
    Manages model discovery and storage for GGUF model files.

    Features:
    - Scan directories for .gguf files
    - Detect quantization formats from filenames
    - Persist model metadata via ConfigManager
    - Add models individually
    - Query model paths by name
    """

    def __init__(self, scan_dir: Path, config_manager):
        """
        Initialize ModelRegistry.

        Args:
            scan_dir: Directory path to scan for .gguf files
            config_manager: ConfigManager instance for persistence
        """
        self.scan_dir = Path(scan_dir)
        self.config_manager = config_manager

    def scan(self) -> tuple:
        """
        Scan for .gguf files in the scan directory.

        Returns:
            Tuple of (models_list, added_count, removed_count)
        """
        # Handle missing scan directory
        if not self.scan_dir.exists():
            return [], 0, 0

        # Find all .gguf files
        gguf_files = list(self.scan_dir.glob("*.gguf"))
        existing_paths = {str(gguf_file) for gguf_file in gguf_files}

        # Get current models from config
        current_models = self.config_manager.get("models", {}).get("models", [])
        models_to_keep = []
        removed_count = 0

        # Check which existing models still exist
        for model in current_models:
            model_path = model.get("path", "")
            if model_path and Path(model_path).exists():
                models_to_keep.append(model)
            else:
                removed_count += 1

        # Add newly discovered models
        added_count = 0
        for gguf_file in gguf_files:
            if str(gguf_file) not in {m.get("path", "") for m in current_models}:
                size_gb = gguf_file.stat().st_size / (1024**3)
                model_info = {
                    "name": gguf_file.stem,
                    "path": str(gguf_file),
                    "size": f"{size_gb:.2f}GB",
                    "format": self.detect_format(gguf_file.name)
                }
                models_to_keep.append(model_info)
                added_count += 1

        # Update config with new model list
        self.config_manager.set("models", {"models": models_to_keep})

        return models_to_keep, added_count, removed_count

    def list_models(self) -> list:
        """
        Return current model list from config.

        Returns:
            List of model info dictionaries
        """
        return self.config_manager.get("models", {}).get("models", [])

    def add_model(self, path: str) -> dict:
        """
        Add a model by path.

        Args:
            path: File path to the model

        Returns:
            Model info dictionary
        """
        gguf_file = Path(path)
        size_gb = gguf_file.stat().st_size / (1024**3)

        model_info = {
            "name": gguf_file.stem,
            "path": str(gguf_file),
            "size": f"{size_gb:.2f}GB",
            "format": self.detect_format(gguf_file.name)
        }

        # Get existing models and add new one
        models_data = self.config_manager.get("models", {}).get("models", [])
        models_data.append(model_info)
        self.config_manager.set("models", {"models": models_data})

        return model_info

    def get_model_path(self, name: str) -> Optional[str]:
        """
        Get the file path for a model by name.

        Args:
            name: Model name

        Returns:
            File path string or None if not found
        """
        models = self.list_models()
        for model in models:
            if model.get("name") == name:
                return model.get("path")
        return None

    @staticmethod
    def detect_format(filename: str) -> str:
        """
        Detect quantization format from filename.

        Args:
            filename: Model filename to analyze

        Returns:
            Detected format string or "Unknown"
        """
        formats = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0",
                  "IQ4_NL", "IQ4_XS", "Q3_K_M", "Q2_K"]
        for fmt in formats:
            if fmt.lower() in filename.lower():
                return fmt
        return "Unknown"
