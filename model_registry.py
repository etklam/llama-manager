"""
Model Registry Module

Manages model discovery, storage, and retrieval for GGUF model files.
"""

from pathlib import Path

from base_registry import BaseModelRegistry


class ModelRegistry(BaseModelRegistry):
    config_key = "models"
    glob_pattern = "*.gguf"
    size_divisor = 3
    size_unit = "GB"
    format_name = "Unknown"

    FORMATS = [
        "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0",
        "IQ4_NL", "IQ4_XS", "Q3_K_M", "Q2_K",
    ]

    def _build_model_info(self, filepath: Path) -> dict:
        info = super()._build_model_info(filepath)
        info["format"] = self.detect_format(filepath.name)
        return info

    @staticmethod
    def detect_format(filename: str) -> str:
        for fmt in ModelRegistry.FORMATS:
            if fmt.lower() in filename.lower():
                return fmt
        return "Unknown"
