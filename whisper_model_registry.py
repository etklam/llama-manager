"""
Whisper Model Registry Module

Manages model discovery, storage, and retrieval for GGML whisper model files.
"""

from pathlib import Path

from base_registry import BaseModelRegistry


class WhisperModelRegistry(BaseModelRegistry):
    config_key = "whisper_models"
    glob_pattern = "*.bin"
    size_divisor = 2
    size_unit = "MB"
    format_name = "GGML"

    def _parse_name(self, filepath: Path) -> str:
        return self.parse_model_name(filepath.name)

    @staticmethod
    def parse_model_name(filename: str) -> str:
        name = filename
        if name.endswith(".bin"):
            name = name[:-4]

        if name.startswith("ggml-whisper-"):
            name = name[len("ggml-whisper-"):]
        elif name.startswith("ggml-"):
            name = name[len("ggml-"):]

        if name.endswith(".en"):
            name = name[:-3] + " (en)"

        return name
