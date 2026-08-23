from pathlib import Path
from typing import Optional


class BaseModelRegistry:
    config_key: str
    glob_pattern: str
    size_divisor: int
    size_unit: str
    format_name: str

    def __init__(self, scan_dir: Path, config_manager):
        self.scan_dir = Path(scan_dir)
        self.config_manager = config_manager

    def _build_model_info(self, filepath: Path) -> dict:
        size_value = filepath.stat().st_size / (1024 ** self.size_divisor)
        return {
            "name": self._parse_name(filepath),
            "path": str(filepath),
            "size": f"{size_value:.2f}{self.size_unit}",
            "format": self.format_name,
        }

    def _parse_name(self, filepath: Path) -> str:
        return filepath.stem

    def _is_scan_candidate(self, filepath: Path) -> bool:
        return True

    def scan(self) -> tuple:
        if not self.scan_dir.exists():
            return [], 0, 0

        files = [
            path for path in self.scan_dir.glob(self.glob_pattern)
            if self._is_scan_candidate(path)
        ]

        current_models = self.config_manager.get(self.config_key, {}).get("models", [])
        models_to_keep = []
        removed_count = 0

        for model in current_models:
            model_path = model.get("path", "")
            if model_path and Path(model_path).exists():
                models_to_keep.append(model)
            else:
                removed_count += 1

        added_count = 0
        seen_paths = {m.get("path", "") for m in models_to_keep}
        for f in files:
            if str(f) not in seen_paths:
                models_to_keep.append(self._build_model_info(f))
                added_count += 1

        self.config_manager.set(self.config_key, {"models": models_to_keep})
        return models_to_keep, added_count, removed_count

    def list_models(self) -> list:
        return self.config_manager.get(self.config_key, {}).get("models", [])

    def add_model(self, path: str) -> dict:
        filepath = Path(path)
        model_info = self._build_model_info(filepath)
        models_data = self.list_models()
        models_data.append(model_info)
        self.config_manager.set(self.config_key, {"models": models_data})
        return model_info

    @staticmethod
    def find_model_path(models: list, name: str) -> Optional[str]:
        """Return the stored path of the named model in a model list, or None."""
        for model in models:
            if model.get("name") == name:
                return model.get("path")
        return None

    def get_model_path(self, name: str) -> Optional[str]:
        """Return the stored path of the named model, or None."""
        return self.find_model_path(self.list_models(), name)

    @staticmethod
    def resolve_model_path(models: list, model_dir: str, model_name: str) -> str:
        """Resolve a model name to a path against a model list.

        A registered model wins; otherwise the name is treated as a file
        inside model_dir. An empty dir or name passes through unchanged.
        """
        if not model_dir or not model_name:
            return model_name
        found = BaseModelRegistry.find_model_path(models, model_name)
        if found:
            return found
        return str(Path(model_dir) / model_name)
