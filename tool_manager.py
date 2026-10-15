"""Small, dependency-free manager for external command line tools.

The desktop app has four tools that can be supplied by a bundled runtime, a
user-selected build, or the operating system PATH.  Keeping that resolution in
one place prevents each tab from growing a different set of platform rules.
Downloads are deliberately outside this module for now; a tool can already be
selected, checked, and used without an installer.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app_paths import default_tool_paths, executable_name


TOOL_NAMES = ("llama-server", "whisper-cli", "ffmpeg", "ffprobe")


@dataclass(frozen=True)
class ToolStatus:
    """Result of checking one executable."""

    name: str
    path: Optional[Path]
    available: bool
    message: str


class ToolManager:
    """Resolve and persist paths for the tools used by the mainline flows."""

    _CONFIG_KEYS = {
        "llama-server": "paths.server_exe",
        "whisper-cli": "whisper.cli_path",
        "ffmpeg": "paths.ffmpeg",
        "ffprobe": "paths.ffprobe",
    }

    def __init__(self, config_manager, data_dir: Path, *, platform: str | None = None):
        self.config = config_manager
        self.data_dir = Path(data_dir)
        self.platform = platform
        self._defaults = default_tool_paths(self.data_dir, platform=platform)

    def configured_path(self, name: str) -> str:
        self._validate_name(name)
        return str(self.config.get(self._CONFIG_KEYS[name], "") or "").strip()

    def set_path(self, name: str, path: str | Path) -> None:
        """Persist a user-selected path; an empty value restores auto-detect."""
        self._validate_name(name)
        value = str(path).strip()
        self.config.set(self._CONFIG_KEYS[name], value)

    def default_path(self, name: str) -> Path:
        self._validate_name(name)
        configured_runtime = self.config.get("paths.runtime_dir", "")
        runtime_dir = (Path(configured_runtime).expanduser()
                       if configured_runtime else self._defaults["runtime_dir"])
        if name == "llama-server":
            return runtime_dir / executable_name("llama-server", platform=self.platform)
        if name == "whisper-cli":
            return runtime_dir / executable_name(
                "whisper-cli", platform=self.platform)
        tool_dir = self.data_dir / "tools" / "ffmpeg" / "bin"
        return tool_dir / executable_name(name, platform=self.platform)

    def resolve(self, name: str) -> Path:
        """Return configured, bundled, PATH, or the conventional default path."""
        self._validate_name(name)
        configured = self.configured_path(name)
        if configured:
            return Path(configured).expanduser()

        bundled = self.default_path(name)
        if bundled.is_file():
            return bundled

        path_name = executable_name(name, platform=self.platform)
        on_path = shutil.which(path_name) or (shutil.which(name) if path_name != name else None)
        return Path(on_path) if on_path else bundled

    def check(self, name: str, path: str | Path | None = None,
              *, probe: bool = False) -> ToolStatus:
        """Check a tool; optionally launch ``--version`` for loader diagnostics."""
        self._validate_name(name)
        candidate = Path(path).expanduser() if path else self.resolve(name)

        # A bare command is valid when it resolves through PATH.  This is useful
        # on macOS/Linux and keeps advanced Windows configurations working too.
        if candidate.parent == Path("."):
            resolved = shutil.which(str(candidate))
            if resolved:
                candidate = Path(resolved)
            else:
                return ToolStatus(name, candidate, False,
                                  f"找不到 {candidate}（不在 PATH）")

        if not candidate.exists():
            return ToolStatus(name, candidate, False,
                              f"檔案不存在：{candidate}")
        if not candidate.is_file():
            return ToolStatus(name, candidate, False,
                              f"不是檔案：{candidate}")
        if os.name != "nt" and not os.access(candidate, os.X_OK):
            return ToolStatus(name, candidate, False,
                              f"沒有執行權限：{candidate}")
        if probe:
            try:
                result = subprocess.run(
                    [str(candidate), "--version"], capture_output=True,
                    text=True, timeout=5, encoding="utf-8", errors="replace")
            except (OSError, subprocess.SubprocessError) as error:
                return ToolStatus(name, candidate, False,
                                  f"無法啟動 {candidate}：{error}")
            output = (result.stdout or result.stderr or "").strip()
            version = output.splitlines()[0] if output else "未回報版本"
            return ToolStatus(name, candidate, True,
                              f"可用：{version}")
        return ToolStatus(name, candidate, True, "可用")

    @staticmethod
    def _validate_name(name: str) -> None:
        if name not in TOOL_NAMES:
            raise ValueError(f"Unknown tool: {name}")
