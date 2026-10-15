"""Platform paths used by the desktop application.

Keep this module small and dependency-free.  Packaged resources are read from
the application bundle; settings, models, and downloaded tools belong in a
user-writable data directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping, Optional


APP_NAME = "llama-manager"


def resource_dir(module_file: Optional[str] = None) -> Path:
    """Return the read-only directory containing application resources."""
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled).resolve()
    return Path(module_file or __file__).resolve().parent


def user_data_dir(
    app_name: str = APP_NAME,
    *,
    platform: Optional[str] = None,
    home: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Path:
    """Return the conventional writable data directory for one platform.

    ``platform``, ``home`` and ``environ`` are injectable for unit tests.  The
    Linux path follows XDG_DATA_HOME; no third-party platform-path package is
    needed for these three targets.
    """
    platform = platform or sys.platform
    home = Path(home) if home is not None else Path.home()
    environ = os.environ if environ is None else environ

    if platform.startswith("win"):
        root = environ.get("LOCALAPPDATA") or environ.get("APPDATA")
        root = Path(root) if root else home / "AppData" / "Local"
    elif platform == "darwin":
        root = home / "Library" / "Application Support"
    else:
        root = Path(environ["XDG_DATA_HOME"]) if environ.get("XDG_DATA_HOME") else home / ".local" / "share"
    return root / app_name


def executable_name(name: str, *, platform: Optional[str] = None) -> str:
    """Add the Windows executable suffix without changing POSIX names."""
    platform = platform or sys.platform
    return name + ".exe" if platform.startswith("win") and not name.endswith(".exe") else name


def default_tool_paths(data_dir: Path, *, platform: Optional[str] = None) -> dict[str, Path]:
    """Return portable defaults; these are hints, not claims that tools exist."""
    data_dir = Path(data_dir)
    runtime_dir = data_dir / "tools" / "llama" / "bin"
    return {
        "base_dir": data_dir / "tools" / "llama",
        "runtime_dir": runtime_dir,
        "model_dir": data_dir / "models" / "llm",
        "whisper_model_dir": data_dir / "models" / "whisper",
        "server_exe": runtime_dir / executable_name("llama-server", platform=platform),
    }
