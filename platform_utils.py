"""Tiny platform bridges shared by the desktop surfaces."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional


def open_path(path: str | Path, *, platform: Optional[str] = None,
              startfile: Optional[Callable] = None,
              popen: Optional[Callable] = None) -> None:
    """Ask the native file manager to open a file or directory.

    Tk has no portable equivalent of ``os.startfile``.  The injectable
    callables keep this bridge easy to test without starting a desktop process.
    """
    target = str(Path(path).expanduser())
    platform = platform or sys.platform
    if platform.startswith("win"):
        (startfile or os.startfile)(target)
        return

    command = ["open", target] if platform == "darwin" else ["xdg-open", target]
    launcher = popen or subprocess.Popen
    try:
        launcher(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 start_new_session=True)
    except FileNotFoundError as exc:
        opener = command[0]
        raise RuntimeError(
            f"找不到系統開啟工具 {opener}，請手動開啟：{target}") from exc
