import os
from pathlib import Path

import pytest

from tool_manager import ToolManager


class Config:
    def __init__(self):
        self.values = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


def test_configured_tool_path_wins(tmp_path):
    config = Config()
    selected = tmp_path / "custom-server"
    config.set("paths.server_exe", str(selected))

    manager = ToolManager(config, tmp_path / "data", platform="linux")

    assert manager.resolve("llama-server") == selected


def test_bundled_default_is_platform_specific(tmp_path):
    manager = ToolManager(Config(), tmp_path / "data", platform="win32")

    assert manager.default_path("llama-server").name == "llama-server.exe"
    assert manager.default_path("ffprobe").name == "ffprobe.exe"


def test_check_requires_posix_execute_permission(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX execute permission is not a Windows check")
    executable = tmp_path / "ffmpeg"
    executable.write_text("#!/bin/sh\n")
    os.chmod(executable, 0o644)
    manager = ToolManager(Config(), tmp_path / "data", platform="linux")

    status = manager.check("ffmpeg", executable)

    assert status.available is False
    assert "執行權限" in status.message


def test_set_path_uses_existing_config_key(tmp_path):
    config = Config()
    manager = ToolManager(config, tmp_path / "data")

    manager.set_path("ffprobe", Path("/opt/ffprobe"))

    assert config.get("paths.ffprobe") == "/opt/ffprobe"
