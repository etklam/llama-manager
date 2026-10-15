from pathlib import Path

from app_paths import default_tool_paths, executable_name, user_data_dir
from config_manager import ConfigManager


def test_user_data_dir_follows_each_supported_platform(tmp_path):
    env = {"LOCALAPPDATA": str(tmp_path / "local"), "APPDATA": str(tmp_path / "roaming")}
    assert user_data_dir(platform="win32", home=tmp_path, environ=env) == tmp_path / "local" / "llama-manager"
    assert user_data_dir(platform="darwin", home=tmp_path, environ={}) == tmp_path / "Library" / "Application Support" / "llama-manager"
    assert user_data_dir(platform="linux", home=tmp_path, environ={}) == tmp_path / ".local" / "share" / "llama-manager"


def test_xdg_data_home_overrides_linux_default(tmp_path):
    root = tmp_path / "xdg"
    assert user_data_dir(platform="linux", home=tmp_path,
                         environ={"XDG_DATA_HOME": str(root)}) == root / "llama-manager"


def test_tool_defaults_are_platform_named_and_user_writable(tmp_path):
    paths = default_tool_paths(tmp_path / "data", platform="darwin")
    assert paths["model_dir"] == tmp_path / "data" / "models" / "llm"
    assert paths["whisper_model_dir"] == tmp_path / "data" / "models" / "whisper"
    assert paths["server_exe"].name == "llama-server"
    assert executable_name("llama-server", platform="win32") == "llama-server.exe"


def test_config_imports_legacy_resource_once(tmp_path):
    legacy = tmp_path / "resource" / "config.json"
    target = tmp_path / "data" / "config.json"
    legacy.parent.mkdir()
    legacy.write_text('{"server": {"port": 9012}}', encoding="utf-8")

    manager = ConfigManager(str(target), legacy_paths=[str(legacy)])
    assert manager.load()["server"]["port"] == 9012
    assert target.exists()
    legacy.write_text('{"server": {"port": 9013}}', encoding="utf-8")
    assert ConfigManager(str(target), legacy_paths=[str(legacy)]).load()["server"]["port"] == 9012
