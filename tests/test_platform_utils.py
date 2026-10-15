from pathlib import Path

from platform_utils import open_path


def test_open_path_uses_macos_open(tmp_path):
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))

    target = tmp_path / "output"
    open_path(target, platform="darwin", popen=fake_popen)

    assert calls[0][0] == ["open", str(target)]
    assert calls[0][1]["start_new_session"] is True


def test_open_path_uses_linux_xdg_open(tmp_path):
    calls = []
    open_path(tmp_path, platform="linux",
              popen=lambda command, **kwargs: calls.append(command))

    assert calls == [["xdg-open", str(tmp_path)]]


def test_open_path_uses_windows_startfile(tmp_path):
    calls = []
    open_path(tmp_path, platform="win32", startfile=calls.append)

    assert calls == [str(tmp_path)]
