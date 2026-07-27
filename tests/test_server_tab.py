"""Headless tests for ServerTab presets and automatic model switching."""
from pathlib import Path
from unittest.mock import Mock, patch

from server_tab import (
    LONG_CONTEXT_PRESET,
    TRANSLATION_PRESET,
    ServerTab,
)


class FakeVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeConfig:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


class FakeServer:
    def __init__(self, running=False, events=None):
        self.running = running
        self.events = events if events is not None else []
        self.start_kwargs = None

    def stop(self):
        self.events.append("stop")
        self.running = False

    def start(self, **kwargs):
        self.events.append("start")
        self.start_kwargs = kwargs
        self.running = True


def _make_tab(model_path: Path, *, server=None, active_model=None):
    tab = object.__new__(ServerTab)
    tab._server = server or FakeServer()
    tab._models = Mock()
    tab._models.get_model_path.return_value = str(model_path)
    tab._config = FakeConfig({
        "server.host": "0.0.0.0",
        "server.gpu_layers": 99,
        "server.context_size": 16384,
        "server.batch_size": 512,
        "server.parallel": 3,
        "server.flash_attn": True,
        "server.cont_batching": True,
        "server.cache_type_k": "q8_0",
        "server.cache_type_v": "q8_0",
    })
    tab._active_model_name = active_model
    tab._monitor_running = False
    tab._monitor_generation = 0
    tab._on_server_state_changed = Mock()

    tab.model_var = FakeVar("new-model")
    tab.port_var = FakeVar(8080)
    tab.gpu_layers_var = FakeVar(99)
    tab.context_var = FakeVar(16384)
    tab.batch_var = FakeVar(512)
    tab.parallel_var = FakeVar(3)
    tab.flash_attn_var = FakeVar(True)
    tab.cache_type_k_var = FakeVar("q8_0")
    tab.cache_type_v_var = FakeVar("q8_0")

    tab.start_button = Mock()
    tab.stop_button = Mock()
    tab.status_label = Mock()
    tab.start_resource_monitor = Mock()
    tab.stop_resource_monitor = Mock()
    tab.log = Mock()
    return tab


class TestServerPresets:
    def test_translation_preset_updates_fields_and_config(self, tmp_path):
        tab = _make_tab(tmp_path / "model.gguf")

        tab._apply_translation_preset()

        assert tab.context_var.get() == 16384
        assert tab.batch_var.get() == 512
        assert tab.parallel_var.get() == 3
        assert tab.flash_attn_var.get() is True
        assert tab.cache_type_k_var.get() == "q8_0"
        assert tab.cache_type_v_var.get() == "q8_0"
        for key, value in TRANSLATION_PRESET.items():
            assert tab._config.get(f"server.{key}") == value

    def test_long_context_preset_updates_fields_and_config(self, tmp_path):
        tab = _make_tab(tmp_path / "model.gguf")

        tab._apply_long_context_preset()

        assert tab.context_var.get() == 65536
        assert tab.batch_var.get() == 512
        assert tab.parallel_var.get() == 1
        assert tab.flash_attn_var.get() is True
        assert tab.cache_type_k_var.get() == "f16"
        assert tab.cache_type_v_var.get() == "f16"
        for key, value in LONG_CONTEXT_PRESET.items():
            assert tab._config.get(f"server.{key}") == value


class TestAutomaticModelSwitch:
    @patch("server_tab.messagebox.showwarning")
    def test_same_running_model_is_not_restarted(self, showwarning, tmp_path):
        server = FakeServer(running=True)
        tab = _make_tab(
            tmp_path / "model.gguf", server=server, active_model="new-model"
        )

        tab.start_server()

        showwarning.assert_called_once()
        assert server.events == []

    @patch("server_tab.gc.collect")
    def test_switch_stops_collects_then_starts_new_model(self, collect, tmp_path):
        model_path = tmp_path / "new.gguf"
        model_path.write_bytes(b"gguf")
        events = []
        server = FakeServer(running=True, events=events)
        tab = _make_tab(model_path, server=server, active_model="old-model")
        collect.side_effect = lambda: events.append("gc")

        assert tab._do_start_server("new-model", 8080) is True

        assert events == ["stop", "gc", "start"]
        assert tab._active_model_name == "new-model"
        assert server.start_kwargs["model_path"] == str(model_path)
        assert server.start_kwargs["parallel"] == 3
        assert server.start_kwargs["cache_type_k"] == "q8_0"
        tab.start_button.config.assert_called_with(
            state="normal", text="\U0001F504 切換模型"
        )
        tab._on_server_state_changed.assert_any_call(False, None, None)
        tab._on_server_state_changed.assert_any_call(
            True, "new-model", 8080
        )

    @patch("server_tab.messagebox.showerror")
    @patch("server_tab.gc.collect")
    def test_failed_stop_cancels_switch(self, collect, showerror, tmp_path):
        model_path = tmp_path / "new.gguf"
        model_path.write_bytes(b"gguf")
        server = FakeServer(running=True)
        server.stop = Mock(side_effect=RuntimeError("cannot stop"))
        tab = _make_tab(model_path, server=server, active_model="old-model")

        assert tab._do_start_server("new-model", 8080) is False

        showerror.assert_called_once()
        collect.assert_not_called()
        assert "start" not in server.events
        assert tab._active_model_name == "old-model"

    def test_manual_stop_resets_active_model_and_button(self, tmp_path):
        server = FakeServer(running=True)
        tab = _make_tab(
            tmp_path / "model.gguf", server=server, active_model="old-model"
        )

        assert tab._do_stop_server() is True

        assert tab._active_model_name is None
        tab.start_button.config.assert_called_with(
            state="normal", text="▶️ 啟動服務器"
        )
        tab.stop_button.config.assert_called_with(state="disabled")


class TestResourceMonitorGeneration:
    @patch("server_tab.threading.Thread")
    def test_restart_uses_new_generation_to_retire_old_thread(self, thread, tmp_path):
        tab = _make_tab(tmp_path / "model.gguf")

        ServerTab.start_resource_monitor(tab)
        first_generation = thread.call_args.kwargs["args"][0]
        ServerTab.stop_resource_monitor(tab)
        ServerTab.start_resource_monitor(tab)
        second_generation = thread.call_args.kwargs["args"][0]

        assert first_generation == 1
        assert second_generation == 3
        assert second_generation != first_generation
        assert thread.return_value.start.call_count == 2
