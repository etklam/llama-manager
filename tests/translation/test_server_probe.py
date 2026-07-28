"""Tests for the llama-server preflight probe and worker clamp."""
from unittest.mock import patch

import httpx
import pytest

from translation.server_probe import (
    ServerInfo,
    clamp_workers,
    probe_server,
    props_url,
    unreachable_message,
)


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, raise_json=False):
        self.status_code = status_code
        self._payload = payload
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("not json")
        return self._payload


class TestPropsUrl:

    def test_strips_v1_suffix(self):
        assert props_url("http://localhost:8080/v1") == "http://localhost:8080/props"

    def test_strips_trailing_slash(self):
        assert props_url("http://localhost:8080/v1/") == "http://localhost:8080/props"

    def test_bare_root_url(self):
        assert props_url("http://localhost:8080") == "http://localhost:8080/props"

    def test_strips_chat_completions_suffix(self):
        assert props_url(
            "http://localhost:8080/v1/chat/completions"
        ) == "http://localhost:8080/props"

    def test_non_default_host_and_port_preserved(self):
        assert props_url("http://10.0.0.5:9999/v1") == "http://10.0.0.5:9999/props"


class TestProbeServer:

    def test_connection_error_is_not_reachable(self):
        with patch("translation.server_probe.httpx.get",
                   side_effect=httpx.ConnectError("refused")):
            info = probe_server("http://localhost:8080/v1")
        assert info.reachable is False
        assert "ConnectError" in info.error

    def test_timeout_is_not_reachable(self):
        with patch("translation.server_probe.httpx.get",
                   side_effect=httpx.ReadTimeout("slow")):
            info = probe_server("http://localhost:8080/v1")
        assert info.reachable is False

    def test_probe_never_raises_on_unexpected_error(self):
        with patch("translation.server_probe.httpx.get",
                   side_effect=RuntimeError("boom")):
            info = probe_server("http://localhost:8080/v1")
        assert info.reachable is False
        assert "RuntimeError" in info.error

    def test_503_while_loading_is_not_reachable(self):
        """A model still loading answers 503; translating now would fail."""
        with patch("translation.server_probe.httpx.get",
                   return_value=_FakeResponse(status_code=503)):
            info = probe_server("http://localhost:8080/v1")
        assert info.reachable is False
        assert "503" in info.error

    def test_reads_total_slots(self):
        payload = {"total_slots": 3}
        with patch("translation.server_probe.httpx.get",
                   return_value=_FakeResponse(payload=payload)):
            info = probe_server("http://localhost:8080/v1")
        assert info.reachable is True
        assert info.slots == 3

    def test_reads_context_size_and_model_path(self):
        payload = {
            "total_slots": 1,
            "model_path": "/models/gemma.gguf",
            "default_generation_settings": {"n_ctx": 16384},
        }
        with patch("translation.server_probe.httpx.get",
                   return_value=_FakeResponse(payload=payload)):
            info = probe_server("http://localhost:8080/v1")
        assert info.context_size == 16384
        assert info.model_path == "/models/gemma.gguf"

    def test_missing_total_slots_leaves_slots_unknown(self):
        """Older builds omit total_slots; unknown must not be read as 1."""
        payload = {"default_generation_settings": {"n_ctx": 16384}}
        with patch("translation.server_probe.httpx.get",
                   return_value=_FakeResponse(payload=payload)):
            info = probe_server("http://localhost:8080/v1")
        assert info.reachable is True
        assert info.slots is None

    def test_nonsense_total_slots_is_unknown(self):
        with patch("translation.server_probe.httpx.get",
                   return_value=_FakeResponse(payload={"total_slots": 0})):
            info = probe_server("http://localhost:8080/v1")
        assert info.slots is None

    def test_non_integer_total_slots_is_unknown(self):
        with patch("translation.server_probe.httpx.get",
                   return_value=_FakeResponse(payload={"total_slots": "three"})):
            info = probe_server("http://localhost:8080/v1")
        assert info.slots is None

    def test_non_json_body_is_rejected(self):
        """A different service on the port must not pass preflight."""
        with patch("translation.server_probe.httpx.get",
                   return_value=_FakeResponse(raise_json=True)):
            info = probe_server("http://localhost:8080/v1")
        assert info.reachable is False
        assert "non-JSON" in info.error

    @pytest.mark.parametrize("payload", [[1, 2], {}, {"service": "other"}])
    def test_non_llama_json_body_is_rejected(self, payload):
        with patch("translation.server_probe.httpx.get",
                   return_value=_FakeResponse(payload=payload)):
            info = probe_server("http://localhost:8080/v1")
        assert info.reachable is False
        assert "Unexpected /props" in info.error

    def test_requests_the_props_endpoint(self):
        with patch("translation.server_probe.httpx.get",
                   return_value=_FakeResponse(payload={})) as mock_get:
            probe_server("http://localhost:8080/v1")
        assert mock_get.call_args[0][0].endswith("/props")


class TestLoopbackPinning:
    """`localhost` resolves to both ::1 and 127.0.0.1.

    httpx walks them in turn, so a probe against a down server pays the connect
    timeout once per family — measured at ~4.4s on Windows with the default
    timeout. llama-server binds IPv4 only (`--host 0.0.0.0`), so the IPv6 attempt
    can never succeed and is pure delay.
    """

    def test_localhost_is_probed_over_ipv4(self):
        with patch("translation.server_probe.httpx.get",
                   return_value=_FakeResponse(payload={})) as mock_get:
            probe_server("http://localhost:8080/v1")
        assert mock_get.call_args[0][0] == "http://127.0.0.1:8080/props"

    def test_other_hosts_are_left_alone(self):
        """Only loopback is rewritten; a real host must be probed as given."""
        with patch("translation.server_probe.httpx.get",
                   return_value=_FakeResponse(payload={})) as mock_get:
            probe_server("http://gpu-box.lan:8080/v1")
        assert mock_get.call_args[0][0] == "http://gpu-box.lan:8080/props"

    def test_error_message_names_the_configured_host(self):
        """The rewrite is a transport detail — errors should read as configured."""
        with patch("translation.server_probe.httpx.get",
                   side_effect=OSError("refused")):
            info = probe_server("http://localhost:8080/v1")
        assert "localhost" in unreachable_message("http://localhost:8080/v1", info)

    def test_connect_budget_is_shorter_than_the_read_budget(self):
        """A refused connect must not cost the full read timeout.

        The connect phase is bounded separately because httpx applies a scalar
        timeout per attempt, so the scalar alone cannot bound the whole call.
        """
        with patch("translation.server_probe.httpx.get",
                   return_value=_FakeResponse(payload={})) as mock_get:
            probe_server("http://localhost:8080/v1")
        timeout = mock_get.call_args[1]["timeout"]
        assert timeout.connect <= timeout.read


class TestClampWorkers:

    def test_clamps_down_to_slot_count(self):
        workers, note = clamp_workers(3, ServerInfo(reachable=True, slots=1))
        assert workers == 1
        assert note is not None

    def test_note_names_both_numbers(self):
        _, note = clamp_workers(3, ServerInfo(reachable=True, slots=1))
        assert "3" in note and "1" in note

    def test_no_note_when_within_slot_count(self):
        workers, note = clamp_workers(2, ServerInfo(reachable=True, slots=3))
        assert workers == 2
        assert note is None

    def test_exact_match_is_not_clamped(self):
        workers, note = clamp_workers(3, ServerInfo(reachable=True, slots=3))
        assert workers == 3
        assert note is None

    def test_unknown_slots_leaves_request_untouched(self):
        """An unrecognised build must not silently serialize the client."""
        workers, note = clamp_workers(3, ServerInfo(reachable=True, slots=None))
        assert workers == 3
        assert note is None

    @pytest.mark.parametrize("requested", [0, -5])
    def test_non_positive_request_floors_to_one(self, requested):
        workers, _ = clamp_workers(requested, ServerInfo(reachable=True, slots=4))
        assert workers == 1

    def test_garbage_request_floors_to_one(self):
        workers, _ = clamp_workers("many", ServerInfo(reachable=True, slots=4))
        assert workers == 1

    def test_float_request_is_truncated(self):
        """Tk Scale hands back a float even through an IntVar."""
        workers, _ = clamp_workers(2.9, ServerInfo(reachable=True, slots=8))
        assert workers == 2


class TestUnreachableMessage:

    def test_includes_probe_url_and_error(self):
        info = ServerInfo(reachable=False, error="ConnectError: refused")
        message = unreachable_message("http://localhost:8080/v1", info)
        assert "http://localhost:8080/props" in message
        assert "ConnectError: refused" in message

    def test_handles_missing_error_detail(self):
        message = unreachable_message("http://localhost:8080/v1",
                                      ServerInfo(reachable=False))
        assert "no response" in message
