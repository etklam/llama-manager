"""Focused contract tests for the manga Vulkan building blocks."""

import base64
import io
import json
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from manga.images import collect, image_payload
from manga.runtime import Profile, command, evidence
from manga.translation import ContentError, DocumentTranslator, parse_translation
from translation.llm_client import CompletionResult
from translation.transport import TransportCancelledError
from server_controller import ServerBusyError, ServerController


def _png(path, size=(20, 20), mode="RGBA"):
    image = Image.new(mode, size, (255, 0, 0, 128) if "A" in mode else "white")
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image = image.convert("RGB")
    image.save(path)


def _profile(tmp_path, *, context=4096, max_tokens=512, mmproj=""):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    projector = ""
    if mmproj:
        p = tmp_path / "mmproj.gguf"
        p.write_bytes(b"projector")
        projector = str(p)
    return Profile(str(model), projector, context, max_tokens)


def test_collect_natural_unicode_nonrecursive_and_output_exclusion(tmp_path):
    source = tmp_path / "漫畫"
    nested = source / "nested"
    output = source / "output"
    nested.mkdir(parents=True)
    output.mkdir()
    names = ["page10.PNG", "page2.jpg", "頁1.webp", "same.png"]
    for name in names:
        _png(source / name)
    _png(source / "same.png")
    _png(nested / "page1.png")
    _png(output / "page3.png")
    found = collect([source], output=output)
    assert [Path(p).name for p in found] == ["page2.jpg", "page10.PNG", "same.png", "頁1.webp"]
    # Two equal basenames in separate roots remain distinct source entries.
    other = tmp_path / "other"
    other.mkdir()
    _png(other / "same.png")
    both = collect([source / "same.png", other / "same.png"])
    assert len(both) == 2 and len(set(both)) == 2


def test_image_payload_handles_exif_alpha_and_reports_bad_or_oversize(tmp_path):
    image = Image.new("RGB", (2, 3), "white")
    exif = image.getexif()
    exif[274] = 6  # rotate 90 degrees; payload must apply orientation
    oriented = tmp_path / "oriented.jpg"
    image.save(oriented, exif=exif)
    alpha = tmp_path / "alpha.png"
    _png(alpha)
    for path in (oriented, alpha):
        payload = image_payload(path)
        assert payload.startswith("data:image/png;base64,")
        decoded = base64.b64decode(payload.split(",", 1)[1])
        with Image.open(io.BytesIO(decoded)) as result:
            assert result.mode == "RGB"
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(Exception):
        image_payload(bad)
    huge = tmp_path / "huge.png"
    _png(huge, size=(16001, 1))
    with pytest.raises(ValueError, match="too large/long"):
        image_payload(huge)


def test_runtime_command_requires_enumerated_device_and_advertised_flags(tmp_path):
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"runtime")
    profile = _profile(tmp_path, mmproj="yes")
    help_text = " ".join([
        "-m --host --port --device -ngl -c -b --parallel --flash-attn",
        "--cache-type-k --cache-type-v --jinja --mmproj --mmproj-offload --mmproj-device --fit",
    ])
    discovery = {"devices": ["Vulkan0"], "help": help_text}
    args = command(str(exe), discovery, "Vulkan0", profile, 8080)
    assert args[args.index("--mmproj") + 1] == profile.mmproj
    assert args[args.index("--mmproj-device") + 1] == "Vulkan0"
    with pytest.raises(ValueError, match="not enumerated"):
        command(str(exe), discovery, "Vulkan1", profile, 8080)
    with pytest.raises(ValueError, match="advertise"):
        command(str(exe), {"devices": ["Vulkan0"], "help": help_text.replace("--jinja", "")}, "Vulkan0", profile, 8080)


def test_runtime_evidence_does_not_promote_health_or_cpu_buffers_to_vulkan():
    health = evidence(["HTTP /health 200", "model loaded"], "Vulkan0", vision=True, request_ok=True)
    assert health["status"] == "unverified"
    assert health["decoder_backend"] == "unknown"
    cpu = evidence(["offloaded 0/40 layers to GPU", "CPU model buffer"], "Vulkan0", request_ok=True)
    assert cpu["status"] == "unverified"
    assert cpu["decoder_backend"] == "CPU-only"
    verified = evidence(["offloaded 40/40 layers to GPU", "Vulkan0 compute buffer", "CLIP using Vulkan0 backend", "reserve_compute_meta: Vulkan0 compute buffer size = 12.00 MiB"], "Vulkan0", vision=True, request_ok=True)
    assert verified["status"] == "verified"


@pytest.mark.parametrize("text", [
    json.dumps([{"id": "a", "translation": "甲"}]),
    json.dumps([{"id": "a", "translation": "甲"}, {"id": "a", "translation": "乙"}]),
    json.dumps([{"id": "z", "translation": "甲"}]),
    json.dumps([{"id": "a", "translation": 1}]),
])
def test_translation_rejects_missing_duplicate_unknown_and_wrong_id_rows(text):
    expected = {"a", "b"}
    with pytest.raises(ContentError):
        parse_translation(text, expected)


def test_translation_preserves_repeated_source_text_by_id():
    result = parse_translation(json.dumps([
        {"id": "p1", "translation": "同一句"},
        {"id": "p2", "translation": "同一句"},
    ]), {"p1", "p2"})
    assert result == {"p1": "同一句", "p2": "同一句"}


def test_translation_rejects_truncation_and_cancellation():
    with pytest.raises(ContentError):
        from manga.translation import usable
        usable(CompletionResult('{"id":"a"}', "length"))
    cancelled = lambda: True
    client = Mock()
    translator = DocumentTranslator(client, _profile_for_translation(), "ja", "zh", False, cancelled)
    with pytest.raises(TransportCancelledError):
        translator.translate_batch([{"id": "a", "text": "文字"}], [])
    client.complete_analysis.assert_not_called()


def _profile_for_translation():
    return ProfileLike("translation", context=8192, max_tokens=512)


class ProfileLike:
    def __init__(self, model, context, max_tokens):
        self.model, self.context, self.max_tokens = model, context, max_tokens
        self.temperature = 0.0


def test_long_text_batches_reconstruct_all_characters_with_stable_parent_ids():
    translator = DocumentTranslator(Mock(), ProfileLike("translation", 4096, 512), "ja", "zh", False, lambda: False)
    source = "あいうえお" * 1000
    batches = translator.batches([{"id": "page-1", "text": source}])
    parts = [part for batch in batches for part in batch]
    assert "".join(part["text"] for part in parts) == source
    assert all(part["id"].startswith("page-1~") and part["parent"] == "page-1" for part in parts)


def test_server_lease_is_busy_across_threads():
    controller = ServerController(Path("llama-server.exe"), lambda _: None)
    entered = threading.Event()
    release = threading.Event()

    def holder():
        with controller.lease("manga"):
            entered.set()
            release.wait(2)

    thread = threading.Thread(target=holder)
    thread.start()
    assert entered.wait(1)
    with pytest.raises(ServerBusyError):
        with controller.lease("subtitle"):
            pass
    release.set()
    thread.join(2)


def test_old_log_monitor_cannot_clear_replacement_process():
    controller = ServerController(Path("llama-server.exe"), lambda _: None)
    old = Mock()
    old.stdout.readline.return_value = ""
    old.poll.return_value = 1
    replacement = Mock()
    replacement.poll.return_value = None
    controller._process = replacement
    controller._running = True
    controller._port = 9000
    controller._monitor_logs(old)
    assert controller._process is replacement
    assert controller.running is True
    assert controller.port == 9000
