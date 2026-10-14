"""Checkpoint and document identity contracts for manga processing."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from manga.images import file_hash
from manga.store import Store, correct_page, documents, page_sources, write_json
from translation.transport import TransportCancelledError
from utils.atomic_io import CommitGate


def _image(path, color="white"):
    Image.new("RGB", (12, 12), color).save(path)


def _document(tmp_path, count=1):
    source_dir = tmp_path / "素材"
    source_dir.mkdir(exist_ok=True)
    paths = []
    for index in range(count):
        path = source_dir / f"page{index + 1}.png"
        _image(path, "red" if index % 2 else "white")
        paths.append(str(path))
    sources = page_sources(paths)
    document_id = documents(paths, independent=False)[0][0]
    gate = CommitGate()
    store = Store(tmp_path / "output", document_id, sources, gate)
    pages = [store.load_page(source, "ocr-model-v1") for source in sources]
    return store, pages, sources, document_id, gate


def test_page_ids_retain_duplicate_images_and_same_names_across_directories(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    _image(first / "page.png")
    _image(second / "page.png")
    same = str(first / "page.png")
    duplicate = page_sources([same, same])
    assert duplicate[0]["id"] != duplicate[1]["id"]
    distinct = page_sources([str(first / "page.png"), str(second / "page.png")])
    assert distinct[0]["id"] != distinct[1]["id"]
    assert [len(group) for _, group in documents([same, same], independent=False)] == [2]
    assert [len(group) for _, group in documents([same, same], independent=True)] == [1, 1]


def test_load_page_invalidates_on_source_hash_or_ocr_identity_but_translation_settings_are_separate(tmp_path):
    store, pages, sources, _, _ = _document(tmp_path)
    page = pages[0]
    store.set_ocr(page, "原文", "ok")
    assert store.load_page(sources[0], "ocr-model-v1")["status"] == "ok"
    assert store.load_page(sources[0], "ocr-model-v2")["status"] == "pending"
    _image(Path(sources[0]["path"]), "blue")
    changed = store.load_page(sources[0], "ocr-model-v1")
    assert changed["status"] == "pending"
    assert changed["source_hash"] == file_hash(sources[0]["path"])
    key_a = store.translation_key({"model": "translate-a", "locale": "zh-Hant"})
    key_b = store.translation_key({"model": "translate-b", "locale": "zh-Hant"})
    assert key_a != key_b


def test_correction_preserves_raw_increments_revision_order_and_duplicate_ids(tmp_path):
    store, pages, _, document_id, _ = _document(tmp_path)
    store.set_ocr(pages[0], "同一句\n\n同一句", "ok")
    store.save_manifest("pending")
    page_id = pages[0]["id"]
    corrected = correct_page(store.root, page_id, ["同一句", "修正版", "同一句"])
    assert corrected["raw"] == "同一句\n\n同一句"
    assert corrected["revision"] == 1
    assert [segment["text"] for segment in corrected["segments"]] == ["同一句", "修正版", "同一句"]
    assert len({segment["id"] for segment in corrected["segments"]}) == 3
    assert corrected["segments"][0]["id"] != corrected["segments"][2]["id"]
    assert json.loads((store.root / "manifest.json").read_text(encoding="utf-8"))["status"] == "needs_translation"


def test_partial_exports_are_explicit_and_formal_names_only_appear_when_complete(tmp_path):
    store, pages, _, _, _ = _document(tmp_path, count=2)
    for page in pages:
        store.set_ocr(page, "原文", "ok")
    segments = store.segments()
    values = {"segments": {segments[0]["id"]: "譯文"}}
    store.export("zh-Hant", values, completed=False)
    assert (store.root / "translations.zh-Hant.partial.json").exists()
    assert "PARTIAL" in (store.root / "translated.zh-Hant.partial.txt").read_text(encoding="utf-8")
    assert "[MISSING TRANSLATION]" in (store.root / "bilingual.zh-Hant.partial.md").read_text(encoding="utf-8")
    assert not (store.root / "translations.zh-Hant.json").exists()
    store.export("zh-Hant", {"segments": {s["id"]: "譯文" for s in segments}}, completed=True)
    assert (store.root / "translations.zh-Hant.json").exists()
    assert "completed" in (store.root / "translations.zh-Hant.json").read_text(encoding="utf-8")


def test_corrupt_checkpoint_and_invalid_schema_are_refused(tmp_path):
    store, pages, _, _, _ = _document(tmp_path)
    store.set_ocr(pages[0], "原文", "ok")
    path = store.root / "pages" / f"{pages[0]['id']}.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupted checkpoint"):
        store.load_page(store.sources[0], "ocr-model-v1")
    invalid = dict(pages[0])
    invalid["status"] = "not-a-page-status"
    write_json(path, invalid)
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        store.load_page(store.sources[0], "ocr-model-v1")


def test_cancelled_gate_suppresses_checkpoint_save(tmp_path):
    store, pages, _, _, gate = _document(tmp_path)
    gate.cancel()
    with pytest.raises(TransportCancelledError):
        store.save_page(pages[0])
    assert not (store.root / "pages" / f"{pages[0]['id']}.json").exists()


def test_atomic_replace_failure_keeps_previous_checkpoint(tmp_path):
    store, pages, _, _, _ = _document(tmp_path)
    store.set_ocr(pages[0], "舊文字", "ok")
    path = store.root / "pages" / f"{pages[0]['id']}.json"
    before = path.read_bytes()
    pages[0]["raw"] = "新文字"
    with patch("utils.atomic_io.os.replace", side_effect=OSError("replace failed")):
        with pytest.raises(OSError, match="replace failed"):
            store.save_page(pages[0])
    assert path.read_bytes() == before
