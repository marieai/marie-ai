"""Unit tests for marie.utils.docs.fetch_asset_to_temp and its use by docs_from_asset."""

import os

import numpy as np
import pytest

from marie.utils import docs as docs_module
from marie.utils.docs import docs_from_asset, fetch_asset_to_temp

# Minimal PNG signature — enough for filetype's magic-byte detection.
PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def _patch_storage(
    monkeypatch,
    *,
    payload: bytes,
    can_handle: bool = True,
    connected: bool = True,
    exists: bool = True,
):
    """Route StorageManager through fakes that mimic a buffered download.

    read_to_file writes to the still-open temp handle WITHOUT flushing, exactly
    like the real buffered write — so any type detection performed before the
    handle closes sees an empty on-disk file.
    """

    monkeypatch.setattr(
        docs_module.StorageManager,
        "can_handle",
        staticmethod(lambda *a, **k: can_handle),
    )
    monkeypatch.setattr(
        docs_module.StorageManager,
        "ensure_connection",
        staticmethod(lambda *a, **k: connected),
    )
    monkeypatch.setattr(
        docs_module.StorageManager, "exists", staticmethod(lambda *a, **k: exists)
    )

    def fake_read_to_file(path, dst, overwrite=False, **kwargs):
        dst.write(payload)  # buffered, intentionally not flushed
        return True

    monkeypatch.setattr(
        docs_module.StorageManager, "read_to_file", staticmethod(fake_read_to_file)
    )


def test_fetch_asset_preserves_suffix_and_detects_type(monkeypatch):
    _patch_storage(monkeypatch, payload=b"a,b,c\n1,2,3\n")

    path, file_type = fetch_asset_to_temp("s3://bucket/sample.csv")

    try:
        assert path.startswith("/tmp/marie/")
        assert path.endswith(".csv")
        assert os.path.exists(path)
        assert file_type == "csv"
    finally:
        os.remove(path)


def test_fetch_asset_detects_small_payload_after_close(monkeypatch):
    """Regression: a <8KB payload is detected via magic bytes only if type
    detection happens AFTER the buffered handle is closed and flushed. The
    .dat suffix does not resolve via the extension fallback, so a premature
    (pre-close, empty-file) detection would raise instead of returning png."""
    _patch_storage(monkeypatch, payload=PNG_MAGIC)

    path, file_type = fetch_asset_to_temp("s3://bucket/tiny.dat")

    try:
        assert os.path.getsize(path) == len(PNG_MAGIC)
        assert file_type == "png"
    finally:
        os.remove(path)


def test_fetch_asset_missing_remote_raises(monkeypatch):
    _patch_storage(monkeypatch, payload=PNG_MAGIC, exists=False)

    with pytest.raises(ValueError, match="Remote file does not exist"):
        fetch_asset_to_temp("s3://bucket/gone.png")


def test_fetch_asset_unhandled_uri_raises(monkeypatch):
    _patch_storage(monkeypatch, payload=PNG_MAGIC, can_handle=False)

    with pytest.raises(Exception, match="no suitable storage manager"):
        fetch_asset_to_temp("weird://bucket/file.png")


def test_fetch_asset_no_connection_raises(monkeypatch):
    _patch_storage(monkeypatch, payload=PNG_MAGIC, connected=False)

    with pytest.raises(ValueError, match="Could not connect to S3"):
        fetch_asset_to_temp("s3://bucket/file.png")


def test_docs_from_asset_routes_through_helper(monkeypatch, tmp_path):
    """docs_from_asset must download via fetch_asset_to_temp (no duplicate
    download code) and then parse the returned local file."""
    local = tmp_path / "downloaded.png"
    local.write_bytes(PNG_MAGIC)

    calls = []

    def spy_fetch(asset_key):
        calls.append(asset_key)
        return str(local), "png"

    monkeypatch.setattr(docs_module, "fetch_asset_to_temp", spy_fetch)
    monkeypatch.setattr(
        docs_module,
        "load_document",
        lambda path, *a, **k: {
            "mode": "frames",
            "frames": [np.zeros((2, 2, 3), dtype=np.uint8)],
        },
    )

    docs = docs_from_asset("s3://bucket/downloaded.png")

    assert calls == ["s3://bucket/downloaded.png"]
    assert len(docs) == 1

    docs2, returned_path = docs_from_asset(
        "s3://bucket/downloaded.png", return_file_path=True
    )
    assert returned_path == str(local)
    assert len(docs2) == 1
