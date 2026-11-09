"""Unit tests for storage safety, filesystem boundaries, quota, and atomic ops."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from helios_pipeline.storage.atomic_ops import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
)
from helios_pipeline.storage.manager import (
    DriveDisconnectedError,
    InternalStorageViolationError,
    OneTouchStorageManager,
    StorageConfig,
    StorageQuotaExceededError,
)


def test_atomic_write_operations() -> None:
    """Verify atomic writes write, flush, fsync, and replace correctly."""
    with TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / "sub" / "test.json"
        data = {"status": "ok", "step": 42}
        res = atomic_write_json(target, data)
        assert res.exists()
        assert target.read_text(encoding="utf-8").strip() == '{\n  "status": "ok",\n  "step": 42\n}'

        # Test text
        text_target = Path(tmp_dir) / "test.txt"
        atomic_write_text(text_target, "helios_atomic_content")
        assert text_target.read_text(encoding="utf-8") == "helios_atomic_content"

        # Test bytes
        bin_target = Path(tmp_dir) / "test.bin"
        atomic_write_bytes(bin_target, b"\x00\x01\x02\x03")
        assert bin_target.read_bytes() == b"\x00\x01\x02\x03"


def test_internal_storage_violation_guard() -> None:
    """Verify that writing to internal Mac root filesystem raises InternalStorageViolationError."""
    cfg = StorageConfig(
        external_drive_path="/",
        candidate_paths=["/"],
        require_external_drive=False,
        prohibit_internal_storage_fallback=True,
    )
    with pytest.raises(InternalStorageViolationError) as exc_info:
        OneTouchStorageManager(config=cfg)
    assert "resolves to internal storage device" in str(exc_info.value)


def test_drive_disconnected_error() -> None:
    """Verify DriveDisconnectedError is raised when candidate paths do not exist."""
    cfg = StorageConfig(
        external_drive_path="/Volumes/NonExistentHeliosDrive12345",
        candidate_paths=["/Volumes/NonExistentHeliosDrive12345"],
        require_external_drive=True,
        prohibit_internal_storage_fallback=True,
    )
    with pytest.raises(DriveDisconnectedError):
        OneTouchStorageManager(config=cfg)


def test_storage_quota_and_reserve_guards() -> None:
    """Verify minimum free space and quota bounds enforcement."""
    with TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        cfg = StorageConfig(
            external_drive_path=str(base),
            candidate_paths=[str(base)],
            require_external_drive=False,
            prohibit_internal_storage_fallback=False,
            minimum_free_space_gb=1000000.0,  # Impossible reserve requirement
            quota_limit_gb=100.0,
        )
        storage = OneTouchStorageManager(config=cfg)
        with pytest.raises(StorageQuotaExceededError) as exc_info:
            storage.check_health()
        assert "below the required 10% reserve floor" in str(exc_info.value)
