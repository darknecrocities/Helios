"""Atomic file operations for ExFAT and POSIX filesystems without hardlink dependencies."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def atomic_write_bytes(target_path: Path | str, data: bytes) -> Path:
    """Atomically write binary data via in-volume temporary file and fsync."""
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = target.parent / f".tmp_{uuid.uuid4().hex}_{target.name}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
        return target
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def atomic_write_text(target_path: Path | str, content: str, encoding: str = "utf-8") -> Path:
    """Atomically write text via in-volume temporary file and fsync."""
    return atomic_write_bytes(target_path, content.encode(encoding))


def atomic_write_json(
    target_path: Path | str,
    data: Any,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """Atomically write JSON data."""
    serialized = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii) + "\n"
    return atomic_write_text(target_path, serialized)
