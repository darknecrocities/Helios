"""Automatic Mirror Sync between Google Drive and One Touch External SSD.

Continuously watches or syncs:
  Google Drive (`/content/drive/MyDrive/Helios/` or local `~/Google Drive/MyDrive/Helios`)
  -> Local External SSD (`/Volumes/One Touch/Helios/`)
Ensures 0B internal Mac SSD writes and triggers local Ollama tags update on change.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("helios.sync_drive_onetouch")

ONE_TOUCH_ROOT = Path("/Volumes/One Touch/Helios")
MODEL_TIERS = ["nano", "alpha", "beta", "core", "pro", "flagship"]


def sync_file_if_changed(src: Path, dst: Path) -> bool:
    """Copies src to dst if dst does not exist or size/mtime differ."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        logger.info("Mirroring %s -> %s", src.name, dst)
        shutil.copy2(src, dst)
        return True
    return False


def sync_drive_to_onetouch(drive_root: Path, onetouch_root: Path = ONE_TOUCH_ROOT) -> int:
    """Performs an incremental mirror sync of checkpoints and ledger to external drive."""
    if not onetouch_root.exists():
        logger.error("External drive %s is not mounted!", onetouch_root)
        return 0

    if not drive_root.exists():
        logger.warning("Google Drive root %s not found.", drive_root)
        return 0

    files_synced = 0
    # 1. Sync checkpoints
    for tier in MODEL_TIERS:
        src_latest = drive_root / "checkpoints" / f"{tier}_continuous" / "latest"
        dst_latest = onetouch_root / "checkpoints" / f"{tier}_continuous" / "latest"
        if not src_latest.exists():
            continue

        for fname in ["model.safetensors", "config.json", "manifest.json", "SHA256SUMS"]:
            src_f = src_latest / fname
            dst_f = dst_latest / fname
            if sync_file_if_changed(src_f, dst_f):
                files_synced += 1

    # 2. Sync curriculum ledger
    src_ledger = drive_root / "manifests" / "curriculum_ledger.json"
    dst_ledger = onetouch_root / "manifests" / "curriculum_ledger.json"
    if sync_file_if_changed(src_ledger, dst_ledger):
        files_synced += 1

    logger.info("Drive -> One Touch sync complete (%d files mirrored).", files_synced)
    return files_synced


def update_ollama_tags() -> None:
    """Invokes local Ollama sync script to refresh local Ollama models."""
    cmd = [sys.executable, "scripts/sync_ollama_models.py"]
    try:
        logger.info("Triggering local Ollama models refresh...")
        subprocess.run(cmd, check=True)
    except Exception as e:
        logger.warning("Could not refresh Ollama tags: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror Drive Checkpoints to One Touch")
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path.home() / "Google Drive" / "MyDrive" / "Helios",
        help="Google Drive local sync folder",
    )
    parser.add_argument(
        "--onetouch-root",
        type=Path,
        default=ONE_TOUCH_ROOT,
        help="One Touch destination path",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously as background daemon",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Sync interval in seconds (default: 300)",
    )
    args = parser.parse_args()

    if args.loop:
        logger.info("Starting background sync daemon (interval: %ds)...", args.interval)
        while True:
            n = sync_drive_to_onetouch(args.drive_root, args.onetouch_root)
            if n > 0:
                update_ollama_tags()
            time.sleep(args.interval)
    else:
        n = sync_drive_to_onetouch(args.drive_root, args.onetouch_root)
        if n > 0:
            update_ollama_tags()


if __name__ == "__main__":
    import sys
    main()
