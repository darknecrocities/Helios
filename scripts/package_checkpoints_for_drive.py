"""Package latest Helios checkpoints and ledger for Google Drive transfer.

Prepares a lightweight bundle containing:
- latest verified model weights (`model.safetensors`, `config.json`, `manifest.json`, `SHA256SUMS`)
- `curriculum_ledger.json` (cursor, epochs, deduplication hashes)
- `helios_massive_pretrain_corpus.jsonl` (omni-domain dataset corpus)

Target: Google Drive (`/content/drive/MyDrive/Helios/`) for zero-loss Google Colab resumption.
Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("helios.package_drive")

ONE_TOUCH_ROOT = Path("/Volumes/One Touch/Helios")
MODEL_TIERS = ["nano", "alpha", "beta", "core", "pro", "flagship"]


def package_latest_checkpoints(
    src_dir: Path,
    dest_dir: Path,
    include_corpus: bool = True,
    compress_tar: bool = False,
) -> Path:
    """Copies or archives the active continuous checkpoints and ledger into Google Drive."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dest = dest_dir / "checkpoints"
    manifests_dest = dest_dir / "manifests"
    datasets_dest = dest_dir / "datasets"

    checkpoints_dest.mkdir(parents=True, exist_ok=True)
    manifests_dest.mkdir(parents=True, exist_ok=True)
    datasets_dest.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning verified checkpoints on source: %s", src_dir)
    package_summary: dict[str, Any] = {"tiers": {}, "timestamp": None}

    for tier in MODEL_TIERS:
        tier_dir = src_dir / "checkpoints" / f"{tier}_continuous"
        latest_dir = tier_dir / "latest"
        if not latest_dir.exists():
            logger.warning("No latest dir found for %s at %s", tier, latest_dir)
            continue

        target_tier_dir = checkpoints_dest / f"{tier}_continuous" / "latest"
        target_tier_dir.mkdir(parents=True, exist_ok=True)

        for fname in ["model.safetensors", "config.json", "manifest.json", "SHA256SUMS"]:
            fpath = latest_dir / fname
            if fpath.exists():
                dst = target_tier_dir / fname
                if not dst.exists() or dst.stat().st_size != fpath.stat().st_size:
                    logger.info("Copying %s -> %s", fpath.name, target_tier_dir)
                    shutil.copy2(fpath, dst)

        manifest_file = latest_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                package_summary["tiers"][tier] = json.load(f)

    # Copy curriculum ledger
    ledger_src = src_dir / "manifests" / "curriculum_ledger.json"
    if ledger_src.exists():
        ledger_dst = manifests_dest / "curriculum_ledger.json"
        logger.info("Copying curriculum ledger: %s -> %s", ledger_src, ledger_dst)
        shutil.copy2(ledger_src, ledger_dst)

    # Copy corpus if requested
    if include_corpus:
        corpus_src = src_dir / "datasets" / "helios_massive_pretrain_corpus.jsonl"
        if corpus_src.exists():
            corpus_dst = datasets_dest / "helios_massive_pretrain_corpus.jsonl"
            if not corpus_dst.exists() or corpus_dst.stat().st_size != corpus_src.stat().st_size:
                logger.info("Copying pretraining corpus (%s) -> %s", corpus_src.name, corpus_dst)
                shutil.copy2(corpus_src, corpus_dst)

    summary_file = dest_dir / "drive_sync_manifest.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(package_summary, f, indent=2)

    logger.info("Package sync complete at %s", dest_dir)
    return dest_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Helios checkpoints for Drive")
    parser.add_argument(
        "--source",
        type=Path,
        default=ONE_TOUCH_ROOT,
        help="Source One Touch directory",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path.home() / "Google Drive" / "MyDrive" / "Helios",
        help="Destination directory (e.g. Google Drive sync path or export folder)",
    )
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="Skip copying the large jsonl corpus file",
    )
    args = parser.parse_args()

    if not args.source.exists():
        logger.error("Source directory %s not mounted!", args.source)
        return

    package_latest_checkpoints(
        src_dir=args.source,
        dest_dir=args.dest,
        include_corpus=not args.skip_corpus,
    )


if __name__ == "__main__":
    main()
