"""Mirror Colab training progress onto the One Touch external drive.

Sources:
  hf    (default) Pull continuous/<tier> checkpoints from Darknecrocities/Helios and
        curriculum_ledger.json from Darknecrocities/Helios-Pretrain-Corpus. The Colab
        runner pushes both after every tier, so this works without a desktop Drive app.
  drive Copy from a locally synced Google Drive folder.

Writes only to /Volumes/One Touch/Helios (HF cache is redirected there too), and
triggers the Ollama refresh when new weights land.

Author: Arron Kian Parejas
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("helios.sync_onetouch")

ONE_TOUCH_ROOT = Path("/Volumes/One Touch/Helios")
MODEL_TIERS = ["nano", "alpha", "beta", "core", "pro", "flagship"]
CKPT_FILES = ["model.safetensors", "config.json", "manifest.json", "SHA256SUMS"]
MODEL_REPO = "Darknecrocities/Helios"
DATA_REPO = "Darknecrocities/Helios-Pretrain-Corpus"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _copy_if_changed(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size and _sha256(dst) == _sha256(src):
        return False
    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)  # atomic swap: a half-copied checkpoint never becomes "latest"
    logger.info("Mirrored %s -> %s", src.name, dst)
    return True


def sync_from_hf(onetouch: Path) -> int:
    os.environ.setdefault("HF_HOME", str(onetouch / "data" / "hf_cache"))
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    changed = 0
    for tier in MODEL_TIERS:
        for fname in CKPT_FILES:
            try:
                src = Path(hf_hub_download(MODEL_REPO, f"continuous/{tier}/{fname}", token=token))
            except Exception as e:
                logger.warning("Skip %s/%s: %s", tier, fname, e)
                continue
            dst = onetouch / "checkpoints" / f"{tier}_continuous" / "latest" / fname
            changed += _copy_if_changed(src, dst)
    try:
        src = Path(hf_hub_download(DATA_REPO, "curriculum_ledger.json", repo_type="dataset", token=token))
        changed += _copy_if_changed(src, onetouch / "manifests" / "curriculum_ledger.json")
    except Exception as e:
        logger.warning("Ledger sync failed: %s", e)
    return changed


def sync_from_drive(drive_root: Path, onetouch: Path) -> int:
    changed = 0
    for tier in MODEL_TIERS:
        for fname in CKPT_FILES:
            rel = Path("checkpoints") / f"{tier}_continuous" / "latest" / fname
            changed += _copy_if_changed(drive_root / rel, onetouch / rel)
    rel = Path("manifests") / "curriculum_ledger.json"
    changed += _copy_if_changed(drive_root / rel, onetouch / rel)
    return changed


def refresh_ollama() -> None:
    try:
        subprocess.run([sys.executable, "scripts/sync_ollama_models.py"], check=True, timeout=3600)
    except Exception as e:
        logger.warning("Ollama refresh failed: %s", e)


def run_once(args: argparse.Namespace) -> None:
    if not args.onetouch_root.exists():
        logger.error("One Touch drive not mounted at %s", args.onetouch_root)
        return
    n = sync_from_hf(args.onetouch_root) if args.source == "hf" else sync_from_drive(args.drive_root, args.onetouch_root)
    logger.info("Sync complete: %d file(s) updated on One Touch.", n)
    if n and not args.skip_ollama:
        refresh_ollama()


def main() -> None:
    p = argparse.ArgumentParser(description="Mirror Colab checkpoints to One Touch")
    p.add_argument("--source", choices=["hf", "drive"], default="hf")
    p.add_argument("--drive-root", type=Path, default=Path.home() / "Google Drive" / "MyDrive" / "Helios")
    p.add_argument("--onetouch-root", type=Path, default=ONE_TOUCH_ROOT)
    p.add_argument("--loop", action="store_true", help="Run continuously")
    p.add_argument("--interval", type=int, default=600, help="Seconds between syncs")
    p.add_argument("--skip-ollama", action="store_true")
    args = p.parse_args()
    if not args.loop:
        run_once(args)
        return
    while True:
        run_once(args)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
