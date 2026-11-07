"""External disk space preservation and backup mirroring engine for Helios.

Safely mirrors heavy checkpoints, distribution wheels, model cards, and tokenizer
artifacts to external storage (default: `/Volumes/One Touch/Helios` with automatic
fallback to `~/Helios_backup`) with end-to-end streaming SHA-256 integrity verification.

Preserves local laptop disk space by safely freeing local checkpoints after verified
mirror confirmation, while creating lightweight pointer stubs so local tools remain
aware of checkpoint history.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("helios.export.disk")

PRIMARY_DISK_PATH: Path = Path("/Volumes/One Touch/Helios")
FALLBACK_DISK_PATH: Path = Path.home() / "Helios_backup"
DEFAULT_DISK_PATH: Path = (
    PRIMARY_DISK_PATH if PRIMARY_DISK_PATH.parent.exists() else FALLBACK_DISK_PATH
)
DEFAULT_ONEDRIVE_PATH: Path = DEFAULT_DISK_PATH


class DiskSyncIntegrityError(ValueError):
    """Raised when mirrored file SHA-256 digest fails to match source digest."""


OneDriveSyncIntegrityError = DiskSyncIntegrityError


@dataclass
class MirroredItem:
    """Catalog entry for a mirrored file."""

    relative_path: str
    source_path: str
    destination_path: str
    size_bytes: int
    sha256: str
    timestamp: float = field(default_factory=time.time)
    local_freed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary."""
        return asdict(self)


def compute_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Compute streaming SHA-256 hexadecimal digest of a binary file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class DiskMirrorManager:
    """Orchestrates verified mirroring to external disk and local disk reclamation."""

    def __init__(
        self,
        disk_root: Path | str | None = None,
        workspace_root: Path | str | None = None,
        onedrive_root: Path | str | None = None,
    ) -> None:
        raw_target = (
            disk_root
            or onedrive_root
            or os.environ.get("HELIOS_DISK_PATH")
            or os.environ.get("HELIOS_ONEDRIVE_PATH")
            or DEFAULT_DISK_PATH
        )
        self.disk_root = Path(raw_target).resolve()
        self.onedrive_root = self.disk_root  # Backward compatibility alias
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.manifest_path = self.disk_root / "disk_manifest.json"

    def ensure_disk_accessible(self) -> None:
        """Verify that the target external disk folder is accessible and writable."""
        try:
            self.disk_root.mkdir(parents=True, exist_ok=True)
            test_probe = self.disk_root / f".probe_{time.time_ns()}.tmp"
            test_probe.write_text("probe", encoding="utf-8")
            test_probe.unlink()
        except Exception as e:
            raise PermissionError(
                f"Cannot write to disk destination at {self.disk_root}: {e}"
            ) from e

    ensure_onedrive_accessible = ensure_disk_accessible

    def mirror_file(
        self,
        src: Path | str,
        dst: Path | str,
        verify_sha: bool = True,
    ) -> MirroredItem:
        """Atomically copy a file to external disk with SHA-256 integrity verification."""
        source = Path(src).resolve()
        destination = Path(dst).resolve()

        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        source_sha = compute_sha256(source)
        source_bytes = source.stat().st_size

        staging_dst = destination.parent / f".tmp_{destination.name}_{time.time_ns()}"
        shutil.copy2(source, staging_dst)

        with open(staging_dst, "rb") as f:
            os.fsync(f.fileno())

        if verify_sha:
            staged_sha = compute_sha256(staging_dst)
            if staged_sha != source_sha:
                staging_dst.unlink(missing_ok=True)
                raise DiskSyncIntegrityError(
                    f"Integrity check failed copying {source} -> {destination}. "
                    f"Expected SHA-256 {source_sha}, got {staged_sha}"
                )

        os.replace(staging_dst, destination)
        logger.info(
            "Successfully mirrored %s -> %s (SHA: %s)", source.name, destination, source_sha[:10]
        )

        rel_path = (
            str(source.relative_to(self.workspace_root))
            if source.is_relative_to(self.workspace_root)
            else source.name
        )

        return MirroredItem(
            relative_path=rel_path,
            source_path=str(source),
            destination_path=str(destination),
            size_bytes=source_bytes,
            sha256=source_sha,
            timestamp=time.time(),
            local_freed=False,
        )

    def mirror_checkpoints(
        self,
        checkpoints_dir: Path | str = "checkpoints",
        free_local: bool = False,
    ) -> list[MirroredItem]:
        """Mirror all checkpoint step folders to external disk, optionally freeing local disk."""
        self.ensure_disk_accessible()
        ckpt_dir = (
            (self.workspace_root / checkpoints_dir).resolve()
            if not Path(checkpoints_dir).is_absolute()
            else Path(checkpoints_dir).resolve()
        )

        if not ckpt_dir.exists():
            logger.warning("Checkpoints directory %s does not exist; skipping.", ckpt_dir)
            return []

        items: list[MirroredItem] = []
        target_base = self.disk_root / "checkpoints"

        for p in ckpt_dir.iterdir():
            if p.is_dir() and (
                p.name.startswith("checkpoint_step_")
                or p.name.startswith("interrupted_checkpoint_step_")
            ):
                dst_step_dir = target_base / p.name
                dst_step_dir.mkdir(parents=True, exist_ok=True)

                # Mirror manifest.json
                local_manifest = p / "manifest.json"
                if local_manifest.exists():
                    item = self.mirror_file(local_manifest, dst_step_dir / "manifest.json")
                    items.append(item)

                # Mirror state.pt
                local_state = p / "state.pt"
                if local_state.exists():
                    item = self.mirror_file(local_state, dst_step_dir / "state.pt")

                    # If requested, free local disk space safely
                    if free_local:
                        pointer_file = p / "state.pt.disk_pointer.json"
                        pointer_data = {
                            "mirrored_to": str(dst_step_dir / "state.pt"),
                            "sha256": item.sha256,
                            "size_bytes": item.size_bytes,
                            "timestamp": time.time(),
                            "local_freed": True,
                        }
                        pointer_file.write_text(
                            json.dumps(pointer_data, indent=2), encoding="utf-8"
                        )
                        # Also create backward-compatible onedrive_pointer
                        (p / "state.pt.onedrive_pointer.json").write_text(
                            json.dumps(pointer_data, indent=2), encoding="utf-8"
                        )
                        local_state.unlink()
                        item.local_freed = True
                        logger.info(
                            "Freed local checkpoint state %s (saved %d bytes)",
                            local_state,
                            item.size_bytes,
                        )

                    items.append(item)

        # Mirror top-level manifest if present
        top_manifest = ckpt_dir / "checkpoint_manifest.json"
        if top_manifest.exists():
            item = self.mirror_file(top_manifest, target_base / "checkpoint_manifest.json")
            items.append(item)

        return items

    def mirror_distribution_wheels(
        self,
        dist_dir: Path | str = "dist",
    ) -> list[MirroredItem]:
        """Mirror compiled distribution wheels (.whl) and archives to external disk."""
        self.ensure_disk_accessible()
        d_dir = (
            (self.workspace_root / dist_dir).resolve()
            if not Path(dist_dir).is_absolute()
            else Path(dist_dir).resolve()
        )

        if not d_dir.exists():
            return []

        items: list[MirroredItem] = []
        dst_dist = self.disk_root / "dist"

        for f in d_dir.iterdir():
            if f.is_file() and (f.suffix == ".whl" or f.name.endswith(".tar.gz")):
                item = self.mirror_file(f, dst_dist / f.name)
                items.append(item)

        return items

    def mirror_deploy_packages(
        self,
        deploy_dir: Path | str = "deploy",
    ) -> list[MirroredItem]:
        """Mirror deploy configuration folders (ollama, huggingface) to external disk."""
        self.ensure_disk_accessible()
        dep_dir = (
            (self.workspace_root / deploy_dir).resolve()
            if not Path(deploy_dir).is_absolute()
            else Path(deploy_dir).resolve()
        )

        if not dep_dir.exists():
            return []

        items: list[MirroredItem] = []
        dst_deploy = self.disk_root / "deploy"

        for sub in dep_dir.rglob("*"):
            if sub.is_file() and not sub.name.startswith("."):
                rel = sub.relative_to(dep_dir)
                item = self.mirror_file(sub, dst_deploy / rel)
                items.append(item)

        return items

    def write_disk_manifest(self, mirrored_items: list[MirroredItem]) -> Path:
        """Write top-level sync manifest on external disk cataloging all mirrored items."""
        self.ensure_disk_accessible()
        existing_catalog: dict[str, dict[str, Any]] = {}

        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, encoding="utf-8") as f:
                    data = json.load(f)
                    for item_data in data.get("items", []):
                        existing_catalog[item_data["relative_path"]] = item_data
            except Exception as e:
                logger.debug("Existing manifest unreadable, recreating: %s", e)

        for item in mirrored_items:
            existing_catalog[item.relative_path] = item.to_dict()

        total_bytes = sum(it["size_bytes"] for it in existing_catalog.values())
        payload = {
            "disk_root": str(self.disk_root),
            "onedrive_root": str(self.disk_root),
            "last_updated": time.time(),
            "total_items": len(existing_catalog),
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "items": list(existing_catalog.values()),
        }

        tmp_man = self.disk_root / ".tmp_manifest.json"
        tmp_man.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_man, self.manifest_path)

        # Also write onedrive_manifest.json for backward compatibility
        onedrive_manifest = self.disk_root / "onedrive_manifest.json"
        try:
            shutil.copy2(self.manifest_path, onedrive_manifest)
        except Exception:
            pass

        return self.manifest_path

    write_onedrive_manifest = write_disk_manifest

    def audit_sync(self) -> dict[str, Any]:
        """Inspect and compare local files vs external disk mirror."""
        self.ensure_disk_accessible()
        status: dict[str, Any] = {
            "disk_accessible": True,
            "disk_root": str(self.disk_root),
            "onedrive_accessible": True,
            "onedrive_root": str(self.disk_root),
            "manifest_exists": self.manifest_path.exists(),
            "local_checkpoints_freed": 0,
            "bytes_saved_locally": 0,
        }

        ckpt_dir = self.workspace_root / "checkpoints"
        if ckpt_dir.exists():
            freed_count = 0
            freed_bytes = 0
            for ptr in ckpt_dir.rglob("*.disk_pointer.json"):
                freed_count += 1
                try:
                    data = json.loads(ptr.read_text(encoding="utf-8"))
                    freed_bytes += data.get("size_bytes", 0)
                except Exception:
                    pass
            if freed_count == 0:
                for ptr in ckpt_dir.rglob("*.onedrive_pointer.json"):
                    freed_count += 1
                    try:
                        data = json.loads(ptr.read_text(encoding="utf-8"))
                        freed_bytes += data.get("size_bytes", 0)
                    except Exception:
                        pass
            status["local_checkpoints_freed"] = freed_count
            status["bytes_saved_locally"] = freed_bytes
            status["mb_saved_locally"] = round(freed_bytes / (1024 * 1024), 2)

        return status


OneDriveMirrorManager = DiskMirrorManager


def mirror_to_disk(
    destination_path: Path | str | None = None,
    onedrive_path: Path | str | None = None,
    checkpoints_dir: Path | str = "checkpoints",
    dist_dir: Path | str = "dist",
    deploy_dir: Path | str = "deploy",
    free_local: bool = False,
) -> dict[str, Any]:
    """High-level entry point to mirror checkpoints and assets to external disk."""
    target_path = destination_path or onedrive_path
    mgr = DiskMirrorManager(disk_root=target_path)
    mgr.ensure_disk_accessible()

    mirrored = []
    mirrored.extend(mgr.mirror_checkpoints(checkpoints_dir=checkpoints_dir, free_local=free_local))
    mirrored.extend(mgr.mirror_distribution_wheels(dist_dir=dist_dir))
    mirrored.extend(mgr.mirror_deploy_packages(deploy_dir=deploy_dir))

    manifest_file = mgr.write_disk_manifest(mirrored)
    audit = mgr.audit_sync()

    total_size = sum(m.size_bytes for m in mirrored)
    return {
        "mirrored_count": len(mirrored),
        "total_bytes_mirrored": total_size,
        "total_mb_mirrored": round(total_size / (1024 * 1024), 2),
        "manifest_path": str(manifest_file),
        "free_local": free_local,
        "audit": audit,
    }


mirror_to_onedrive = mirror_to_disk
