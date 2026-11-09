"""Strict external OneTouch storage manager enforcing zero internal storage fallback and quota boundaries."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DriveDisconnectedError(RuntimeError):
    """Raised when the required external OneTouch volume is missing or unmounted."""


class StorageQuotaExceededError(RuntimeError):
    """Raised when pipeline disk usage exceeds the storage budget ceiling."""


class InternalStorageViolationError(RuntimeError):
    """Raised when an operation targets or falls back to the internal host filesystem."""


@dataclass
class StorageConfig:
    """Configuration for external OneTouch persistent volume."""

    external_drive_path: str = "/Volumes/One Touch"
    candidate_paths: list[str] = field(
        default_factory=lambda: ["/Volumes/One Touch", "/Volumes/OneTouch"]
    )
    require_external_drive: bool = True
    prohibit_internal_storage_fallback: bool = True
    total_capacity_gb: float = 1000.0
    minimum_free_space_gb: float = 100.0  # 10% reserve capacity floor
    quota_limit_gb: float = 450.0  # Sub-500GB quota ceiling

    # Directory layout relative to mount point
    base_folder: str = "Helios"
    dataset_cache: str = "datasets/cache"
    raw_shards: str = "datasets/raw"
    processed_data: str = "datasets/processed"
    tokenized_data: str = "datasets/tokenized"
    checkpoints: str = "checkpoints"
    checkpoint_staging: str = "checkpoints/staging"
    logs: str = "logs"
    manifests: str = "manifests"


class OneTouchStorageManager:
    """Manages directory lifecycle and verifies filesystem isolation on the OneTouch volume."""

    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or StorageConfig()
        self.root_dev: int = Path("/").stat().st_dev
        self.mount_path: Path = self._discover_and_validate_mount()

        # Initialize subpaths
        self.base_dir: Path = self.mount_path / self.config.base_folder
        self.cache_dir: Path = self.base_dir / self.config.dataset_cache
        self.raw_dir: Path = self.base_dir / self.config.raw_shards
        self.processed_dir: Path = self.base_dir / self.config.processed_data
        self.tokenized_dir: Path = self.base_dir / self.config.tokenized_data
        self.checkpoints_dir: Path = self.base_dir / self.config.checkpoints
        self.staging_dir: Path = self.base_dir / self.config.checkpoint_staging
        self.logs_dir: Path = self.base_dir / self.config.logs
        self.manifests_dir: Path = self.base_dir / self.config.manifests

        self._ensure_directories()

    def _discover_and_validate_mount(self) -> Path:
        """Locate external mount, verify device identity, and prohibit internal fallback."""
        target: Path | None = None
        for cand in self.config.candidate_paths:
            p = Path(cand)
            if p.exists() and os.path.ismount(str(p)):
                target = p
                break

        if target is None:
            # Check configured path directly
            p = Path(self.config.external_drive_path)
            if p.exists() and os.path.ismount(str(p)):
                target = p

        if target is None:
            if self.config.require_external_drive:
                raise DriveDisconnectedError(
                    f"External storage required but not mounted at candidate paths: "
                    f"{self.config.candidate_paths}. Internal fallback strictly prohibited."
                )
            target = Path(self.config.external_drive_path).resolve()

        self.assert_not_internal(target)
        return target

    def assert_not_internal(self, path: Path | str) -> None:
        """Assert that path does not reside on the host internal root filesystem."""
        if not self.config.prohibit_internal_storage_fallback:
            return

        p = Path(path).resolve()
        while not p.exists() and p != p.parent:
            p = p.parent

        if p.exists() and p.stat().st_dev == self.root_dev:
            raise InternalStorageViolationError(
                f"Path '{path}' resolves to internal storage device (st_dev={self.root_dev}). "
                f"Silent internal storage fallback is prohibited by safety policy."
            )

    def _ensure_directories(self) -> None:
        """Create required directory hierarchy on external storage."""
        dirs = [
            self.base_dir,
            self.cache_dir,
            self.raw_dir,
            self.processed_dir,
            self.tokenized_dir,
            self.checkpoints_dir,
            self.staging_dir,
            self.logs_dir,
            self.manifests_dir,
        ]
        for d in dirs:
            self.assert_not_internal(d)
            d.mkdir(parents=True, exist_ok=True)

    def check_health(self) -> dict[str, Any]:
        """Verify storage connection, reserve capacity, and quota limits."""
        if not self.mount_path.exists():
            raise DriveDisconnectedError(
                f"Drive disconnected: Mount path {self.mount_path} missing."
            )

        self.assert_not_internal(self.mount_path)

        usage = shutil.disk_usage(str(self.mount_path))
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)

        if free_gb < self.config.minimum_free_space_gb:
            raise StorageQuotaExceededError(
                f"Disk free space ({free_gb:.1f} GB) is below the required 10% reserve floor "
                f"({self.config.minimum_free_space_gb:.0f} GB). Training halted."
            )

        # Calculate Helios usage
        helios_bytes = sum(
            f.stat().st_size for f in self.base_dir.rglob("*") if f.is_file() and not f.is_symlink()
        )
        helios_gb = helios_bytes / (1024**3)

        if helios_gb > self.config.quota_limit_gb:
            raise StorageQuotaExceededError(
                f"Helios storage usage ({helios_gb:.1f} GB) exceeds quota limit "
                f"({self.config.quota_limit_gb:.0f} GB). Clean disposable caches before resuming."
            )

        return {
            "mount_path": str(self.mount_path),
            "device_id": self.mount_path.stat().st_dev,
            "total_gb": total_gb,
            "free_gb": free_gb,
            "helios_usage_gb": helios_gb,
            "status": "HEALTHY",
        }


StorageManager = OneTouchStorageManager
