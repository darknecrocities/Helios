"""Conservative storage cleanup manager enforcing checkpoint and manifest verification before unlinking."""

from __future__ import annotations

import logging
import shutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass
from helios_pipeline.storage.manager import OneTouchStorageManager
from helios_pipeline.training.atomic_checkpoint import AtomicCheckpointManager

logger = logging.getLogger(__name__)


class ConservativeCleanupEngine:
    """Safely cleans up temporary processing shards while strictly protecting checkpoints and datasets."""

    def __init__(
        self,
        storage_manager: OneTouchStorageManager,
        checkpoint_manager: AtomicCheckpointManager,
    ) -> None:
        self.storage = storage_manager
        self.checkpoints = checkpoint_manager

    def can_safely_cleanup(self, manifest: Any | None = None) -> tuple[bool, str]:
        """Check all safety preconditions before allowing any file removal."""
        # Precondition 1: Checkpoint redundancy check (must have >= 2 valid checkpoints)
        valid_ckpts = self.checkpoints.list_checkpoints()
        if len(valid_ckpts) < 2:
            return (
                False,
                f"Safety invariant failed: Only {len(valid_ckpts)} valid checkpoints found (minimum 2 required).",
            )

        # Precondition 2: If a manifest is supplied, verify it reached COMPLETED status
        if manifest is not None:
            if getattr(manifest, "status", "") != "COMPLETED":
                return (
                    False,
                    f"Cannot clean cluster {getattr(manifest, 'cluster_id', 'unknown')}: Status is not 'COMPLETED'.",
                )
            if not getattr(manifest, "content_hash", ""):
                return (
                    False,
                    f"Cannot clean cluster {getattr(manifest, 'cluster_id', 'unknown')}: Missing verified content hash.",
                )

        return True, "Preconditions satisfied for conservative cleanup."

    def cleanup_cluster_scratch(
        self,
        manifest: Any,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Clean intermediate staging files for a verified completed cluster."""
        can_clean, reason = self.can_safely_cleanup(manifest)
        if not can_clean:
            logger.warning(f"Cleanup aborted: {reason}")
            return {"reclaimed_bytes": 0, "status": "ABORTED", "reason": reason}

        reclaimed_bytes = 0
        deleted_paths: list[str] = []
        cluster_id = getattr(manifest, "cluster_id", "unknown")

        scratch_candidates = [
            self.storage.cache_dir / f".tmp_{cluster_id}",
            self.storage.staging_dir / f".tmp_{cluster_id}",
            self.storage.processed_dir / f".tmp_{cluster_id}",
        ]

        for cand in scratch_candidates:
            if cand.exists():
                size = sum(f.stat().st_size for f in cand.rglob("*") if f.is_file())
                if not dry_run:
                    shutil.rmtree(cand, ignore_errors=True)
                reclaimed_bytes += size
                deleted_paths.append(str(cand))
                logger.info(
                    f"{'[DRY RUN] Would delete' if dry_run else 'Cleaned up'} scratch path: {cand} ({size / 1024:.1f} KB)"
                )

        return {
            "reclaimed_bytes": reclaimed_bytes,
            "status": "COMPLETED",
            "deleted_paths": deleted_paths,
            "dry_run": dry_run,
        }

    def sweep_dangling_staging(self, dry_run: bool = False) -> int:
        """Scan staging directories for orphaned .tmp_* files."""
        valid_ckpts = self.checkpoints.list_checkpoints()
        if len(valid_ckpts) < 2:
            logger.warning("Skipping staging sweep: Fewer than 2 valid checkpoints exist.")
            return 0

        reclaimed = 0
        for base_dir in [self.storage.staging_dir, self.storage.cache_dir]:
            if not base_dir.exists():
                continue
            for item in base_dir.glob(".tmp_*"):
                try:
                    if item.is_file():
                        size = item.stat().st_size
                        if not dry_run:
                            item.unlink(missing_ok=True)
                        reclaimed += size
                    elif item.is_dir():
                        size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                        if not dry_run:
                            shutil.rmtree(item, ignore_errors=True)
                        reclaimed += size
                except Exception as e:
                    logger.warning(f"Failed to remove dangling temp item {item}: {e}")

        logger.info(f"Swept dangling staging files. Reclaimed: {reclaimed / (1024 * 1024):.2f} MB")
        return reclaimed
