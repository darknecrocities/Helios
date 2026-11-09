"""Unit tests for conservative storage cleanup engine and safety preconditions."""

from pathlib import Path
from tempfile import TemporaryDirectory

from helios_pipeline.data.manifest import ClusterManifest
from helios_pipeline.storage.cleanup import ConservativeCleanupEngine
from helios_pipeline.storage.manager import OneTouchStorageManager, StorageConfig
from helios_pipeline.testing.tiny_model import create_tiny_helios_model
from helios_pipeline.training.atomic_checkpoint import AtomicCheckpointManager


def test_cleanup_refuses_when_fewer_than_two_checkpoints() -> None:
    """Precondition: cleanup must abort if fewer than 2 valid checkpoints exist."""
    with TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        cfg = StorageConfig(
            external_drive_path=str(base),
            candidate_paths=[str(base)],
            require_external_drive=False,
            prohibit_internal_storage_fallback=False,
        )
        storage = OneTouchStorageManager(config=cfg)
        ckpt_mgr = AtomicCheckpointManager(storage.checkpoints_dir, storage.staging_dir)
        cleaner = ConservativeCleanupEngine(storage, ckpt_mgr)

        manifest = ClusterManifest(
            cluster_id="test_cluster",
            status="COMPLETED",
            content_hash="verified_hash_123",
        )

        # 0 checkpoints: should abort
        can_clean, reason = cleaner.can_safely_cleanup(manifest)
        assert not can_clean
        assert "Only 0 valid checkpoints found" in reason

        # Create 1 checkpoint
        model = create_tiny_helios_model(vocab_size=128)
        ckpt_mgr.save_checkpoint(
            model=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            global_step=10,
            epoch=0,
            loss=2.5,
            cluster_id="c1",
            batch_idx=0,
        )

        can_clean, reason = cleaner.can_safely_cleanup(manifest)
        assert not can_clean
        assert "Only 1 valid checkpoints found" in reason

        # Create 2nd checkpoint
        ckpt_mgr.save_checkpoint(
            model=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            global_step=20,
            epoch=0,
            loss=2.1,
            cluster_id="c2",
            batch_idx=0,
        )

        can_clean, reason = cleaner.can_safely_cleanup(manifest)
        assert can_clean


def test_cleanup_deletes_scratch_when_preconditions_met() -> None:
    """Verify clean deletion of staging scratch directory when preconditions pass."""
    with TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        cfg = StorageConfig(
            external_drive_path=str(base),
            candidate_paths=[str(base)],
            require_external_drive=False,
            prohibit_internal_storage_fallback=False,
        )
        storage = OneTouchStorageManager(config=cfg)
        ckpt_mgr = AtomicCheckpointManager(storage.checkpoints_dir, storage.staging_dir)
        cleaner = ConservativeCleanupEngine(storage, ckpt_mgr)

        model = create_tiny_helios_model(vocab_size=128)
        ckpt_mgr.save_checkpoint(model, None, None, None, 10, 0, 2.0, "c1", 0)
        ckpt_mgr.save_checkpoint(model, None, None, None, 20, 0, 1.8, "c2", 0)

        # Create scratch directory for cluster
        manifest = ClusterManifest(
            cluster_id="cluster_alpha",
            status="COMPLETED",
            content_hash="sha256_mock_digest",
        )
        scratch_dir = storage.cache_dir / f".tmp_{manifest.cluster_id}"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        (scratch_dir / "temp_shard.dat").write_bytes(b"disposable temporary shard data")
        assert scratch_dir.exists()

        res = cleaner.cleanup_cluster_scratch(manifest, dry_run=False)
        assert res["status"] == "COMPLETED"
        assert res["reclaimed_bytes"] > 0
        assert not scratch_dir.exists()
