"""End-to-end integration test of the HELIOS pipeline using synthetic data and tiny model."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from helios_pipeline.cli.run_cmd import execute_pipeline
from helios_pipeline.storage.manager import OneTouchStorageManager, StorageConfig
from helios_pipeline.training.atomic_checkpoint import AtomicCheckpointManager


def test_pipeline_synthetic_execution_and_resumption() -> None:
    """Run full pipeline iteration, verify checkpoints and documentation, then resume."""
    with TemporaryDirectory() as tmp_dir:
        storage_root = Path(tmp_dir) / "OneTouchMock"
        storage_root.mkdir(parents=True, exist_ok=True)

        mock_config = StorageConfig(
            external_drive_path=str(storage_root),
            candidate_paths=[str(storage_root)],
            require_external_drive=False,
            prohibit_internal_storage_fallback=False,
            minimum_free_space_gb=0.01,
            quota_limit_gb=100.0,
        )

        with patch(
            "helios_pipeline.cli.run_cmd.OneTouchStorageManager",
            return_value=OneTouchStorageManager(config=mock_config),
        ):
            # Pass 1: Run 1 cluster, 4 steps
            code1 = execute_pipeline(
                mode="once",
                dry_run=False,
                synthetic=True,
                tier="nano",
                clusters_count=1,
                max_steps_per_cluster=4,
                device="cpu",
                git_sync=False,
            )
            assert code1 == 0

            # Verify checkpoint created
            storage = OneTouchStorageManager(config=mock_config)
            ckpt_mgr = AtomicCheckpointManager(storage.checkpoints_dir, storage.staging_dir)
            ckpts = ckpt_mgr.list_checkpoints()
            assert len(ckpts) >= 1
            payload, latest_path = ckpt_mgr.load_latest_checkpoint()
            assert payload is not None
            step_after_pass1 = payload["dataloader_cursor"]["global_step"]
            assert step_after_pass1 > 0

            # Pass 2: Resume for another cluster
            code2 = execute_pipeline(
                mode="once",
                dry_run=False,
                synthetic=True,
                tier="nano",
                clusters_count=1,
                max_steps_per_cluster=4,
                device="cpu",
                git_sync=False,
            )
            assert code2 == 0

            # Verify resumed step advanced
            payload2, _ = ckpt_mgr.load_latest_checkpoint()
            assert payload2 is not None
            step_after_pass2 = payload2["dataloader_cursor"]["global_step"]
            assert step_after_pass2 > step_after_pass1
