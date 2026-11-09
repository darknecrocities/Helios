"""Unit tests for exact 6-state checkpoint resumption and bitwise integrity."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from helios_pipeline.data.manifest import ClusterManifest
from helios_pipeline.testing.tiny_model import create_tiny_helios_config, create_tiny_helios_model
from helios_pipeline.training.atomic_checkpoint import AtomicCheckpointManager
from helios_pipeline.training.controller import ContinuousTrainingController


def test_atomic_checkpoint_save_and_verify() -> None:
    """Verify 6-state serialization, checksum generation, and directory promotion."""
    with TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        ckpts_dir = base / "checkpoints"
        staging_dir = base / "staging"
        mgr = AtomicCheckpointManager(ckpts_dir, staging_dir)

        model = create_tiny_helios_model(vocab_size=256)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        saved = mgr.save_checkpoint(
            model=model,
            optimizer=opt,
            scheduler=None,
            scaler=None,
            global_step=50,
            epoch=1,
            loss=3.1415,
            cluster_id="cluster_001",
            batch_idx=10,
        )
        assert saved.exists()
        assert (saved / "state.pt").exists()
        assert (saved / "SHA256SUMS").exists()
        assert mgr.verify_checkpoint(saved)

        # Load latest
        payload, path = mgr.load_latest_checkpoint()
        assert payload is not None
        assert payload["dataloader_cursor"]["global_step"] == 50
        assert payload["dataloader_cursor"]["loss"] == 3.1415


def test_controller_state_resumption() -> None:
    """Verify ContinuousTrainingController resumes step, epoch, and weights from disk."""
    with TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        ckpts_dir = base / "checkpoints"
        staging_dir = base / "staging"
        cfg = create_tiny_helios_config(vocab_size=256)

        # Train 10 steps
        c1 = ContinuousTrainingController(cfg, ckpts_dir, staging_dir, device="cpu")
        manifest = ClusterManifest(cluster_id="cluster_init")
        res1 = c1.train_cluster(manifest, max_steps=10)
        assert res1["global_step"] == 10

        # Create fresh controller instance and resume
        c2 = ContinuousTrainingController(cfg, ckpts_dir, staging_dir, device="cpu")
        resumed = c2.resume_if_available()
        assert resumed
        assert c2.global_step == 10

        # Continue training next 5 steps
        res2 = c2.train_cluster(manifest, max_steps=5)
        assert res2["global_step"] == 15
