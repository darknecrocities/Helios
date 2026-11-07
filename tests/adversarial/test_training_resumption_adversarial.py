"""Tier-5 Adversarial Verification Suite for Helios Training Resumption and Fallbacks.

Empirically stress-tests core mathematical and systems invariants:
1. Exact 6-State Resumption Equivalence: Bitwise/bounded loss divergence (< 1e-4),
   exact weight alignment, AdamW momentum (m and v), scheduler LR, and output logits.
2. Loss Spike & NaN/Inf Rollback: Injection of NaN loss, Inf gradients, and extreme loss
   spikes (> 3x EMA) triggering MicroCheckpoint rollback, LR halving, and finite weights.
3. Cursor DataLoader Integrity: Zero duplicated or skipped tokens across interrupted
   multi-epoch streaming, boundary seek stress, and multi-shard dataset streaming.
4. Nano Overfit Robustness: Verified convergence across multiple random seeds and
   varying batch sizes without numerical divergence.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios.training.checkpoint import (
    CheckpointCorruptionError,
    CheckpointManager,
)
from helios.training.config import TrainingConfig
from helios.training.dataloader import (
    CursorDataLoader,
    TokenDataset,
)
from helios.training.smoke_test import (
    build_synthetic_batch,
    run_overfit_one_batch_test,
)
from helios.training.trainer import (
    LossSpikeDetector,
    LossSpikeDivergenceError,
    Trainer,
)


def _build_test_trainer(
    tmp_path: Path,
    vocab: int = 128,
    d_model: int = 32,
    accum_steps: int = 1,
    lr: float = 1e-3,
    model: nn.Module | None = None,
) -> Trainer:
    """Helper creating standardized test trainer with reproducible defaults."""
    if model is None:
        cfg = HeliosConfig(
            vocab_size=vocab, d_model=d_model, n_layers=1, n_heads=1, n_kv_heads=1, max_seq_len=32
        )
        model = HeliosForCausalLM(cfg)
    t_cfg = TrainingConfig(
        learning_rate=lr,
        warmup_steps=2,
        max_steps=10,
        gradient_accumulation_steps=accum_steps,
        checkpoint_dir=str(tmp_path / "ckpts"),
        save_interval_steps=0,
        save_on_interruption=False,
        device="cpu",
    )
    return Trainer(model, t_cfg)


@pytest.mark.tier5
class TestExact6StateResumptionAdversarial:
    """Stress-tests bitwise and numerical equivalence between continuous and resumed runs."""

    def test_continuous_vs_interrupted_resumption_equivalence(self, tmp_path: Path) -> None:
        """Prove exact 6-state alignment and bounded loss divergence (< 1e-4) across steps."""
        model_cfg = HeliosConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ffn=128,
            max_seq_len=32,
        )
        torch.manual_seed(42)
        model_cont = HeliosForCausalLM(model_cfg)
        model_res = copy.deepcopy(model_cont)

        raw_tokens = np.random.randint(10, 240, size=5000, dtype=np.uint16)
        loader_cont = CursorDataLoader(TokenDataset(raw_tokens), batch_size=2, seq_len=16)
        loader_res = CursorDataLoader(TokenDataset(raw_tokens), batch_size=2, seq_len=16)

        trainer_cont = _build_test_trainer(tmp_path / "c", accum_steps=2, model=model_cont, lr=1e-3)

        # Run continuous trainer for 4 global steps (8 microbatches)
        for _ in range(8):
            batch = next(loader_cont)
            trainer_cont.train_step(batch, dataloader_cursor=loader_cont.get_state())

        assert trainer_cont.global_step == 4 and trainer_cont.micro_step == 8
        saved_ckpt = trainer_cont.save_checkpoint()
        assert saved_ckpt.exists()

        # Capture step 4 snapshot state for direct resumption equivalence checking
        weights_at_step_4 = {k: v.cpu().clone() for k, v in model_cont.state_dict().items()}
        opt_at_step_4 = copy.deepcopy(trainer_cont.optimizer.state_dict())
        sched_at_step_4 = copy.deepcopy(trainer_cont.scheduler.state_dict())

        # Run continuous trainer for 4 more global steps (steps 5..8, microbatches 9..16)
        losses_cont: list[float] = []
        for _ in range(8):
            batch = next(loader_cont)
            trainer_cont.train_microbatch(batch)
            if trainer_cont.micro_step % 2 == 0:
                losses_cont.append(trainer_cont.step_optimizer()["loss"])

        assert trainer_cont.global_step == 8 and len(losses_cont) == 4

        # Initialize resumed trainer and restore exact 6 states from disk checkpoint
        trainer_res = _build_test_trainer(tmp_path / "r", accum_steps=2, model=model_res, lr=1e-3)
        restored_meta = trainer_res.load_checkpoint(saved_ckpt)
        loader_res.load_state(restored_meta["dataloader_cursor"])

        assert trainer_res.global_step == 4 and trainer_res.micro_step == 8
        assert loader_res.cursor.token_offset == loader_cont.cursor.token_offset - (8 * 32)

        # 1. Weights alignment at step 4
        for (k_c, v_c), (k_r, v_r) in zip(
            weights_at_step_4.items(), model_res.state_dict().items(), strict=True
        ):
            assert k_c == k_r and torch.allclose(v_c, v_r, atol=1e-7)

        # 2. AdamW momentum (m) and variance (v) buffer alignment
        opt_c = opt_at_step_4["state"]
        opt_r = trainer_res.optimizer.state_dict()["state"]
        for p_idx in opt_c:
            if "exp_avg" in opt_c[p_idx]:
                assert torch.allclose(opt_c[p_idx]["exp_avg"], opt_r[p_idx]["exp_avg"], atol=1e-7)
                assert torch.allclose(
                    opt_c[p_idx]["exp_avg_sq"], opt_r[p_idx]["exp_avg_sq"], atol=1e-7
                )

        # 3. Scheduler LR equivalence
        assert trainer_res.scheduler.state_dict()["_step_count"] == sched_at_step_4["_step_count"]

        # 4. Forward logits equivalence on evaluation probe at step 4
        probe = build_synthetic_batch(vocab_size=256, seq_len=16, batch_size=2, seed=888)
        model_snap = HeliosForCausalLM(model_cfg)
        model_snap.load_state_dict(weights_at_step_4)
        with torch.no_grad():
            logits_snap = model_snap(probe["input_ids"])["logits"]
            logits_res = model_res(probe["input_ids"])["logits"]
        assert torch.allclose(logits_snap, logits_res, atol=1e-5)

        # Execute resumed steps 5..8 and verify loss divergence is bounded (< 1e-4)
        losses_res: list[float] = []
        for _ in range(8):
            batch = next(loader_res)
            trainer_res.train_microbatch(batch)
            if trainer_res.micro_step % 2 == 0:
                losses_res.append(trainer_res.step_optimizer()["loss"])

        assert trainer_res.global_step == 8 and len(losses_res) == 4
        for step_idx, (l_c, l_r) in enumerate(zip(losses_cont, losses_res, strict=True), start=5):
            div = abs(l_c - l_r)
            assert div < 1e-4, f"Divergence at step {step_idx}: {l_c:.6f} vs {l_r:.6f}"

        # Final verification: model probe logits after 4 resumed steps
        with torch.no_grad():
            final_c = model_cont(probe["input_ids"])["logits"]
            final_r = model_res(probe["input_ids"])["logits"]
        assert torch.allclose(final_c, final_r, atol=1e-4)

    def test_checkpoint_tampering_sha256_detection(self, tmp_path: Path) -> None:
        """Verify tampering triggers CheckpointCorruptionError via streaming SHA-256."""
        mgr = CheckpointManager(tmp_path / "tamper", keep_last_n=1)
        model = nn.Linear(8, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        saved = mgr.save_checkpoint(
            step=1, model=model, optimizer=opt, scheduler=None, dataloader_cursor={}, loss=0.5
        )
        assert (saved / "state.pt").exists()

        # Corrupt single byte in weights payload
        with open(saved / "state.pt", "r+b") as f:
            f.seek(10)
            orig = f.read(1)
            f.seek(10)
            f.write(bytes([(orig[0] ^ 0xFF)]))

        fresh_model = nn.Linear(8, 2)
        with pytest.raises(CheckpointCorruptionError, match="Checkpoint corruption detected"):
            mgr.load_checkpoint(saved, fresh_model)


@pytest.mark.tier5
class TestLossSpikeAndRollbackAdversarial:
    """Stress-tests numerical anomaly detection, MicroCheckpoint rollback, and LR decay."""

    def test_extreme_loss_spike_triggers_rollback_and_lr_halving(self, tmp_path: Path) -> None:
        """Inject 5x EMA loss spike and verify automatic rollback with LR decayed by 0.5x."""
        trainer = _build_test_trainer(tmp_path / "spike")
        trainer.spike_detector.warmup_steps = 1
        trainer.spike_detector.spike_multiplier = 3.0

        b1 = build_synthetic_batch(vocab_size=128, seq_len=8, batch_size=2, seed=1)
        b2 = build_synthetic_batch(vocab_size=128, seq_len=8, batch_size=2, seed=2)
        trainer.train_step(b1)
        trainer.train_step(b2)
        assert trainer.global_step == 2 and trainer.spike_detector.clean_snapshot is not None
        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        # Step 3: Inject extreme loss spike (> 3x EMA)
        b3 = build_synthetic_batch(vocab_size=128, seq_len=8, batch_size=2, seed=3)
        trainer.train_microbatch(b3)
        ema = trainer.spike_detector.ema_loss or 1.0
        trainer.accumulated_loss = ema * 5.0

        res = trainer.step_optimizer()
        assert res["status"] == "rollback"
        assert "Loss exploded" in res["reason"]
        assert trainer.global_step == 1  # Restored to 0-indexed step 1 snapshot

        new_lr = trainer.optimizer.param_groups[0]["lr"]
        assert abs(new_lr - initial_lr * 0.5) < 1e-9

        # Subsequent valid step successfully advances to step 2 with decayed LR
        b4 = build_synthetic_batch(vocab_size=128, seq_len=8, batch_size=2, seed=4)
        trainer.train_step(b4)
        assert trainer.global_step == 2

    def test_inf_gradient_injection_triggers_rollback(self, tmp_path: Path) -> None:
        """Inject infinite gradients and prove rollback prevents parameter contamination."""
        trainer = _build_test_trainer(tmp_path / "inf")
        trainer.spike_detector.warmup_steps = 2

        b1 = build_synthetic_batch(vocab_size=128, seq_len=8, batch_size=2, seed=1)
        b2 = build_synthetic_batch(vocab_size=128, seq_len=8, batch_size=2, seed=2)
        trainer.train_step(b1)
        trainer.train_step(b2)
        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        # Inject Inf into gradients
        b3 = build_synthetic_batch(vocab_size=128, seq_len=8, batch_size=2, seed=3)
        trainer.train_microbatch(b3)
        for p in trainer.model.parameters():
            if p.grad is not None:
                p.grad.fill_(float("inf"))
                break

        res = trainer.step_optimizer()
        assert res["status"] == "rollback"
        assert "Non-finite grad_norm" in res["reason"] or "Gradient norm explosion" in res["reason"]
        assert trainer.global_step == 1
        assert trainer.optimizer.param_groups[0]["lr"] == initial_lr * 0.5

        for p in trainer.model.parameters():
            assert torch.isfinite(p).all()

    def test_nan_loss_injection_triggers_rollback(self, tmp_path: Path) -> None:
        """Inject NaN loss with clean snapshot available and prove rollback without corruption."""
        trainer = _build_test_trainer(tmp_path / "nan")
        trainer.spike_detector.warmup_steps = 2

        b1 = build_synthetic_batch(vocab_size=128, seq_len=8, batch_size=2, seed=1)
        b2 = build_synthetic_batch(vocab_size=128, seq_len=8, batch_size=2, seed=2)
        trainer.train_step(b1)
        trainer.train_step(b2)

        # Inject NaN loss
        b3 = build_synthetic_batch(vocab_size=128, seq_len=8, batch_size=2, seed=3)
        trainer.train_microbatch(b3)
        trainer.accumulated_loss = float("nan")

        res = trainer.step_optimizer()
        assert res["status"] == "rollback"
        assert "Non-finite loss" in res["reason"]
        assert trainer.global_step == 1

        for p in trainer.model.parameters():
            assert torch.isfinite(p).all()

    def test_consecutive_spikes_exceed_ceiling_raises_divergence(self) -> None:
        """Prove repeated numerical spikes beyond threshold raise LossSpikeDivergenceError."""
        detector = LossSpikeDetector(max_consecutive_spikes=3)
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        detector.record_clean_snapshot(1, 1, 1.0, model, opt, None, {"step": 1})
        for _ in range(3):
            detector.rollback(model, opt, None, torch.device("cpu"))
        assert detector.consecutive_spikes == 3

        with pytest.raises(LossSpikeDivergenceError, match="Exceeded maximum consecutive spikes"):
            detector.rollback(model, opt, None, torch.device("cpu"))


@pytest.mark.tier5
class TestCursorDataLoaderIntegrityAdversarial:
    """Stress-tests token streaming integrity, zero-skip/duplicate guarantees, and seeking."""

    def test_zero_token_duplication_or_skip_across_interrupted_epochs(self) -> None:
        """Prove uninterrupted vs interrupted multi-epoch token streams are 100% identical."""
        total_tokens = 320
        batch_size = 2
        seq_len = 8
        tokens_per_batch = batch_size * seq_len  # 16 tokens/batch
        batches_per_epoch = total_tokens // tokens_per_batch  # 20 batches
        num_epochs = 3
        total_batches = batches_per_epoch * num_epochs  # 60 batches

        tokens = np.arange(total_tokens, dtype=np.uint16)
        dataset = TokenDataset(tokens)

        # Continuous reference stream
        loader_cont = CursorDataLoader(dataset, batch_size=batch_size, seq_len=seq_len)
        ground_truth: list[int] = []
        for _ in range(total_batches):
            b = next(loader_cont)
            ground_truth.extend(b["input_ids"].view(-1).tolist())

        assert len(ground_truth) == total_tokens * num_epochs

        # Interrupted stream across mid-epoch and epoch boundaries
        interruption_points = {7, 19, 20, 35, 40, 52}
        loader_res = CursorDataLoader(dataset, batch_size=batch_size, seq_len=seq_len)
        resumed_stream: list[int] = []

        for b_idx in range(total_batches):
            if b_idx in interruption_points:
                state = loader_res.get_state()
                fresh_loader = CursorDataLoader(dataset, batch_size=batch_size, seq_len=seq_len)
                fresh_loader.load_state(state)
                loader_res = fresh_loader

            b = next(loader_res)
            resumed_stream.extend(b["input_ids"].view(-1).tolist())

        # Exact empirical proof: zero tokens duplicated, zero tokens skipped
        assert len(resumed_stream) == len(ground_truth)
        assert resumed_stream == ground_truth

    def test_dataloader_random_seek_stress(self) -> None:
        """Seek across 25 random token offsets and assert bitwise token alignment."""
        tokens = np.arange(1000, dtype=np.uint16)
        dataset = TokenDataset(tokens)
        loader = CursorDataLoader(dataset, batch_size=4, seq_len=16)
        tokens_per_batch = 64

        rng = np.random.RandomState(42)
        for _ in range(25):
            offset = int(rng.randint(0, 1000 - tokens_per_batch))
            loader.seek(token_offset=offset, epoch=1)
            assert loader.cursor.token_offset == offset

            batch = next(loader)
            expected = tokens[offset : offset + tokens_per_batch]
            actual = batch["input_ids"].view(-1).numpy()
            assert np.array_equal(actual, expected)
            assert loader.cursor.token_offset == offset + tokens_per_batch

    def test_dataloader_seek_boundary_validation(self) -> None:
        """Verify out-of-bounds seeks correctly raise ValueError."""
        dataset = TokenDataset(np.arange(100, dtype=np.uint16))
        loader = CursorDataLoader(dataset, batch_size=2, seq_len=5)
        with pytest.raises(ValueError, match="out of bounds"):
            loader.seek(-1)
        with pytest.raises(ValueError, match="out of bounds"):
            loader.seek(100)

    def test_multi_shard_concatenated_boundary_streaming(self, tmp_path: Path) -> None:
        """Stream across virtual shard boundaries and assert seamless token concatenation."""
        shard0 = tmp_path / "shard_0.bin"
        shard1 = tmp_path / "shard_1.bin"
        np.arange(0, 100, dtype=np.uint16).tofile(shard0)
        np.arange(100, 200, dtype=np.uint16).tofile(shard1)

        dataset = TokenDataset([shard0, shard1], dtype=np.uint16)
        assert len(dataset) == 200

        loader = CursorDataLoader(dataset, batch_size=2, seq_len=10)
        # Seek straddling shard0 [0..99] and shard1 [100..199]
        loader.seek(90)
        batch = next(loader)
        actual = batch["input_ids"].view(-1).numpy()
        expected = np.arange(90, 110, dtype=np.uint16)
        assert np.array_equal(actual, expected)
        assert loader.cursor.token_offset == 110


@pytest.mark.tier5
class TestNanoOverfitRobustnessAdversarial:
    """Stress-tests overfit convergence across random seeds and batch dimensions."""

    @pytest.mark.parametrize(
        "seed,batch_size",
        [
            (42, 1),
            (123, 2),
            (777, 4),
        ],
    )
    def test_nano_overfit_convergence_multi_seed_and_batch_sizes(
        self, seed: int, batch_size: int
    ) -> None:
        """Verify reliable overfit convergence (< 0.08) with zero NaNs across multiple seeds."""
        config = HeliosConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            d_ffn=128,
            max_seq_len=32,
        )
        torch.manual_seed(seed)
        model = HeliosForCausalLM(config)
        train_cfg = TrainingConfig(
            learning_rate=3.5e-3,
            min_learning_rate=1e-4,
            warmup_steps=3,
            max_steps=120,
            gradient_accumulation_steps=1,
            grad_clip=1.0,
            device="cpu",
            nan_inf_detection=True,
        )
        res = run_overfit_one_batch_test(
            model=model,
            config=train_cfg,
            target_loss=0.08,
            max_steps=120,
            batch_size=batch_size,
            seq_len=16,
            device="cpu",
        )
        assert res["success"] is True
        assert res["initial_loss"] > 3.0 and res["final_loss"] < 0.08
        for loss in res["loss_history"]:
            assert not np.isnan(loss) and not np.isinf(loss)

    def test_gradient_accumulation_and_clip_stability(self) -> None:
        """Verify gradient accumulation with clipping retains finite gradients and steps cleanly."""
        model_cfg = HeliosConfig(vocab_size=128, d_model=32, n_layers=1, n_heads=1, n_kv_heads=1)
        model = HeliosForCausalLM(model_cfg)
        train_cfg = TrainingConfig(
            device="cpu",
            gradient_accumulation_steps=3,
            grad_clip=0.5,
            save_interval_steps=0,
            nan_inf_detection=True,
        )
        trainer = Trainer(model, train_cfg)
        batch = build_synthetic_batch(vocab_size=128, seq_len=8, batch_size=2)

        # Micro steps 1 and 2: no optimizer step
        trainer.train_step(batch)
        assert trainer.micro_step == 1 and trainer.global_step == 0
        trainer.train_step(batch)
        assert trainer.micro_step == 2 and trainer.global_step == 0

        # Micro step 3: accumulate into gradients, check unclipped norm, then step
        trainer.train_microbatch(batch)
        assert trainer.micro_step == 3
        grad_norm_before_step = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]
            )
        )
        assert grad_norm_before_step > 0.0

        step_res = trainer.step_optimizer()
        assert step_res["status"] == "success"
        assert trainer.global_step == 1
        assert trainer.last_grad_norm > 0.0
        # After optimizer step, optimizer.zero_grad(set_to_none=True) freed gradients
        for p in model.parameters():
            assert p.grad is None
            assert torch.isfinite(p).all()
