"""Tier-5 Adversarial Verification: Signal Trapping, Checkpoint Tamper Resistance & Atomicity.

Empirically challenges:
1. Microbatch boundary integrity:
   - Signals (SIGINT/SIGTERM) delivered during active training loop preserve in-flight
     forward/backward passes without mid-pass severance.
   - Rapid signal interception halts cleanly at microbatch boundary without partial weight updates.
2. Watchdog and double-signal abort:
   - Repeated signals force immediate process termination via os._exit(128 + signum).
   - EmergencyWatchdog timeout aborts hung persistence cleanly before external SIGKILL.
3. SHA-256 tamper resistance:
   - Single-byte mutation in state.pt triggers CheckpointCorruptionError and aborts before
     mutating target model/optimizer state (0 corrupted state loaded).
   - Single-byte mutation in manifest.json sha256 triggers CheckpointCorruptionError with
     zero parameter leakage.
4. Atomic staging durability:
   - Incomplete .tmp staging directories and .tmp files are completely ignored during discovery.
   - Incomplete .tmp artifacts are automatically purged upon CheckpointManager initialization.
5. Best-loss shielding:
   - 15-step training trajectory with keep_last_n=2 shields global minimum (step 3) from deletion.
   - Comprehensive retention rotation stress-test verifies sliding window + best loss preservation.

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

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
from helios.training.signals import InterruptionHandler, SignalState
from helios.training.trainer import Trainer


@pytest.fixture
def tiny_config() -> HeliosConfig:
    """Minimal HeliosConfig for fast deterministic testing."""
    return HeliosConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ffn=64,
        max_seq_len=16,
    )


@pytest.fixture
def tiny_model(tiny_config: HeliosConfig) -> HeliosForCausalLM:
    """Deterministic tiny model instance."""
    torch.manual_seed(42)
    return HeliosForCausalLM(tiny_config)


@pytest.mark.tier5
class TestMicrobatchBoundaryIntegrity:
    """Verify signal trapping defers shutdown to clean microbatch boundaries."""

    def test_signal_interception_preserves_inflight_microbatch(
        self, tiny_model: HeliosForCausalLM, tmp_path: Path
    ) -> None:
        """Deliver SIGINT mid-loop; verify in-flight forward/backward completes cleanly."""
        train_cfg = TrainingConfig(
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu",
            gradient_accumulation_steps=2,
            save_on_interruption=True,
        )
        handler = InterruptionHandler()
        trainer = Trainer(tiny_model, train_cfg, interruption_handler=handler)

        batches = [
            {
                "input_ids": torch.randint(0, 64, (2, 8)),
                "labels": torch.randint(0, 64, (2, 8)),
            }
            for _ in range(6)
        ]

        signal_delivered = False

        def deliver_signal_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            nonlocal signal_delivered
            if not signal_delivered:
                handler.trigger_interrupt(signal.SIGINT)
                signal_delivered = True

        # Hook into first layer forward pass to deliver signal during computation
        hook_handle = tiny_model.layers[0].register_forward_pre_hook(deliver_signal_hook)

        completed_microbatches = 0
        with handler:
            for batch in batches:
                if handler.should_stop():
                    trainer.save_checkpoint(is_interrupted=True)
                    break

                loss = trainer.train_microbatch(batch)
                completed_microbatches += 1
                assert loss > 0.0

                # Verify all parameter gradients are valid finite numbers
                for p in tiny_model.parameters():
                    if p.requires_grad and p.grad is not None:
                        assert not torch.isnan(p.grad).any()
                        assert not torch.isinf(p.grad).any()

        hook_handle.remove()

        # Microbatch 0 completed its forward & backward cleanly despite mid-pass signal
        assert completed_microbatches == 1
        assert signal_delivered is True
        assert handler.interrupted
        assert handler.received_signal == signal.SIGINT

        # Verify emergency interrupted checkpoint was written and is valid
        latest = trainer.checkpoint_manager.get_latest_checkpoint()
        assert latest is not None
        assert "interrupted" in latest.name

    def test_sigterm_trapping_at_microbatch_boundary(
        self, tiny_model: HeliosForCausalLM, tmp_path: Path
    ) -> None:
        """Verify SIGTERM follows identical graceful boundary semantics as SIGINT."""
        train_cfg = TrainingConfig(
            checkpoint_dir=str(tmp_path / "sigterm_ckpts"),
            device="cpu",
            save_on_interruption=True,
        )
        handler = InterruptionHandler(signals=(signal.SIGINT, signal.SIGTERM))
        trainer = Trainer(tiny_model, train_cfg, interruption_handler=handler)

        batch = {
            "input_ids": torch.randint(0, 64, (2, 8)),
            "labels": torch.randint(0, 64, (2, 8)),
        }

        signal_delivered = False

        def deliver_sigterm_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            nonlocal signal_delivered
            if not signal_delivered:
                handler.trigger_interrupt(signal.SIGTERM)
                signal_delivered = True

        hook_handle = tiny_model.layers[0].register_forward_pre_hook(deliver_sigterm_hook)

        with handler:
            assert not handler.should_stop()
            # In-flight microbatch executes forward and backward pass
            loss = trainer.train_microbatch(batch)
            assert isinstance(loss, float)
            assert loss > 0.0

            # Signal was delivered during pass; boundary poll now indicates stop
            assert signal_delivered is True
            assert handler.should_stop()
            assert handler.state == SignalState.INTERRUPT_REQUESTED
            assert handler.received_signal == signal.SIGTERM

            # Calling train_microbatch again raises KeyboardInterrupt because stop was requested
            with pytest.raises(KeyboardInterrupt, match="Training interrupted"):
                trainer.train_microbatch(batch)

        hook_handle.remove()


@pytest.mark.tier5
class TestWatchdogAndDoubleSignalAbort:
    """Verify rapid repeated signals and persistent hanging persistence abort process."""

    def test_double_signal_triggers_immediate_process_abort(self) -> None:
        """Second signal delivery triggers os._exit(128 + signum) in subprocess."""
        code = (
            "import signal, sys\n"
            "from helios.training.signals import InterruptionHandler\n"
            "handler = InterruptionHandler(auto_exit_on_second_signal=True)\n"
            "with handler:\n"
            "    handler.trigger_interrupt(signal.SIGINT)\n"
            "    assert handler.signal_count == 1\n"
            "    handler.trigger_interrupt(signal.SIGINT)\n"
            "    sys.stdout.write('FAILED_SHOULD_HAVE_EXITED')\n"
            "    sys.stdout.flush()\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        expected_code = 128 + signal.SIGINT
        assert proc.returncode == expected_code
        assert "FAILED_SHOULD_HAVE_EXITED" not in proc.stdout

    def test_emergency_watchdog_hang_aborts_process(self) -> None:
        """Watchdog timeout triggers emergency fallback when persistence blocks."""
        code = (
            "import time, sys\n"
            "from helios.training.signals import InterruptionHandler\n"
            "handler = InterruptionHandler(watchdog_timeout=0.2)\n"
            "def hung_action():\n"
            "    time.sleep(10.0)\n"
            "handler.execute_with_watchdog(hung_action)\n"
            "sys.stdout.write('FAILED_WATCHDOG_DID_NOT_ABORT')\n"
            "sys.stdout.flush()\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        expected_code = 128 + signal.SIGTERM
        assert proc.returncode == expected_code
        assert "CRITICAL HELIOS EMERGENCY" in proc.stderr
        assert "FAILED_WATCHDOG_DID_NOT_ABORT" not in proc.stdout


@pytest.mark.tier5
class TestCheckpointTamperResistance:
    """Verify cryptographic SHA-256 validation prevents loading corrupted state."""

    def test_single_byte_mutation_in_state_pt_rejects_and_loads_zero_state(
        self, tmp_path: Path
    ) -> None:
        """Mutate single byte in state.pt; assert CheckpointCorruptionError and 0 mutation."""
        ckpt_dir = tmp_path / "tamper_ckpts"
        mgr = CheckpointManager(ckpt_dir)

        model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        saved_path = mgr.save_checkpoint(
            step=1,
            model=model,
            optimizer=opt,
            scheduler=None,
            dataloader_cursor={"batch_idx": 42},
            loss=1.234,
        )

        # Mutate a single byte at offset 64 in state.pt
        state_file = saved_path / "state.pt"
        data = bytearray(state_file.read_bytes())
        data[64] ^= 0x55
        state_file.write_bytes(bytes(data))

        # Fresh model with distinct known weights
        fresh_model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
        fresh_weights_before = {k: v.clone() for k, v in fresh_model.state_dict().items()}

        with pytest.raises(CheckpointCorruptionError, match="Checkpoint corruption detected"):
            mgr.load_checkpoint(saved_path, fresh_model)

        # Empirically verify 0 corrupted state was loaded into fresh_model
        for k, v in fresh_model.state_dict().items():
            assert torch.equal(v, fresh_weights_before[k])

    def test_single_byte_mutation_in_manifest_sha_rejects(self, tmp_path: Path) -> None:
        """Mutate single character of sha256 in manifest.json; assert corruption error."""
        ckpt_dir = tmp_path / "manifest_tamper"
        mgr = CheckpointManager(ckpt_dir)

        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        saved_path = mgr.save_checkpoint(
            step=1, model=model, optimizer=opt, scheduler=None, dataloader_cursor={}, loss=0.5
        )

        manifest_file = saved_path / "manifest.json"
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))

        # Flip the first character of sha256
        old_sha = manifest_data["sha256"]
        flipped_char = "b" if old_sha[0] != "b" else "a"
        manifest_data["sha256"] = flipped_char + old_sha[1:]
        manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

        fresh_model = nn.Linear(4, 2)
        fresh_weights_before = {k: v.clone() for k, v in fresh_model.state_dict().items()}

        with pytest.raises(CheckpointCorruptionError, match="SHA-256 mismatch"):
            mgr.load_checkpoint(saved_path, fresh_model)

        # Assert zero state loaded
        for k, v in fresh_model.state_dict().items():
            assert torch.equal(v, fresh_weights_before[k])


@pytest.mark.tier5
class TestAtomicStagingDurability:
    """Verify incomplete .tmp artifacts are completely ignored and pruned upon resume."""

    def test_incomplete_tmp_directories_and_files_ignored(self, tmp_path: Path) -> None:
        """Incomplete .tmp staging directories and files are ignored during checkpoint discovery."""
        ckpt_dir = tmp_path / "staging_ckpts"
        mgr = CheckpointManager(ckpt_dir)

        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        # Save step 1 and step 2
        mgr.save_checkpoint(
            step=1,
            model=model,
            optimizer=opt,
            scheduler=None,
            dataloader_cursor={},
            loss=0.9,
        )
        step2_dir = mgr.save_checkpoint(
            step=2, model=model, optimizer=opt, scheduler=None, dataloader_cursor={}, loss=0.8
        )

        # Inject various incomplete .tmp artifacts
        (ckpt_dir / ".tmp_checkpoint_step_00000003_123456").mkdir()
        ((ckpt_dir / ".tmp_checkpoint_step_00000003_123456") / "state.pt.tmp").write_bytes(b"bad")
        (ckpt_dir / "checkpoint_step_00000004.tmp").mkdir()
        (ckpt_dir / "stray.tmp").write_text("broken")
        (ckpt_dir / "checkpoint_manifest.json.tmp").write_text("{}")

        # Verify discovery ignores all .tmp items and returns valid step 2
        latest = mgr.get_latest_checkpoint()
        assert latest is not None
        assert latest.resolve() == step2_dir.resolve()

        best = mgr.get_best_checkpoint()
        assert best is not None
        assert best.resolve() == step2_dir.resolve()

    def test_incomplete_tmp_artifacts_pruned_on_new_manager_init(self, tmp_path: Path) -> None:
        """Re-instantiating CheckpointManager automatically purges all .tmp artifacts."""
        ckpt_dir = tmp_path / "pruning_init_ckpts"
        mgr = CheckpointManager(ckpt_dir)

        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        valid_dir = mgr.save_checkpoint(
            step=1, model=model, optimizer=opt, scheduler=None, dataloader_cursor={}, loss=0.5
        )

        # Inject dangling .tmp files and directories
        tmp_dir_1 = ckpt_dir / ".tmp_checkpoint_step_00000002_999"
        tmp_dir_1.mkdir()
        tmp_file_1 = ckpt_dir / "stray_payload.tmp"
        tmp_file_1.write_bytes(b"junk")

        assert tmp_dir_1.exists()
        assert tmp_file_1.exists()

        # Re-initialize CheckpointManager (simulating process restart / resume)
        new_mgr = CheckpointManager(ckpt_dir)

        assert not tmp_dir_1.exists()
        assert not tmp_file_1.exists()
        assert valid_dir.exists()

        # Load succeeds cleanly
        restored = new_mgr.load_checkpoint(valid_dir, model)
        assert restored["step"] == 1


@pytest.mark.tier5
class TestBestLossShielding:
    """Verify global lowest-loss checkpoint is shielded from rotation deletion across 15 steps."""

    def test_best_loss_shielded_across_15_steps_keep_last_2(self, tmp_path: Path) -> None:
        """Save across 15 steps with keep_last_n=2; verify step 3 (best) is never deleted."""
        ckpt_dir = tmp_path / "shield_ckpts"
        mgr = CheckpointManager(ckpt_dir, keep_last_n=2, keep_best=True)

        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        # Step losses: step 3 has global minimum (0.042)
        losses = [
            1.50,  # step 1
            1.20,  # step 2
            0.042,  # step 3: GLOBAL MINIMUM
            0.95,  # step 4
            0.90,  # step 5
            0.85,  # step 6
            0.80,  # step 7
            0.75,  # step 8
            0.70,  # step 9
            0.65,  # step 10
            0.60,  # step 11
            0.55,  # step 12
            0.50,  # step 13
            0.45,  # step 14
            0.40,  # step 15
        ]

        for idx, loss_val in enumerate(losses, start=1):
            mgr.save_checkpoint(
                step=idx,
                model=model,
                optimizer=opt,
                scheduler=None,
                dataloader_cursor={"step": idx},
                loss=loss_val,
            )

        # Check existing checkpoint directories on disk
        remaining_dirs = sorted(p.name for p in ckpt_dir.iterdir() if p.is_dir())

        # Step 3 MUST be shielded and present!
        assert "checkpoint_step_00000003" in remaining_dirs
        # Steps 14 and 15 MUST be present (last 2)!
        assert "checkpoint_step_00000014" in remaining_dirs
        assert "checkpoint_step_00000015" in remaining_dirs

        # All other non-best older steps MUST be pruned!
        for step_num in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            assert f"checkpoint_step_{step_num:08d}" not in remaining_dirs

        # Exactly 3 directories remain: 1 best + 2 sliding window
        assert len(remaining_dirs) == 3

        # Verify best checkpoint discovery and exact restoration
        best_ckpt = mgr.get_best_checkpoint()
        assert best_ckpt is not None
        assert best_ckpt.name == "checkpoint_step_00000003"

        fresh_model = nn.Linear(4, 2)
        restored = mgr.load_checkpoint(best_ckpt, fresh_model)
        assert restored["step"] == 3
        assert restored["loss"] == 0.042
        assert restored["is_best"] is True

    def test_trainer_integration_best_loss_shielding(
        self, tiny_model: HeliosForCausalLM, tmp_path: Path
    ) -> None:
        """Integration test: Trainer with keep_best=True preserves best step in multi-step run."""
        ckpt_dir = tmp_path / "trainer_shield_ckpts"
        train_cfg = TrainingConfig(
            checkpoint_dir=str(ckpt_dir),
            device="cpu",
            save_interval_steps=1,
            keep_last_n_checkpoints=2,
        )
        trainer = Trainer(tiny_model, train_cfg)
        trainer.checkpoint_manager.keep_best = True

        # Manually save 15 steps with step 3 as lowest loss
        for step in range(1, 16):
            loss = 0.01 if step == 3 else float(1.0 + step * 0.1)
            trainer.global_step = step
            trainer.last_loss = loss
            trainer.save_checkpoint()

        remaining = sorted(p.name for p in ckpt_dir.iterdir() if p.is_dir())
        assert "checkpoint_step_00000003" in remaining
        assert "checkpoint_step_00000014" in remaining
        assert "checkpoint_step_00000015" in remaining
        assert len(remaining) == 3
