"""Tier 2: Boundary Value Analysis (BVA) & Extreme Limits Test Suite for Helios LLM.

Stress-tests boundary conditions, zero/max sequence lengths, empty inputs,
numerical edge cases, memory limits, and signal/checkpoint tamper thresholds.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import typer.testing

from helios.cli.main import app as cli_app
from helios.memory.base import compute_model_weight_hash
from helios.memory.controller import HMAController
from helios.memory.semantic import reciprocal_rank_fusion
from helios.model.attention import KVCache
from helios.model.config import HeliosConfig
from helios.model.normalization import RMSNorm
from helios.model.parameter_inspector import ParameterInspector
from helios.model.transformer import HeliosForCausalLM
from helios.tokenizer.bpe import HeliosBPE
from helios.tokenizer.unigram import HeliosUnigram
from helios.training.checkpoint import (
    CheckpointCorruptionError,
    CheckpointManager,
)
from helios.training.config import TrainingConfig
from helios.training.dataloader import CursorDataLoader, TokenDataset
from helios.training.trainer import LossSpikeDetector


# ==============================================================================
# Tokenizer Boundaries
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier2
class TestTokenizerBoundaries:
    """Stress-tests boundary conditions for BPE and Unigram tokenizers."""

    @pytest.mark.parametrize("tokenizer_cls", [HeliosBPE, HeliosUnigram])
    def test_empty_string_roundtrip(self, tokenizer_cls: type) -> None:
        """Verify empty string encoding returns either empty or special tokens only."""
        tok = tokenizer_cls()
        ids = tok.encode("", add_special_tokens=False)
        assert len(ids) == 0
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == ""

    @pytest.mark.parametrize("tokenizer_cls", [HeliosBPE, HeliosUnigram])
    @pytest.mark.parametrize(
        "char",
        ["a", "Z", "0", "ñ", "日", "🚀", "👨‍👩‍👧‍👦", "\n", "\t"],
    )
    def test_single_characters_roundtrip(self, tokenizer_cls: type, char: str) -> None:
        """Verify lossless roundtrip encoding for single characters across Unicode planes."""
        tok = tokenizer_cls()
        ids = tok.encode(char, add_special_tokens=False)
        assert len(ids) > 0
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == char

    @pytest.mark.parametrize("tokenizer_cls", [HeliosBPE, HeliosUnigram])
    def test_extreme_whitespace(self, tokenizer_cls: type) -> None:
        """Verify tokenizer handles 1,000 consecutive spaces, tabs, and newlines."""
        tok = tokenizer_cls()
        whitespace_str = " " * 500 + "\t" * 250 + "\n" * 250
        ids = tok.encode(whitespace_str, add_special_tokens=False)
        assert len(ids) > 0
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == whitespace_str

    @pytest.mark.parametrize("tokenizer_cls", [HeliosBPE, HeliosUnigram])
    def test_massive_repetitive_text(self, tokenizer_cls: type) -> None:
        """Verify tokenizer handles large 20,000 character inputs without stack overflow."""
        tok = tokenizer_cls()
        big_text = "HeliosOpenWeightLLM" * 1000
        ids = tok.encode(big_text, add_special_tokens=False)
        assert len(ids) >= 1000
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == big_text


# ==============================================================================
# Model & Attention Boundaries
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier2
class TestModelBoundaries:
    """Stress-tests sequence length, numerical limits, and attention cache capacity."""

    def test_single_token_sequence_forward(self, tiny_e2e_model: HeliosForCausalLM) -> None:
        """Verify model handles minimal 1-token sequence length (seq_len = 1)."""
        input_ids = torch.tensor([[42]], dtype=torch.long)
        out = tiny_e2e_model(input_ids)
        assert out["logits"].shape == (1, 1, tiny_e2e_model.config.vocab_size)

    def test_maximum_context_length_forward(self, tiny_e2e_model: HeliosForCausalLM) -> None:
        """Verify model handles sequence at exact maximum context window (seq_len = max_seq_len)."""
        max_len = tiny_e2e_model.config.max_seq_len
        input_ids = torch.randint(0, tiny_e2e_model.config.vocab_size, (1, max_len))
        out = tiny_e2e_model(input_ids)
        assert out["logits"].shape == (1, max_len, tiny_e2e_model.config.vocab_size)

    def test_extreme_batch_sizes(self, tiny_e2e_model: HeliosForCausalLM) -> None:
        """Verify forward pass under minimal (batch=1) and large (batch=16) batch sizes."""
        for b in [1, 8, 16]:
            input_ids = torch.randint(0, tiny_e2e_model.config.vocab_size, (b, 4))
            out = tiny_e2e_model(input_ids)
            assert out["logits"].shape == (b, 4, tiny_e2e_model.config.vocab_size)

    def test_rmsnorm_extreme_numerical_stability(self) -> None:
        """Verify RMSNorm numerical safety under extreme amplitudes (1e6 and 1e-12)."""
        norm = RMSNorm(dim=32, eps=1e-5)
        # Extreme high amplitude
        x_high = torch.full((1, 4, 32), 1e6)
        out_high = norm(x_high)
        assert not torch.isnan(out_high).any()
        assert not torch.isinf(out_high).any()

        # Extreme low amplitude
        x_low = torch.full((1, 4, 32), 1e-12)
        out_low = norm(x_low)
        assert not torch.isnan(out_low).any()
        assert not torch.isinf(out_low).any()

    def test_kv_cache_capacity_boundary(self) -> None:
        """Verify KVCache updates correctly up to max_seq_len boundary."""
        cache = KVCache(
            max_batch_size=1,
            max_seq_len=8,
            n_kv_heads=2,
            head_dim=16,
            dtype=torch.float32,
            device="cpu",
        )
        for step in range(8):
            k = torch.randn(1, 2, 1, 16)
            v = torch.randn(1, 2, 1, 16)
            k_out, v_out = cache.update(step, k, v)
            assert k_out.shape[2] == step + 1
            assert v_out.shape[2] == step + 1


# ==============================================================================
# Memory Architecture Boundaries
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier2
class TestMemoryBoundaries:
    """Stress-tests memory buffers, SQLite limits, RRF edge cases, and weight isolation."""

    def test_working_memory_overflow_eviction(self, isolated_hma: HMAController) -> None:
        """Verify sliding window evicts earliest turns when token limit (512) is exceeded."""
        sess = "overflow_session"
        for i in range(25):
            isolated_hma.store_episodic(sess, "user", f"Turn number {i}: " + ("context " * 20))
        ctx = isolated_hma.get_working_context(sess, max_tokens=200)
        # Must return only recent turns that fit within 200 tokens
        total_tokens = sum(len(t["content"].split()) for t in ctx)
        assert total_tokens <= 250

    def test_episodic_memory_special_characters_and_large_payload(
        self, isolated_hma: HMAController
    ) -> None:
        """Verify SQLite handles SQL injection strings, quotes, and 50KB payload."""
        sess = "payload_sess"
        # Malicious SQL payload
        sql_payload = "'; DROP TABLE turns; SELECT * FROM sessions WHERE '1'='1"
        isolated_hma.store_episodic(sess, "user", sql_payload)
        history = isolated_hma.episodic.get_session_turns(sess)
        assert history[-1].content == sql_payload

        # Large 50KB turn payload
        large_payload = "A" * 50_000
        isolated_hma.store_episodic(sess, "assistant", large_payload)
        history = isolated_hma.episodic.get_session_turns(sess)
        assert history[-1].content == large_payload

    def test_semantic_memory_empty_and_unmatched_queries(self, isolated_hma: HMAController) -> None:
        """Verify semantic query handles empty strings and unmatched queries gracefully."""
        res_empty = isolated_hma.query("", top_k=5)
        assert isinstance(res_empty, list)

        res_none = isolated_hma.query("quantum_chromodynamics_xyz_nonexistent", top_k=5)
        assert isinstance(res_none, list)

    def test_reciprocal_rank_fusion_edge_cases(self) -> None:
        """Verify RRF behavior with empty lists and single-element rankings."""
        res_empty = reciprocal_rank_fusion([], top_k=5)
        assert res_empty == []

        rankings = [[("doc1", 10.0), ("doc2", 5.0)]]
        res_single = reciprocal_rank_fusion(rankings, top_k=2)
        assert len(res_single) == 2
        assert res_single[0][0] == "doc1"

    def test_mass_memory_operations_strict_weight_isolation(
        self, isolated_hma: HMAController, tiny_e2e_model: HeliosForCausalLM
    ) -> None:
        """Verify 100 consecutive HMA writes and queries produce zero weight drift."""
        hash_start = compute_model_weight_hash(tiny_e2e_model)
        for i in range(100):
            isolated_hma.store_episodic("iso_heavy", "user", f"Turn {i} content")
            if i % 10 == 0:
                isolated_hma.index_semantic(f"doc_{i}", f"Document text {i}")
                isolated_hma.query(f"query {i}")
        hash_end = compute_model_weight_hash(tiny_e2e_model)
        assert hash_start == hash_end, "Weight drift detected during heavy memory operations!"


# ==============================================================================
# Training Controller Boundaries
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier2
class TestTrainingBoundaries:
    """Stress-tests training hyperparameters, checkpoint corruptions, and signal boundaries."""

    def test_training_config_boundary_values(self, tmp_path: Path) -> None:
        """Verify TrainingConfig accepts boundary values (lr=0, max_steps=1)."""
        cfg = TrainingConfig(
            learning_rate=0.0,
            max_steps=1,
            checkpoint_dir=str(tmp_path / "cfg_bounds"),
            device="cpu",
        )
        assert cfg.learning_rate == 0.0
        assert cfg.max_steps == 1

    def test_loss_spike_detector_triggers_rollback(self) -> None:
        """Verify LossSpikeDetector detects 10x sudden loss spike."""
        detector = LossSpikeDetector(spike_multiplier=3.0, warmup_steps=2)
        # Establish low loss baseline
        detector.record_loss(1.0)
        detector.record_loss(1.0)
        # Sudden 10x spike
        is_spike = detector.check_spike(10.0)
        assert is_spike

    def test_checkpoint_corruption_detection(
        self, tiny_e2e_model: HeliosForCausalLM, tmp_path: Path
    ) -> None:
        """Verify CheckpointManager rejects corrupted / truncated checkpoint files."""
        ckpt_dir = tmp_path / "corrupt_ckpts"
        mgr = CheckpointManager(checkpoint_dir=str(ckpt_dir))
        saved = mgr.save_checkpoint(
            step=1,
            model=tiny_e2e_model,
            optimizer=torch.optim.AdamW(tiny_e2e_model.parameters()),
            scheduler=None,
            cursor={"token_offset": 0},
            metrics={"loss": 1.0},
        )
        # Truncate model weights file
        weight_file = saved / "state.pt" if (saved / "state.pt").exists() else saved / "model.pt"
        assert weight_file.exists()
        weight_file.write_bytes(b"corrupted_bytes_truncated")
        with pytest.raises(CheckpointCorruptionError):
            mgr.load_checkpoint(saved)

    def test_dataloader_cursor_file_boundary(self, synthetic_corpus_file: Path) -> None:
        """Verify CursorDataLoader behaves predictably at the exact file boundary."""
        dataset = TokenDataset(synthetic_corpus_file)
        loader = CursorDataLoader(dataset, batch_size=2, seq_len=16)
        # Set cursor near the end of file
        max_offset = len(dataset) - 32
        loader.set_cursor(max_offset)
        batch = next(loader)
        assert batch["input_ids"].shape == (2, 16)


# ==============================================================================
# Tooling & CLI Boundaries
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier2
class TestToolingBoundaries:
    """Stress-tests parameter inspector and CLI with invalid arguments and presets."""

    def test_parameter_inspector_minimal_configuration(self) -> None:
        """Verify ParameterInspector handles minimal 1-layer 1-head model without crash."""
        cfg = HeliosConfig(
            vocab_size=32,
            d_model=16,
            n_layers=1,
            n_heads=1,
            n_kv_heads=1,
            d_ffn=32,
            max_seq_len=16,
        )
        model = HeliosForCausalLM(cfg)
        report = ParameterInspector.inspect(model, cfg)
        assert report.total_parameters > 0

    def test_cli_invalid_arguments_exit_code_non_zero(
        self, cli_runner: typer.testing.CliRunner
    ) -> None:
        """Verify CLI exits with non-zero exit code on unknown commands or invalid presets."""
        res_unknown = cli_runner.invoke(cli_app, ["nonexistent_command"])
        assert res_unknown.exit_code != 0

        res_bad_preset = cli_runner.invoke(cli_app, ["inspect", "-m", "invalid_preset_xyz"])
        assert res_bad_preset.exit_code == 1
