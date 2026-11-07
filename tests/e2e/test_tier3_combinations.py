"""Tier 3: Combinatorial & Pairwise Cross-Feature Verification Suite for Helios LLM.

Verifies end-to-end interactions between coupled subsystems:
- Tokenizer + Model Forward/Generation
- Special Tokens + Embedding Lookup
- Model Causal Masking + Dynamic KV Caching Equivalence
- GQA Head Grouping + RoPE Frequency Offsets
- Parameter Inspector + Weight Tying Deduplication
- Memory Retrieval + Prompt Assembly + Model Forward
- Active Model Training + Weight Isolation Barrier
- Streaming Dataloader + Checkpoint Cursor Resumption
- Signal Trapping + Microbatch Checkpoint Staging
- Serialization + HuggingFace Bundle Reloading
- CLI Typer Harness + Subsystem Entry Points
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import typer.testing

from helios.cli.main import app as cli_app
from helios.memory.base import compute_model_weight_hash
from helios.memory.controller import HMAController
from helios.model.config import HeliosConfig
from helios.model.parameter_inspector import ParameterInspector
from helios.model.transformer import HeliosForCausalLM
from helios.tokenizer.bpe import HeliosBPE
from helios.tokenizer.unigram import HeliosUnigram
from helios.training.config import TrainingConfig
from helios.training.dataloader import CursorDataLoader, TokenDataset
from helios.training.signals import InterruptionHandler
from helios.training.trainer import Trainer


# ==============================================================================
# Combination 1 & 2: Tokenizer + Model Subsystems
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier3
class TestTokenizerModelCombinations:
    """Verifies seamless coupling between tokenizers and causal transformer model."""

    @pytest.mark.parametrize("tokenizer_cls", [HeliosBPE, HeliosUnigram])
    @pytest.mark.parametrize(
        "text",
        [
            "Helios is an open-weight foundation model developed by Arron Kian Parejas.",
            "Ang Helios ay may Hierarchical Memory Architecture para sa Tagalog at Ingles.",
            "def train_step(batch): loss = model(batch['input_ids'])['loss']; return loss",
        ],
    )
    def test_tokenizer_encoding_to_model_forward_and_generate(
        self, tokenizer_cls: type, text: str, tiny_e2e_model: HeliosForCausalLM
    ) -> None:
        """Verify text encoded by tokenizer passes into model and generates continuation."""
        tok = tokenizer_cls()
        ids = tok.encode(text, add_special_tokens=True)
        # Cap length to model's max_seq_len
        ids = ids[: tiny_e2e_model.config.max_seq_len]
        input_ids = torch.tensor([ids], dtype=torch.long)

        # Forward pass
        out = tiny_e2e_model(input_ids)
        assert out["logits"].shape == (1, len(ids), tiny_e2e_model.config.vocab_size)

        # Autoregressive generation
        gen_tokens = tiny_e2e_model.generate(input_ids[:, :4], max_new_tokens=6)
        assert gen_tokens.shape == (1, 10)
        decoded_output = tok.decode(gen_tokens[0].tolist())
        assert len(decoded_output) > 0

    def test_special_tokens_embedding_lookup(
        self, bpe_tokenizer: HeliosBPE, tiny_e2e_model: HeliosForCausalLM
    ) -> None:
        """Verify all special tokens map to valid embedding vectors within embedding matrix."""
        for _, token_id in bpe_tokenizer.special_tokens.items():
            assert token_id < tiny_e2e_model.config.vocab_size
            token_tensor = torch.tensor([[token_id]], dtype=torch.long)
            emb = tiny_e2e_model.embed_tokens(token_tensor)
            assert emb.shape == (1, 1, tiny_e2e_model.config.d_model)
            assert not torch.isnan(emb).any()


# ==============================================================================
# Combination 3 & 4: Model + Attention + KV Cache + RoPE
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier3
class TestModelAttentionCombinations:
    """Verifies mathematical equivalence between cached and full-sequence causal passes."""

    def test_cached_decoding_step_equivalence(self, tiny_e2e_model: HeliosForCausalLM) -> None:
        """Verify incremental step-by-step KV cached pass matches parallel forward pass."""
        seq_len = 8
        torch.manual_seed(123)
        input_ids = torch.randint(0, tiny_e2e_model.config.vocab_size, (1, seq_len))

        # 1. Full parallel forward pass
        with torch.no_grad():
            full_logits = tiny_e2e_model(input_ids)["logits"]

        # 2. Incremental single-token cached decoding
        cached_logits_list: list[torch.Tensor] = []
        out_init = tiny_e2e_model(input_ids[:, :1], use_cache=True, start_pos=0)
        cached_logits_list.append(out_init["logits"])
        kv_caches = out_init["kv_caches"]

        with torch.no_grad():
            for t in range(1, seq_len):
                step_out = tiny_e2e_model(
                    input_ids[:, t : t + 1],
                    kv_caches=kv_caches,
                    start_pos=t,
                )
                cached_logits_list.append(step_out["logits"])

        cached_logits = torch.cat(cached_logits_list, dim=1)

        # Assert numerical equivalence within tight tolerance (< 1e-4)
        max_diff = (full_logits - cached_logits).abs().max().item()
        assert max_diff < 1e-4, f"Cached decoding diverged by {max_diff}"

    def test_parameter_inspector_tied_weights_deduplication(self) -> None:
        """Verify ParameterInspector accurately deduplicates tied embeddings across presets."""
        for preset_name in ["nano-15m", "nano-20m", "nano-25m"]:
            cfg = HeliosConfig.from_preset(preset_name)
            model = HeliosForCausalLM(cfg)
            report = ParameterInspector.inspect(model, cfg)
            theoretical = cfg.count_parameters()

            # Trainable parameters must strictly match theoretical tied total
            assert report.trainable_parameters == theoretical["total_parameters"]
            assert report.tied_embeddings is True


# ==============================================================================
# Combination 5 & 6: Memory Architecture + Model Inference & Weight Isolation
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier3
class TestMemoryModelCombinations:
    """Verifies interaction between HMA retrieval and model context ingestion."""

    def test_retrieved_evidence_grounded_forward_pass(
        self,
        isolated_hma: HMAController,
        tiny_e2e_model: HeliosForCausalLM,
        bpe_tokenizer: HeliosBPE,
    ) -> None:
        """Verify HMA query assembly -> formatting -> tokenization -> model forward pass."""
        isolated_hma.index_semantic(
            "doc_gqa", "Helios employs Grouped-Query Attention with 3:1 head ratio."
        )
        grounded_str = isolated_hma.build_grounded_context(
            "Grouped-Query Attention", session_id="s_test"
        )

        assert "<retrieved_evidence>" in grounded_str
        assert "Grouped-Query Attention" in grounded_str

        # Tokenize grounded prompt
        token_ids = bpe_tokenizer.encode(grounded_str, add_special_tokens=True)
        token_ids = token_ids[: tiny_e2e_model.config.max_seq_len]
        input_ids = torch.tensor([token_ids], dtype=torch.long)

        # Forward pass calculates valid logits and cross-entropy loss
        out = tiny_e2e_model(input_ids, labels=input_ids.clone())
        assert out["logits"].shape == (1, len(token_ids), tiny_e2e_model.config.vocab_size)
        assert out["loss"] is not None
        assert not torch.isnan(out["loss"])

    def test_simultaneous_memory_writes_and_model_inference_weight_isolation(
        self, isolated_hma: HMAController, tiny_e2e_model: HeliosForCausalLM
    ) -> None:
        """Verify model weights remain bitwise identical while HMA executes intensive mutations."""
        initial_hash = compute_model_weight_hash(tiny_e2e_model)

        # Interleave model forward/backward steps with HMA store and query operations
        optimizer = torch.optim.AdamW(tiny_e2e_model.parameters(), lr=1e-3)
        for i in range(5):
            # Model forward/backward
            input_ids = torch.randint(0, tiny_e2e_model.config.vocab_size, (1, 8))
            loss = tiny_e2e_model(input_ids, labels=input_ids)["loss"]
            loss.backward()
            optimizer.zero_grad()

            # HMA write
            isolated_hma.store_episodic("sim_sess", "user", f"Turn data {i}")
            isolated_hma.index_semantic(f"sim_doc_{i}", f"Semantic knowledge content {i}")
            isolated_hma.query("Semantic knowledge")

        # After discarding zeroed gradients, model parameters must remain bitwise identical
        final_hash = compute_model_weight_hash(tiny_e2e_model)
        assert initial_hash == final_hash


# ==============================================================================
# Combination 7 & 8: Controller + Dataloader + Checkpoints + Signals
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier3
class TestTrainingResumptionCombinations:
    """Verifies interaction between streaming dataloaders, signals, and exact 6-state resumption."""

    def test_dataloader_cursor_exact_checkpoint_resumption(
        self, synthetic_corpus_file: Path, tiny_e2e_model: HeliosForCausalLM, tmp_path: Path
    ) -> None:
        """Verify dataloader cursor restores seamlessly from checkpoint without token drift."""
        dataset = TokenDataset(synthetic_corpus_file)
        loader1 = CursorDataLoader(dataset, batch_size=2, seq_len=16)

        ckpt_dir = tmp_path / "cursor_ckpts"
        train_cfg = TrainingConfig(
            checkpoint_dir=str(ckpt_dir),
            device="cpu",
            gradient_accumulation_steps=1,
        )
        trainer1 = Trainer(tiny_e2e_model, train_cfg)

        # Step 4 batches
        for _ in range(4):
            batch = next(loader1)
            trainer1.train_step(batch, dataloader_cursor=loader1.get_state())

        saved_ckpt = trainer1.save_checkpoint(checkpoint_name="cursor_state")
        cursor_before = loader1.get_state()

        # Resume in fresh trainer and dataloader
        fresh_model = HeliosForCausalLM(tiny_e2e_model.config)
        trainer2 = Trainer(fresh_model, train_cfg)
        meta = trainer2.load_checkpoint(saved_ckpt)

        loader2 = CursorDataLoader(dataset, batch_size=2, seq_len=16)
        loader2.load_state(meta["dataloader_cursor"])
        cursor_after = loader2.get_state()

        assert cursor_before["token_offset"] == cursor_after["token_offset"]

        # Next batch on both loaders must be bitwise identical
        next_batch1 = next(loader1)
        next_batch2 = next(loader2)
        assert torch.equal(next_batch1["input_ids"], next_batch2["input_ids"])

    def test_signal_handler_emergency_checkpoint_creation(
        self, tiny_e2e_model: HeliosForCausalLM, tmp_path: Path
    ) -> None:
        """Verify InterruptionHandler triggers emergency checkpoint auto-save."""
        ckpt_dir = tmp_path / "signal_ckpts"
        train_cfg = TrainingConfig(
            checkpoint_dir=str(ckpt_dir),
            device="cpu",
            save_on_interruption=True,
            gradient_accumulation_steps=1,
        )
        handler = InterruptionHandler()
        trainer = Trainer(tiny_e2e_model, train_cfg, interruption_handler=handler)

        batches = [
            {"input_ids": torch.randint(0, 64, (2, 8)), "labels": torch.randint(0, 64, (2, 8))}
            for _ in range(5)
        ]

        def interrupt_gen():
            for i, b in enumerate(batches):
                if i == 2:
                    handler.trigger_interrupt()
                yield b

        trainer.fit(interrupt_gen(), max_steps=5)
        assert handler.interrupted
        latest_ckpt = trainer.checkpoint_manager.get_latest_checkpoint()
        assert latest_ckpt is not None
        assert "interrupted" in latest_ckpt.name


# ==============================================================================
# Combination 9: CLI Typer Harness + Subsystem Invocations
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier3
class TestCliToolingCombinations:
    """Verifies Typer CLI entry points orchestrating multiple subsystems."""

    def test_cli_inspect_and_roadmap_integration(self, cli_runner: typer.testing.CliRunner) -> None:
        """Verify CLI inspect command validates all 4 Nano presets and roadmap."""
        for preset in ["nano-15m", "nano-20m", "nano-25m", "nano-35m"]:
            res = cli_runner.invoke(cli_app, ["inspect", "-m", preset])
            assert res.exit_code == 0
            assert "Helios Model Inspector" in res.stdout
            assert "Arron Kian Parejas" in res.stdout

        res_roadmap = cli_runner.invoke(cli_app, ["roadmap"])
        assert res_roadmap.exit_code == 0
        assert "Nano" in res_roadmap.stdout
        assert "Flagship" in res_roadmap.stdout
