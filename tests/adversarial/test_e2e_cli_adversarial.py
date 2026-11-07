"""Tier-5 Adversarial Verification Suite for E2E Test Suites & Interactive Live CLI.

Empirically stress-tests:
1. Single-token generation KV cache speedup fast-path numerical equivalence to
   full causal unmasked attention passes across single and multi-batch runs.
2. CLI generation boundary extremes: empty prompts, whitespace, extreme sampling
   parameters (temperature <= 0, top_k <= 0, extreme top_k, max_new_tokens <= 0).
3. Long-horizon generation & sequence length boundary overflow.
4. Multi-turn chat memory accumulation, SQLite persistence, and saturation boundaries.
5. CLI eval & export command error resilience.

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from helios.cli.chat import ChatSession
from helios.cli.generate import stream_generate
from helios.cli.main import app as cli_app
from helios.cli.utils import load_model_and_config, load_or_create_tokenizer
from helios.export.disk import DiskMirrorManager
from helios.model.attention import KVCache
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM

cli_runner = CliRunner()


@pytest.fixture
def adversarial_cli_model() -> tuple[HeliosForCausalLM, HeliosConfig]:
    """Lightweight deterministic model for adversarial CLI & KV cache tests."""
    torch.manual_seed(42)
    cfg = HeliosConfig(
        vocab_size=512,
        d_model=64,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        d_ffn=128,
        max_seq_len=256,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
    )
    model = HeliosForCausalLM(cfg).eval()
    return model, cfg


# ==============================================================================
# 1. Single-Token KV Cache Speedup Fast-Path vs Full Causal Pass Equivalence
# ==============================================================================
@pytest.mark.tier5
class TestKVCacheFastPathEquivalence:
    """Empirical mathematical proof that single-token KV cache decode produces

    numerically identical outputs to full causal attention passes without drift.
    """

    def test_single_token_decode_fast_path_identical_to_full_causal_pass(
        self,
        adversarial_cli_model: tuple[HeliosForCausalLM, HeliosConfig],
    ) -> None:
        """Verify that single-token KV cache decode matches full causal attention.

        Iterates 16 autoregressive steps and checks max absolute logit diff < 1e-5.
        """
        model, cfg = adversarial_cli_model
        prompt_len = 8
        gen_len = 16
        head_dim = cfg.d_model // cfg.n_heads

        torch.manual_seed(2026)
        prompt_tokens = torch.randint(0, cfg.vocab_size, (1, prompt_len))

        # Method A: Full causal forward pass at every decode step
        seq_full = prompt_tokens.clone()
        logits_full: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(gen_len):
                out = model.forward(seq_full, start_pos=0, kv_caches=None)
                last_logits = out["logits"][:, -1, :]
                logits_full.append(last_logits)
                next_tok = torch.argmax(last_logits, dim=-1, keepdim=True)
                seq_full = torch.cat([seq_full, next_tok], dim=-1)

        # Method B: Pre-allocated contiguous KV Cache fast-path
        kv_caches = [
            KVCache(
                max_batch_size=1,
                max_seq_len=prompt_len + gen_len + 16,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=head_dim,
                dtype=torch.float32,
            )
            for _ in range(cfg.n_layers)
        ]

        seq_kv = prompt_tokens.clone()
        logits_kv: list[torch.Tensor] = []
        with torch.no_grad():
            # Step 0: Prefill
            out = model.forward(prompt_tokens, start_pos=0, kv_caches=kv_caches)
            last_logits = out["logits"][:, -1, :]
            logits_kv.append(last_logits)
            cur_tok = torch.argmax(last_logits, dim=-1, keepdim=True)
            seq_kv = torch.cat([seq_kv, cur_tok], dim=-1)

            # Steps 1..gen_len-1: Incremental single-token decode
            for step in range(1, gen_len):
                cur_pos = prompt_len + step - 1
                out = model.forward(cur_tok, start_pos=cur_pos, kv_caches=kv_caches)
                step_logits = out["logits"][:, 0, :]
                logits_kv.append(step_logits)
                cur_tok = torch.argmax(step_logits, dim=-1, keepdim=True)
                seq_kv = torch.cat([seq_kv, cur_tok], dim=-1)

        # Assertions
        assert len(logits_full) == len(logits_kv) == gen_len
        max_drift = 0.0
        for step in range(gen_len):
            diff = (logits_full[step] - logits_kv[step]).abs().max().item()
            max_drift = max(max_drift, diff)
            assert diff < 1e-5, f"Step {step} numerical drift {diff:.2e} >= 1e-5"

        # Bitwise token identity
        assert (seq_full == seq_kv).all().item(), "Generated token sequence divergence detected!"
        assert max_drift < 1e-6, f"Empirical max drift was {max_drift:.2e}, expected < 1e-6"

    def test_batched_kv_cache_fast_path_identical_to_full_causal_pass(
        self,
        adversarial_cli_model: tuple[HeliosForCausalLM, HeliosConfig],
    ) -> None:
        """Verify batched (batch=4) KV cache fast-path equivalence to full causal pass."""
        model, cfg = adversarial_cli_model
        batch_size = 4
        prompt_len = 6
        gen_len = 10
        head_dim = cfg.d_model // cfg.n_heads

        torch.manual_seed(777)
        prompt_tokens = torch.randint(0, cfg.vocab_size, (batch_size, prompt_len))

        # Method A: Full causal
        seq_full = prompt_tokens.clone()
        logits_full: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(gen_len):
                out = model.forward(seq_full, start_pos=0, kv_caches=None)
                logits_full.append(out["logits"][:, -1, :])
                next_tok = torch.argmax(out["logits"][:, -1, :], dim=-1, keepdim=True)
                seq_full = torch.cat([seq_full, next_tok], dim=-1)

        # Method B: Batched KV Cache
        kv_caches = [
            KVCache(
                max_batch_size=batch_size,
                max_seq_len=prompt_len + gen_len + 16,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=head_dim,
                dtype=torch.float32,
            )
            for _ in range(cfg.n_layers)
        ]

        seq_kv = prompt_tokens.clone()
        logits_kv: list[torch.Tensor] = []
        with torch.no_grad():
            out = model.forward(prompt_tokens, start_pos=0, kv_caches=kv_caches)
            logits_kv.append(out["logits"][:, -1, :])
            cur_tok = torch.argmax(out["logits"][:, -1, :], dim=-1, keepdim=True)
            seq_kv = torch.cat([seq_kv, cur_tok], dim=-1)

            for step in range(1, gen_len):
                cur_pos = prompt_len + step - 1
                out = model.forward(cur_tok, start_pos=cur_pos, kv_caches=kv_caches)
                logits_kv.append(out["logits"][:, 0, :])
                cur_tok = torch.argmax(out["logits"][:, 0, :], dim=-1, keepdim=True)
                seq_kv = torch.cat([seq_kv, cur_tok], dim=-1)

        for step in range(gen_len):
            diff = (logits_full[step] - logits_kv[step]).abs().max().item()
            assert diff < 1e-5, f"Batched step {step} drift {diff:.2e} >= 1e-5"

        assert (seq_full == seq_kv).all().item(), "Batched token divergence detected!"


# ==============================================================================
# 2. CLI Generation Boundary Extremes & Invalid Parameters
# ==============================================================================
@pytest.mark.tier5
class TestCLIGenerateBoundaries:
    """Stress-tests boundary and anomalous inputs to live text generation."""

    def test_empty_and_whitespace_prompt_handling(self) -> None:
        """Verify empty and whitespace strings execute without unhandled crashes."""
        model, config = load_model_and_config("nano-15m", device="cpu")
        tokenizer = load_or_create_tokenizer(vocab_size=config.vocab_size)

        # Empty string
        gen_empty = list(stream_generate(model, tokenizer, "", max_new_tokens=4))
        assert len(gen_empty) >= 1

        # Whitespace and newlines
        gen_ws = list(stream_generate(model, tokenizer, "  \n\t  ", max_new_tokens=4))
        assert len(gen_ws) >= 1

        # CLI invoke with whitespace
        res = cli_runner.invoke(
            cli_app,
            [
                "generate",
                "--prompt",
                "   ",
                "--preset",
                "nano-15m",
                "--device",
                "cpu",
                "--no-stream",
                "--max-new-tokens",
                "4",
            ],
        )
        assert res.exit_code == 0
        assert "Helios Generation Telemetry" in res.stdout

    def test_extreme_and_boundary_hyperparameters(self) -> None:
        """Verify temperature, top_k, and max_new_tokens extremes do not crash."""
        model, config = load_model_and_config("nano-15m", device="cpu")
        tokenizer = load_or_create_tokenizer(vocab_size=config.vocab_size)

        test_cases = [
            {"temperature": -1.0, "top_k": 50, "max_new_tokens": 4},  # negative temp -> argmax
            {"temperature": 0.0, "top_k": 0, "max_new_tokens": 4},  # zero temp & top_k 0
            {"temperature": 1.5, "top_k": -10, "max_new_tokens": 4},  # negative top_k disabled
            {"temperature": 0.7, "top_k": 999999, "max_new_tokens": 4},  # huge top_k clamped
            {"temperature": 0.0, "top_k": 50, "max_new_tokens": 0},  # zero max_new_tokens
            {"temperature": 0.0, "top_k": 50, "max_new_tokens": -5},  # negative max_new_tokens
        ]

        for case in test_cases:
            gen = stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt="test prompt",
                temperature=case["temperature"],
                top_k=case["top_k"],
                max_new_tokens=case["max_new_tokens"],
            )
            tokens = list(gen)
            assert isinstance(tokens, list)

    def test_prompt_length_exceeding_max_seq_len_boundary(self) -> None:
        """Verify prompt exceeding max_seq_len is safely truncated without unhandled crash."""
        model, config = load_model_and_config("nano-15m", device="cpu")
        tokenizer = load_or_create_tokenizer(vocab_size=config.vocab_size)

        # Huge prompt exceeding max_seq_len (2048)
        huge_prompt = "helios " * (config.max_seq_len + 50)
        tokens = list(stream_generate(model, tokenizer, huge_prompt, max_new_tokens=4))
        assert len(tokens) >= 1

        # CLI runner succeeds with exit code 0 when prompt is safely truncated
        res = cli_runner.invoke(
            cli_app,
            [
                "generate",
                "--prompt",
                huge_prompt,
                "--preset",
                "nano-15m",
                "--device",
                "cpu",
                "--no-stream",
                "--max-new-tokens",
                "4",
            ],
        )
        assert res.exit_code == 0
        assert "Helios Generation Telemetry" in res.stdout


# ==============================================================================
# 3. Multi-Turn Chat Memory Saturation & Session Lifecycle
# ==============================================================================
@pytest.mark.tier5
class TestMultiTurnChatSaturation:
    """Stress-tests multi-turn chat sessions and memory persistence."""

    def test_multi_turn_chat_session_accumulation_and_reset(self, tmp_path: Path) -> None:
        """Verify chat session accumulates turns and flushes cleanly on reset."""
        model, config = load_model_and_config("nano-15m", device="cpu")
        tokenizer = load_or_create_tokenizer(vocab_size=config.vocab_size)
        db_path = tmp_path / "chat_stress.db"

        session = ChatSession(
            model=model,
            tokenizer=tokenizer,
            session_id="stress_session",
            db_path=db_path,
            device=torch.device("cpu"),
        )

        # Run 8 normal conversation turns
        for i in range(8):
            reply = session.handle_turn(
                user_input=f"Turn {i}: Question about architecture component {i}",
                max_new_tokens=4,
                stream=False,
            )
            assert isinstance(reply, str)

        assert session.total_turns == 8
        assert session.total_generated_tokens > 0

        # Verify persistent episodic SQLite store
        turns = session.hma.episodic.get_session_turns("stress_session")
        assert len(turns) == 16  # 8 user turns + 8 assistant replies

        # Verify reset / clear clears active working memory
        session.clear()
        assert session.total_turns == 0
        working_buf = session.hma.working.get_or_create("stress_session")
        assert working_buf.turn_count == 0


# ==============================================================================
# 4. CLI Eval & Export Boundaries
# ==============================================================================
@pytest.mark.tier5
class TestCLIEvalAndExportBoundaries:
    """Stress-tests eval and export CLI commands under boundary conditions."""

    def test_cli_eval_minimal_lengths(self) -> None:
        """Verify helios eval executes with minimal sequence lengths."""
        res = cli_runner.invoke(
            cli_app,
            [
                "eval",
                "--preset",
                "nano-15m",
                "--device",
                "cpu",
                "--prompt-len",
                "8",
                "--gen-len",
                "8",
            ],
        )
        assert res.exit_code == 0
        assert "Helios Performance & Perplexity Report" in res.stdout
        assert "Prefill Latency" in res.stdout
        assert "Decode Latency" in res.stdout

    def test_disk_mirror_manager_boundary_audit(self, tmp_path: Path) -> None:
        """Verify DiskMirrorManager handles accessible temporary destinations cleanly."""
        dest = tmp_path / "mirror_target"
        dest.mkdir(parents=True, exist_ok=True)
        mgr = DiskMirrorManager(disk_root=dest)

        audit = mgr.audit_sync()
        assert audit["disk_accessible"] is True
        assert audit["disk_root"] == str(dest)
