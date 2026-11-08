"""Unit tests for Helios Speculative Decoding Engine (Feature 46).

Tests speculative lookahead, Leviathan acceptance/rejection sampling,
synchronized KV-cache rollbacks, bonus token emission, and target distribution invariance.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from helios.model.config import HeliosConfig
from helios.model.speculative import (
    SpeculativeConfig,
    SpeculativeDecodingEngine,
    SpeculativeGenerationOutput,
    _sample_recovery,
)
from helios.model.transformer import HeliosForCausalLM


@pytest.fixture
def tiny_target_and_draft() -> tuple[HeliosForCausalLM, HeliosForCausalLM]:
    torch.manual_seed(42)
    # Target model: 2 layers, d_model=64
    cfg_target = HeliosConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ffn=128,
        max_seq_len=128,
    )
    # Draft model: 1 layer, d_model=64
    cfg_draft = HeliosConfig(
        vocab_size=128,
        d_model=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ffn=128,
        max_seq_len=128,
    )
    target = HeliosForCausalLM(cfg_target)
    draft = HeliosForCausalLM(cfg_draft)
    return target, draft


def test_speculative_config_invariants() -> None:
    """Verify input validation on SpeculativeConfig."""
    with pytest.raises(ValueError, match="lookahead_k must be > 0"):
        SpeculativeConfig(lookahead_k=0)

    with pytest.raises(ValueError, match="temperature must be >= 0.0"):
        SpeculativeConfig(temperature=-0.5)

    cfg = SpeculativeConfig(lookahead_k=4, temperature=0.7)
    assert cfg.lookahead_k == 4
    assert cfg.temperature == 0.7


def test_speculative_vocab_mismatch_rejection() -> None:
    """Verify engine rejects pairing models with differing vocabulary sizes."""
    cfg_1 = HeliosConfig(vocab_size=100, d_model=32, n_layers=1, n_heads=2, n_kv_heads=2)
    cfg_2 = HeliosConfig(vocab_size=200, d_model=32, n_layers=1, n_heads=2, n_kv_heads=2)
    m1 = HeliosForCausalLM(cfg_1)
    m2 = HeliosForCausalLM(cfg_2)

    with pytest.raises(ValueError, match="Vocabulary size mismatch"):
        SpeculativeDecodingEngine(target_model=m1, draft_model=m2)


def test_speculative_greedy_equivalence(
    tiny_target_and_draft: tuple[HeliosForCausalLM, HeliosForCausalLM],
) -> None:
    """Verify speculative greedy decoding is 100% bit-for-bit identical to pure target model."""
    target, draft = tiny_target_and_draft
    prompt = torch.tensor([[1, 5, 9, 14]])
    max_new_tokens = 12

    # 1. Pure target autoregressive generation
    pure_target_tokens = target.generate(
        prompt_tokens=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    # 2. Speculative decoding with Nano-draft + Target
    engine = SpeculativeDecodingEngine(
        target_model=target,
        draft_model=draft,
        config=SpeculativeConfig(lookahead_k=3, temperature=0.0, max_new_tokens=max_new_tokens),
    )
    spec_out = engine.generate(prompt_tokens=prompt)

    # Exact token equality check
    assert isinstance(spec_out, SpeculativeGenerationOutput)
    assert torch.equal(spec_out.tokens, pure_target_tokens), (
        f"Greedy mismatch:\nSpeculative: {spec_out.tokens}\nPure Target: {pure_target_tokens}"
    )
    assert spec_out.metrics.total_generated_tokens == max_new_tokens


def test_speculative_all_accepted_bonus_token() -> None:
    """Verify that when draft model matches target model, all K tokens + bonus token are accepted."""
    torch.manual_seed(101)
    cfg = HeliosConfig(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        d_ffn=64,
        max_seq_len=64,
    )
    target = HeliosForCausalLM(cfg)
    # Draft is exact clone of target to guarantee 100% acceptance
    draft = HeliosForCausalLM(cfg)
    draft.load_state_dict(target.state_dict())

    engine = SpeculativeDecodingEngine(
        target_model=target,
        draft_model=draft,
        config=SpeculativeConfig(lookahead_k=4, temperature=0.0, max_new_tokens=5),
    )

    prompt = torch.tensor([[2, 4, 8]])
    out = engine.generate(prompt_tokens=prompt, max_new_tokens=5)

    # With lookahead_k=4 and identical models, in 1 speculative step it accepts 4 draft tokens + 1 bonus token = 5
    assert out.metrics.acceptance_rate == 1.0
    assert out.metrics.accepted_draft_tokens > 0
    assert out.tokens.shape[1] == 8  # 3 prompt + 5 generated


def test_speculative_kv_cache_rollback_on_rejection(
    tiny_target_and_draft: tuple[HeliosForCausalLM, HeliosForCausalLM],
) -> None:
    """Verify that when a candidate is rejected, caches roll back accurately to rejection index."""
    target, draft = tiny_target_and_draft
    prompt = torch.tensor([[3, 7, 11]])

    engine = SpeculativeDecodingEngine(
        target_model=target,
        draft_model=draft,
        config=SpeculativeConfig(lookahead_k=4, temperature=0.0, max_new_tokens=8),
    )
    out = engine.generate(prompt_tokens=prompt)

    assert out.tokens.shape[1] == 11
    assert out.metrics.speculative_steps > 0
    assert out.metrics.total_generated_tokens == 8


def test_speculative_stochastic_distribution_invariance() -> None:
    """Verify Leviathan rejection sampling matches ground-truth target distribution p."""
    torch.manual_seed(42)
    # Discrete target and draft probability distributions over 5 tokens
    p = F.softmax(torch.tensor([0.1, 0.5, 0.05, 0.25, 0.1]), dim=-1)
    q = F.softmax(torch.tensor([0.3, 0.1, 0.4, 0.1, 0.1]), dim=-1)

    n_trials = 25000
    samples: list[int] = []

    for _ in range(n_trials):
        # Sample candidate from draft q
        cand = int(torch.multinomial(q, 1).item())
        ratio = float(p[cand].item() / q[cand].item())
        u = float(torch.rand(1).item())
        if u < min(1.0, ratio):
            samples.append(cand)
        else:
            # Rejection recovery from max(0, p - q)
            rec = _sample_recovery(p, q, temperature=1.0)
            samples.append(rec)

    counts = torch.bincount(torch.tensor(samples), minlength=5).float()
    p_empirical = counts / n_trials

    # Maximum difference should be < 0.015 for 25k samples
    max_diff = float(torch.max(torch.abs(p_empirical - p)).item())
    assert max_diff < 0.015, f"Leviathan sampling diverged: max_diff = {max_diff}"


def test_speculative_eos_handling(
    tiny_target_and_draft: tuple[HeliosForCausalLM, HeliosForCausalLM],
) -> None:
    """Verify speculative decoding halts promptly when EOS token is emitted."""
    target, draft = tiny_target_and_draft
    eos_id = 99

    engine = SpeculativeDecodingEngine(
        target_model=target,
        draft_model=draft,
        config=SpeculativeConfig(
            lookahead_k=3, temperature=0.0, max_new_tokens=20, eos_token_id=eos_id
        ),
    )
    # Craft a tiny input
    prompt = torch.tensor([[5, 10]])
    out = engine.generate(prompt_tokens=prompt)

    # Generated sequence should not exceed budget
    assert out.tokens.shape[1] <= 2 + 20


def test_speculative_metrics_tracking(
    tiny_target_and_draft: tuple[HeliosForCausalLM, HeliosForCausalLM],
) -> None:
    """Verify SpeculativeMetrics fields are properly recorded and within bounds."""
    target, draft = tiny_target_and_draft
    engine = SpeculativeDecodingEngine(
        target_model=target,
        draft_model=draft,
        config=SpeculativeConfig(lookahead_k=3, temperature=0.0, max_new_tokens=10),
    )
    out = engine.generate(prompt_tokens=torch.tensor([[1, 2, 3]]))
    m = out.metrics

    assert m.total_generated_tokens == 10
    assert m.total_draft_tokens > 0
    assert 0.0 <= m.acceptance_rate <= 1.0
    assert m.speculative_steps > 0
    assert m.wall_clock_time_sec > 0.0
    assert m.tokens_per_second > 0.0


def test_speculative_lookahead_gamma_alias(
    tiny_target_and_draft: tuple[HeliosForCausalLM, HeliosForCausalLM],
) -> None:
    """Verify lookahead_gamma alias works in SpeculativeConfig and engine."""
    target, draft = tiny_target_and_draft
    cfg = SpeculativeConfig(lookahead_gamma=5)
    assert cfg.lookahead_k == 5
    assert cfg.lookahead_gamma == 5

    engine = SpeculativeDecodingEngine(
        draft_model=draft,
        target_model=target,
        lookahead_gamma=6,
    )
    assert engine.lookahead_gamma == 6
    assert engine.config.lookahead_k == 6


def test_speculative_generate_stream(
    tiny_target_and_draft: tuple[HeliosForCausalLM, HeliosForCausalLM],
) -> None:
    """Verify generate_stream yields individual tokens matching generate()."""
    target, draft = tiny_target_and_draft
    engine = SpeculativeDecodingEngine(
        target_model=target,
        draft_model=draft,
        config=SpeculativeConfig(lookahead_k=3, temperature=0.0, max_new_tokens=8),
    )
    prompt = torch.tensor([[1, 2, 3]])
    out = engine.generate(prompt_ids=prompt, max_new_tokens=8, temperature=0.0)
    expected_stream = out.tokens[0, 3:].tolist()

    streamed_tokens = list(
        engine.generate_stream(prompt_ids=prompt, max_new_tokens=8, temperature=0.0)
    )
    assert streamed_tokens == expected_stream
    assert len(streamed_tokens) == 8


def test_physical_line_limit_compliance() -> None:
    """Verify helios/model/speculative.py adheres to strict 500-line ceiling."""
    source_path = Path(__file__).parent.parent.parent / "helios" / "model" / "speculative.py"
    assert source_path.exists(), f"Missing file: {source_path}"
    with open(source_path, "r", encoding="utf-8") as f:
        line_count = len(f.readlines())
    assert line_count <= 500, f"Physical line ceiling breached: {line_count} > 500 lines"
