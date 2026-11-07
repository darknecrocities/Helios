"""Tier-5 Adversarial Verification Suite for Helios Transformer Architecture.

Empirically stress-tests core mathematical invariants:
1. Causal Masking Invariance (bitwise identical past representations, max diff == 0.0).
2. Rotary Position Embedding (RoPE) Invariance (relative translation invariance < 1e-6).
3. Incremental KV Cache Numerical Equivalence (cached decode vs full pass < 1e-5).
4. Float16 / BFloat16 Numerical Stability (RMSNorm overflow safety at x=350.0+).
5. Parameter Inspector Precision and Roadmap Nano Sizing Gates.
"""

from __future__ import annotations

import pytest
import torch

from helios.model.attention import KVCache
from helios.model.config import HeliosConfig
from helios.model.normalization import RMSNorm
from helios.model.parameter_inspector import (
    calculate_theoretical_parameters,
    count_parameters,
    inspect_parameters,
)
from helios.model.rotary_embedding import RotaryEmbedding, apply_rotary_emb
from helios.model.transformer import HeliosForCausalLM


@pytest.fixture
def adversarial_config() -> HeliosConfig:
    """Configured model with GQA 3:1 ratio and multi-layer depth for stress-testing."""
    return HeliosConfig(
        vocab_size=256,
        d_model=96,
        n_layers=3,
        n_heads=6,
        n_kv_heads=2,
        d_ffn=256,
        max_seq_len=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )


@pytest.fixture
def adversarial_model(adversarial_config: HeliosConfig) -> HeliosForCausalLM:
    """Deterministic model instance for adversarial challenge tests."""
    torch.manual_seed(999)
    return HeliosForCausalLM(adversarial_config).eval()


@pytest.mark.tier5
class TestCausalMaskingInvariance:
    """Empirical proofs that future tokens exert strictly ZERO influence on past states."""

    @pytest.mark.parametrize("perturb_k", [0, 3, 7, 14, 20])
    def test_bitwise_causal_isolation(
        self, adversarial_model: HeliosForCausalLM, perturb_k: int
    ) -> None:
        """Perturb token at k+1 and assert 100% bitwise identity at all t in [0, k]."""
        seq_len = 24
        torch.manual_seed(42 + perturb_k)
        seq1 = torch.randint(0, adversarial_model.config.vocab_size, (2, seq_len))
        seq2 = seq1.clone()

        # Adversarially perturb token at index k+1
        seq2[:, perturb_k + 1] = (seq2[:, perturb_k + 1] + 17) % adversarial_model.config.vocab_size

        with torch.no_grad():
            out1 = adversarial_model(seq1)["logits"]
            out2 = adversarial_model(seq2)["logits"]

        # Past representations [0..k] must be bitwise identical (max diff == 0.0)
        past_diff = (out1[:, : perturb_k + 1, :] - out2[:, : perturb_k + 1, :]).abs().max().item()
        assert past_diff == 0.0, (
            f"Causal leak detected! Past logits differed by {past_diff} at k={perturb_k}"
        )
        assert torch.equal(out1[:, : perturb_k + 1, :], out2[:, : perturb_k + 1, :]), (
            f"Bitwise divergence in past logits at k={perturb_k}"
        )

        # Future representations [k+1..seq_len-1] must be altered
        future_diff = (out1[:, perturb_k + 1 :, :] - out2[:, perturb_k + 1 :, :]).abs().max().item()
        assert future_diff > 0.0, "Perturbation had zero effect downstream!"

    def test_layerwise_hidden_state_isolation(self, adversarial_model: HeliosForCausalLM) -> None:
        """Verify past hidden states remain bitwise identical at every intermediate layer."""
        seq_len = 16
        k = 6
        seq1 = torch.randint(0, adversarial_model.config.vocab_size, (1, seq_len))
        seq2 = seq1.clone()
        seq2[:, k + 1] = (seq2[:, k + 1] + 23) % adversarial_model.config.vocab_size

        with torch.no_grad():
            h1 = adversarial_model.embed_tokens(seq1)
            h2 = adversarial_model.embed_tokens(seq2)
            assert torch.equal(h1[:, : k + 1, :], h2[:, : k + 1, :])

            for idx, block in enumerate(adversarial_model.layers):
                h1 = block(h1)
                h2 = block(h2)
                diff = (h1[:, : k + 1, :] - h2[:, : k + 1, :]).abs().max().item()
                assert diff == 0.0, f"Layer {idx} leaked information backwards: diff={diff}"
                assert torch.equal(h1[:, : k + 1, :], h2[:, : k + 1, :]), (
                    f"Layer {idx} past hidden states were not bitwise identical!"
                )


@pytest.mark.tier5
class TestRoPETranslationInvariance:
    """Stress tests on relative position translation invariance for Rotary Embeddings."""

    @pytest.mark.parametrize("head_dim", [16, 32, 64])
    @pytest.mark.parametrize("shift", [1, 7, 33, 128])
    def test_rope_scaled_attention_invariance(self, head_dim: int, shift: int) -> None:
        """Dot product <R_m q, R_n k> == <R_{m+s} q, R_{n+s} k> within FP32 machine epsilon."""
        max_seq = 512
        rope = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq)
        cos, sin = rope(torch.empty(0), seq_len=max_seq)

        torch.manual_seed(2026 + head_dim + shift)
        # Scaled attention query and key representations
        q = torch.randn(1, 1, 1, head_dim) / (head_dim**0.5)
        k = torch.randn(1, 1, 1, head_dim) / (head_dim**0.5)

        for m, n in [(0, 5), (4, 12), (15, 8), (50, 50)]:
            if m + shift >= max_seq or n + shift >= max_seq:
                continue

            cos_m = cos[:, :, m : m + 1, :]
            sin_m = sin[:, :, m : m + 1, :]
            cos_n = cos[:, :, n : n + 1, :]
            sin_n = sin[:, :, n : n + 1, :]

            q_m = apply_rotary_emb(q, cos_m, sin_m)
            k_n = apply_rotary_emb(k, cos_n, sin_n)
            dot_un = (q_m * k_n).sum().item()

            cos_ms = cos[:, :, m + shift : m + shift + 1, :]
            sin_ms = sin[:, :, m + shift : m + shift + 1, :]
            cos_ns = cos[:, :, n + shift : n + shift + 1, :]
            sin_ns = sin[:, :, n + shift : n + shift + 1, :]

            q_ms = apply_rotary_emb(q, cos_ms, sin_ms)
            k_ns = apply_rotary_emb(k, cos_ns, sin_ns)
            dot_sh = (q_ms * k_ns).sum().item()

            abs_diff = abs(dot_un - dot_sh)
            assert abs_diff < 1e-6, (
                f"RoPE invariance violated! m={m}, n={n}, s={shift}, diff={abs_diff:.2e} >= 1e-6"
            )

    def test_rope_subspace_double_precision_invariance(self) -> None:
        """In float64, RoPE translation invariance holds to mathematical machine epsilon."""
        head_dim = 64
        m, n, s = 100, 20, 500
        torch.manual_seed(777)
        q = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)
        k = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))

        def get_rot(pos: int) -> tuple[torch.Tensor, torch.Tensor]:
            t = torch.tensor([pos], dtype=torch.float64)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos(), emb.sin()

        cm, sm = get_rot(m)
        cn, sn = get_rot(n)
        cms, sms = get_rot(m + s)
        cns, sns = get_rot(n + s)

        qm = apply_rotary_emb(q, cm, sm)
        kn = apply_rotary_emb(k, cn, sn)
        qms = apply_rotary_emb(q, cms, sms)
        kns = apply_rotary_emb(k, cns, sns)

        dot_un = (qm * kn).sum().item()
        dot_sh = (qms * kns).sum().item()
        assert abs(dot_un - dot_sh) < 1e-12, f"Float64 RoPE diff: {abs(dot_un - dot_sh)}"


@pytest.mark.tier5
class TestKVCacheNumericalEquivalence:
    """Stress tests verifying incremental decode matches full parallel forward pass."""

    @pytest.mark.parametrize(
        ("prompt_len", "decode_len", "batch_size"),
        [(1, 8, 1), (4, 6, 1), (8, 8, 2), (16, 8, 1)],
    )
    def test_incremental_step_equivalence(
        self,
        adversarial_model: HeliosForCausalLM,
        prompt_len: int,
        decode_len: int,
        batch_size: int,
    ) -> None:
        """Autoregressive step-by-step decode matches parallel pass with max diff < 1e-5."""
        total_len = prompt_len + decode_len
        cfg = adversarial_model.config
        head_dim = cfg.d_model // cfg.n_heads

        torch.manual_seed(1234 + prompt_len)
        tokens = torch.randint(0, cfg.vocab_size, (batch_size, total_len))

        with torch.no_grad():
            full_out = adversarial_model(tokens)
            full_logits = full_out["logits"]

        kv_caches = [
            KVCache(
                max_batch_size=batch_size,
                n_kv_heads=cfg.n_kv_heads,
                max_seq_len=total_len + 16,
                head_dim=head_dim,
                dtype=torch.float32,
            )
            for _ in range(cfg.n_layers)
        ]

        with torch.no_grad():
            # 1. Prefill
            prefill_out = adversarial_model(
                tokens[:, :prompt_len], start_pos=0, kv_caches=kv_caches
            )
            prefill_diff = (
                (prefill_out["logits"] - full_logits[:, :prompt_len, :]).abs().max().item()
            )
            assert prefill_diff < 1e-5, f"Prefill diff {prefill_diff} >= 1e-5"

            # 2. Step-by-step decode
            for step in range(decode_len):
                t = prompt_len + step
                token_step = tokens[:, t : t + 1]
                step_out = adversarial_model(token_step, start_pos=t, kv_caches=kv_caches)
                step_diff = (step_out["logits"][:, 0, :] - full_logits[:, t, :]).abs().max().item()
                assert step_diff < 1e-5, (
                    f"Mismatch at step {step} (t={t}): diff={step_diff:.2e} >= 1e-5"
                )

    def test_kv_cache_capacity_guards(self) -> None:
        """Verify strict buffer boundary protection in KVCache."""
        cache = KVCache(max_batch_size=2, n_kv_heads=2, max_seq_len=8, head_dim=16)
        k = torch.randn(2, 2, 4, 16)
        v = torch.randn(2, 2, 4, 16)
        cache.update(0, k, v)

        # Exceeding seq_len: start_pos=6 + seq_len=4 = 10 > 8
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            cache.update(6, k, v)

        # Exceeding batch size: 4 > 2
        k_large = torch.randn(4, 2, 2, 16)
        v_large = torch.randn(4, 2, 2, 16)
        with pytest.raises(ValueError, match="exceeds pre-allocated cache max_batch_size"):
            cache.update(0, k_large, v_large)


@pytest.mark.tier5
class TestNumericalStabilityRMSNorm:
    """Stress tests on float16 / bfloat16 numerical stability under extreme activation spikes."""

    @pytest.mark.parametrize("spike", [350.0, 500.0, 1000.0, 65000.0])
    def test_fp16_extreme_spike_overflow_safety(self, spike: float) -> None:
        """x^2 >> 65504 must not overflow to inf or NaN due to float32 variance upcasting."""
        norm = RMSNorm(dim=128).half()
        x = torch.full((4, 16, 128), spike, dtype=torch.float16)
        out = norm(x)

        assert not torch.isnan(out).any(), f"NaN generated for spike={spike}"
        assert not torch.isinf(out).any(), f"Inf generated for spike={spike}"
        assert out.dtype == torch.float16
        # Output should be exactly 1.0 within half-precision tolerance
        expected = torch.ones_like(out)
        assert torch.allclose(out, expected, atol=1e-2)

    def test_fp16_isolated_feature_spike(self) -> None:
        """A single feature spiking to 350.0 in a normal distribution must not destabilize."""
        norm = RMSNorm(dim=64).half()
        x = torch.randn(2, 8, 64, dtype=torch.float16)
        x[:, :, 0] = 350.0
        out = norm(x)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert out.dtype == torch.float16

    @pytest.mark.parametrize("spike", [350.0, 1e10, 1e25])
    def test_bfloat16_extreme_spikes(self, spike: float) -> None:
        """BFloat16 extreme activations must produce finite normalized activations."""
        norm = RMSNorm(dim=64).to(torch.bfloat16)
        x = torch.full((2, 4, 64), spike, dtype=torch.bfloat16)
        out = norm(x)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert out.dtype == torch.bfloat16


@pytest.mark.tier5
class TestParameterInspectorAndRoadmapGates:
    """Adversarial stress-testing of model sizing, tying, and parameter accounting."""

    @pytest.mark.parametrize(
        ("preset", "min_p", "max_p", "expected_tied"),
        [
            ("nano_15m", 10_000_000, 20_000_000, 14_528_160),
            ("nano_25m", 20_000_000, 30_000_000, 24_877_440),
            ("nano_35m", 30_000_000, 40_000_000, 38_543_872),
        ],
    )
    def test_nano_preset_exact_sizing(
        self, preset: str, min_p: int, max_p: int, expected_tied: int
    ) -> None:
        """Nano presets must strictly adhere to 10M-40M target and match analytical counts."""
        cfg = HeliosConfig.from_preset(preset)
        model = HeliosForCausalLM(cfg)
        report = inspect_parameters(model)
        theoretical = calculate_theoretical_parameters(cfg)

        assert report.total_parameters == expected_tied
        assert report.total_parameters == theoretical["total_parameters"]
        assert min_p <= report.total_parameters <= max_p
        assert report.preset_match == preset

        # Verify count_parameters contract
        counts = count_parameters(model)
        assert counts["total_parameters"] == expected_tied
        assert counts["trainable_parameters"] == expected_tied

    def test_pointer_deduplication_integrity(self) -> None:
        """Ensure tied embeddings are never double counted."""
        cfg = HeliosConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ffn=256,
            tie_word_embeddings=True,
        )
        model = HeliosForCausalLM(cfg)
        report_tied = inspect_parameters(model)

        # Untie weights manually
        model.lm_head.weight = torch.nn.Parameter(model.embed_tokens.weight.clone())
        report_untied = inspect_parameters(model)

        assert report_tied.tied_embeddings is True
        assert report_untied.tied_embeddings is False
        assert (
            report_untied.total_parameters - report_tied.total_parameters
            == cfg.vocab_size * cfg.d_model
        )
