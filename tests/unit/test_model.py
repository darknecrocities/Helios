"""Comprehensive Unit Test Suite for Helios Decoder-Only Transformer Architecture.

Validates forward shapes, causal masking invariance, RoPE translation invariance,
KV Cache numerical equivalence, gradient backpropagation, parameter inspector precision,
and hardware device compatibility.
"""

from __future__ import annotations

import pytest
import torch

from helios.model.attention import Attention, KVCache, repeat_kv
from helios.model.config import HeliosConfig
from helios.model.feed_forward import FeedForward, calculate_d_ffn
from helios.model.normalization import RMSNorm
from helios.model.parameter_inspector import (
    calculate_theoretical_parameters,
    count_parameters,
    inspect_parameters,
)
from helios.model.rotary_embedding import RotaryEmbedding, apply_rotary_emb
from helios.model.transformer import HeliosForCausalLM
from helios.model.transformer_block import HeliosBlock


@pytest.fixture
def small_config() -> HeliosConfig:
    """Fast, lightweight configuration for unit tests."""
    return HeliosConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ffn=128,
        max_seq_len=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )


@pytest.fixture
def small_model(small_config: HeliosConfig) -> HeliosForCausalLM:
    """Instantiate small model with deterministic weights."""
    torch.manual_seed(42)
    return HeliosForCausalLM(small_config)


@pytest.mark.unit
class TestHeliosConfig:
    """Tests for HeliosConfig validation, presets, and serialization."""

    def test_validation_invariants(self):
        with pytest.raises(ValueError, match="divisible by n_heads"):
            HeliosConfig(d_model=65, n_heads=4)
        with pytest.raises(ValueError, match="divisible by n_kv_heads"):
            HeliosConfig(n_heads=6, n_kv_heads=4)
        with pytest.raises(ValueError, match="must be positive"):
            HeliosConfig(vocab_size=0)
        with pytest.raises(ValueError, match="dropout must be in"):
            HeliosConfig(dropout=1.0)

    def test_aliases_and_serialization(self, tmp_path):
        cfg = HeliosConfig(d_ffn=512, rope_theta=5000.0, tie_word_embeddings=False)
        assert cfg.d_ff == 512
        assert cfg.rope_base == 5000.0
        assert not cfg.tie_embeddings

        saved_path = cfg.save_pretrained(tmp_path)
        loaded = HeliosConfig.from_pretrained(saved_path)
        assert loaded == cfg

    def test_preset_parameter_bounds(self):
        presets = [
            HeliosConfig.from_preset("helios_nano_15m"),
            HeliosConfig.from_preset("nano_25m"),
            HeliosConfig.from_preset("nano_35m"),
        ]
        for p in presets:
            params = p.count_parameters()["total_parameters"]
            assert 10_000_000 <= params <= 40_000_000

        with pytest.raises(KeyError, match="Unknown preset"):
            HeliosConfig.from_preset("unknown_preset")

        # Test alias resolution in from_dict
        d = {"d_ff": 512, "rope_base": 8000.0, "tie_embeddings": False}
        cfg_d = HeliosConfig.from_dict(d)
        assert cfg_d.d_ffn == 512 and cfg_d.rope_theta == 8000.0 and not cfg_d.tie_word_embeddings


@pytest.mark.unit
class TestRMSNorm:
    """Tests for RMSNorm numerical precision invariant and scaling."""

    def test_fp16_overflow_protection(self):
        norm = RMSNorm(dim=64).half()
        # 300.0^2 = 90000 > 65504 (max float16)
        x = torch.full((2, 8, 64), 300.0, dtype=torch.float16)
        out = norm(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert out.dtype == torch.float16

    def test_scale_invariance(self):
        norm = RMSNorm(dim=32)
        x = torch.randn(2, 4, 32)
        out1 = norm(x)
        out2 = norm(x * 5.5)
        assert torch.allclose(out1, out2, atol=1e-5)


@pytest.mark.unit
class TestRotaryEmbedding:
    """Tests for RoPE relative translation invariance and caching."""

    def test_relative_dot_product_invariance(self):
        head_dim = 16
        rope = RotaryEmbedding(dim=head_dim, max_seq_len=64)
        cos, sin = rope(torch.empty(0), seq_len=64)

        torch.manual_seed(123)
        q = torch.randn(1, 1, 1, head_dim)
        k = torch.randn(1, 1, 1, head_dim)
        m, n, shift = 7, 3, 15

        cos_m = cos[:, :, m : m + 1, :]
        sin_m = sin[:, :, m : m + 1, :]
        cos_n = cos[:, :, n : n + 1, :]
        sin_n = sin[:, :, n : n + 1, :]

        q_m = apply_rotary_emb(q, cos_m, sin_m)
        k_n = apply_rotary_emb(k, cos_n, sin_n)
        dot_unshifted = (q_m * k_n).sum()

        cos_ms = cos[:, :, m + shift : m + shift + 1, :]
        sin_ms = sin[:, :, m + shift : m + shift + 1, :]
        cos_ns = cos[:, :, n + shift : n + shift + 1, :]
        sin_ns = sin[:, :, n + shift : n + shift + 1, :]

        q_ms = apply_rotary_emb(q, cos_ms, sin_ms)
        k_ns = apply_rotary_emb(k, cos_ns, sin_ns)
        dot_shifted = (q_ms * k_ns).sum()

        assert torch.allclose(dot_unshifted, dot_shifted, atol=1e-5)

    def test_dynamic_cache_extension(self):
        rope = RotaryEmbedding(dim=16, max_seq_len=8)
        assert rope.max_seq_len_cached == 8
        dummy = torch.randn(1, 2, 32, 16)
        cos, sin = rope(dummy, seq_len=32)
        assert rope.max_seq_len_cached >= 32
        assert cos.shape == (1, 1, 32, 16)


@pytest.mark.unit
class TestAttentionAndKVCache:
    """Tests for GQA head expansion, pre-allocated KVCache, and Attention."""

    def test_repeat_kv(self):
        x = torch.randn(2, 2, 8, 16)
        out = repeat_kv(x, n_rep=3)
        assert out.shape == (2, 6, 8, 16)
        assert repeat_kv(x, n_rep=1) is x

    def test_kv_cache_slice_mutation(self):
        cache = KVCache(max_batch_size=2, n_kv_heads=2, max_seq_len=16, head_dim=16)
        k1 = torch.randn(2, 2, 4, 16)
        v1 = torch.randn(2, 2, 4, 16)
        k_out, v_out = cache.update(start_pos=0, k_val=k1, v_val=v1)
        assert k_out.shape == (2, 2, 4, 16)
        assert cache.cur_len == 4

        # Incremental token step
        k2 = torch.randn(2, 2, 1, 16)
        v2 = torch.randn(2, 2, 1, 16)
        k_out2, v_out2 = cache.update(start_pos=4, k_val=k2, v_val=v2)
        assert k_out2.shape == (2, 2, 5, 16)
        assert cache.cur_len == 5
        assert cache.memory_footprint_mb() > 0

        cache.reset()
        assert cache.cur_len == 0
        cache.to(device="cpu", dtype=torch.float32)

    def test_attention_forward(self, small_config: HeliosConfig):
        attn = Attention(config=small_config)
        x = torch.randn(2, 8, small_config.d_model)
        out = attn(x)
        assert out.shape == (2, 8, small_config.d_model)


@pytest.mark.unit
class TestFeedForward:
    """Tests for SwiGLU activation and tensor-core alignment."""

    def test_swiglu_parity_and_alignment(self):
        d_ffn = calculate_d_ffn(d_model=288, multiple_of=64)
        assert d_ffn == 768
        assert d_ffn % 64 == 0

        ffn = FeedForward(d_model=64, d_ffn=128)
        x = torch.randn(2, 8, 64)
        out = ffn(x)
        assert out.shape == (2, 8, 64)


@pytest.mark.unit
class TestTransformerBlock:
    """Tests for HeliosBlock decoder layer."""

    def test_block_forward(self, small_config: HeliosConfig):
        block = HeliosBlock(small_config)
        x = torch.randn(2, 8, small_config.d_model)
        out = block(x)
        assert out.shape == (2, 8, small_config.d_model)

    def test_block_with_cache(self, small_config: HeliosConfig):
        block = HeliosBlock(small_config)
        head_dim = small_config.d_model // small_config.n_heads
        cache = KVCache(
            max_batch_size=2,
            n_kv_heads=small_config.n_kv_heads,
            max_seq_len=16,
            head_dim=head_dim,
        )
        x = torch.randn(2, 4, small_config.d_model)
        out = block(x, start_pos=0, kv_cache=cache)
        assert out.shape == (2, 4, small_config.d_model)
        assert cache.cur_len == 4


@pytest.mark.unit
class TestHeliosForCausalLM:
    """Tests for full causal language model stack."""

    @pytest.mark.parametrize(
        ("batch_size", "seq_len"),
        [(1, 1), (1, 16), (2, 32), (4, 7)],
    )
    def test_forward_tensor_shapes(
        self, small_model: HeliosForCausalLM, batch_size: int, seq_len: int
    ):
        input_ids = torch.randint(0, small_model.config.vocab_size, (batch_size, seq_len))
        out = small_model(input_ids)
        assert "logits" in out
        assert out["logits"].shape == (batch_size, seq_len, small_model.config.vocab_size)
        assert out["loss"] is None

    def test_forward_with_labels_computes_loss(self, small_model: HeliosForCausalLM):
        input_ids = torch.randint(0, small_model.config.vocab_size, (2, 16))
        labels = input_ids.clone()
        out = small_model(input_ids, labels=labels)
        assert out["loss"] is not None
        assert out["loss"].ndim == 0
        assert out["loss"].item() > 0.0
        assert torch.isfinite(out["loss"])

    def test_weight_tying_aliasing(self):
        cfg_tied = HeliosConfig(
            vocab_size=64,
            d_model=32,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ffn=64,
            tie_word_embeddings=True,
        )
        model_tied = HeliosForCausalLM(cfg_tied)
        assert model_tied.lm_head.weight.data_ptr() == model_tied.embed_tokens.weight.data_ptr()

        cfg_untied = HeliosConfig(
            vocab_size=64,
            d_model=32,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ffn=64,
            tie_word_embeddings=False,
        )
        model_untied = HeliosForCausalLM(cfg_untied)
        assert model_untied.lm_head.weight.data_ptr() != model_untied.embed_tokens.weight.data_ptr()

    def test_causal_masking_invariance(self, small_model: HeliosForCausalLM):
        small_model.eval()
        seq1 = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
        seq2 = torch.tensor([[10, 20, 30, 40, 99, 60, 70, 80]])

        with torch.no_grad():
            out1 = small_model(seq1)["logits"]
            out2 = small_model(seq2)["logits"]

        past_diff = (out1[:, :4, :] - out2[:, :4, :]).abs().max().item()
        assert past_diff == 0.0, f"Future token leaked into past logits! Diff: {past_diff}"

        future_diff = (out1[:, 4:, :] - out2[:, 4:, :]).abs().max().item()
        assert future_diff > 0.0, "Modified token failed to alter downstream representations!"

    def test_step_by_step_decode_matches_full_forward(self, small_model: HeliosForCausalLM):
        small_model.eval()
        torch.manual_seed(42)
        total_len, prompt_len = 10, 4
        tokens = torch.randint(0, small_model.config.vocab_size, (1, total_len))

        with torch.no_grad():
            full_logits = small_model(tokens)["logits"]

        head_dim = small_model.config.d_model // small_model.config.n_heads
        kv_caches = [
            KVCache(
                max_batch_size=1,
                n_kv_heads=small_model.config.n_kv_heads,
                max_seq_len=total_len,
                head_dim=head_dim,
            )
            for _ in range(small_model.config.n_layers)
        ]

        with torch.no_grad():
            prefill_out = small_model(tokens[:, :prompt_len], start_pos=0, kv_caches=kv_caches)
            assert torch.allclose(prefill_out["logits"], full_logits[:, :prompt_len, :], atol=1e-5)

            for t in range(prompt_len, total_len):
                step_token = tokens[:, t : t + 1]
                step_out = small_model(step_token, start_pos=t, kv_caches=kv_caches)
                assert torch.allclose(
                    step_out["logits"][:, 0, :], full_logits[:, t, :], atol=1e-5
                ), f"Mismatch at step {t}!"

    def test_generate_greedy_reproducibility(self, small_model: HeliosForCausalLM):
        prompt = torch.tensor([[10, 20, 30]])
        gen1 = small_model.generate(prompt, max_new_tokens=5, temperature=0.0)
        gen2 = small_model.generate(prompt, max_new_tokens=5, temperature=0.0)
        assert torch.equal(gen1, gen2)
        assert gen1.shape == (1, 8)
        # Test max_new_tokens <= 0
        assert torch.equal(small_model.generate(prompt, max_new_tokens=0), prompt)

        # Test sampling with temperature and top_k
        sampled = small_model.generate(prompt, max_new_tokens=3, temperature=0.8, top_k=5)
        assert sampled.shape == (1, 6)

        # Test early stopping with eos_token_id
        eos_id = sampled[0, 3].item()
        stopped = small_model.generate(
            prompt, max_new_tokens=10, temperature=0.0, eos_token_id=eos_id
        )
        assert stopped.shape[-1] <= 13

        # Test embedding accessors
        assert small_model.get_input_embeddings() is small_model.embed_tokens
        assert small_model.get_output_embeddings() is small_model.lm_head
        small_model.set_input_embeddings(small_model.embed_tokens)

    def test_all_parameters_receive_gradients(self, small_model: HeliosForCausalLM):
        small_model.train()
        input_ids = torch.randint(0, small_model.config.vocab_size, (2, 8))
        labels = input_ids.clone()

        out = small_model(input_ids, labels=labels)
        loss = out["loss"]
        loss.backward()

        for name, param in small_model.named_parameters():
            assert param.grad is not None, f"Parameter {name} did not receive a gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients!"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients!"
            assert param.grad.abs().sum().item() > 0.0, f"Parameter {name} has zero gradients!"


@pytest.mark.unit
class TestParameterInspector:
    """Tests for parameter counting precision and report generation."""

    def test_theoretical_parameter_match(self, small_config: HeliosConfig):
        model = HeliosForCausalLM(small_config)
        report = inspect_parameters(model)
        theoretical = calculate_theoretical_parameters(small_config)

        assert report.total_parameters == theoretical["total_parameters"]
        assert report.non_embedding_parameters == theoretical["non_embedding_parameters"]
        assert report.embedding_parameters == theoretical["embedding_parameters"]

        counts = count_parameters(model)
        assert counts["total_parameters"] == report.total_parameters
        assert counts["trainable_parameters"] == report.trainable_parameters

        summary = report.summary()
        assert "Helios Parameter Inspection Report" in summary
        assert report.to_dict()["total_parameters"] == report.total_parameters

    @pytest.mark.parametrize(
        (
            "preset_name",
            "d_model",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "d_ffn",
            "expected_total",
        ),
        [
            ("nano_15m", 288, 6, 6, 2, 768, 14_528_160),
            ("nano_25m", 384, 8, 6, 2, 1024, 24_877_440),
            ("nano_35m", 512, 8, 8, 2, 1376, 38_543_872),
        ],
    )
    def test_nano_presets_bounds(
        self,
        preset_name,
        d_model,
        n_layers,
        n_heads,
        n_kv_heads,
        d_ffn,
        expected_total,
    ):
        config = HeliosConfig(
            vocab_size=32000,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ffn=d_ffn,
            tie_word_embeddings=True,
        )
        theoretical = calculate_theoretical_parameters(config)
        assert theoretical["total_parameters"] == expected_total
        assert 10_000_000 <= theoretical["total_parameters"] <= 40_000_000


@pytest.mark.unit
class TestDeviceCompatibility:
    """Tests execution across available hardware targets."""

    def test_cpu_execution(self, small_model: HeliosForCausalLM):
        small_model.to("cpu")
        tokens = torch.randint(0, small_model.config.vocab_size, (1, 4), device="cpu")
        out = small_model(tokens)
        assert out["logits"].device.type == "cpu"

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_execution(self, small_config: HeliosConfig):
        device = torch.device("mps")
        model = HeliosForCausalLM(small_config).to(device)
        tokens = torch.randint(0, small_config.vocab_size, (1, 4), device=device)
        out = model(tokens)
        assert out["logits"].device.type == "mps"

        gen = model.generate(tokens, max_new_tokens=3, temperature=0.0)
        assert gen.device.type == "mps"
