"""Adversarial Verification Suite for Helios Autoregressive Generation & Gradients.

Empirically stress-tests:
1. Autoregressive `generate()`: greedy determinism, top-k sampling bounds,
   EOS early termination (single and batch with padding), prompt boundary limits.
2. Gradient backpropagation integrity: shifted causal loss, 100% parameter gradient flow
   (zero NaNs, zero Infs, zero dead/zero gradients).
3. Parameter inspector: weight tying buffer deduplication, untied delta precision,
   exact closed-form analytical formulas for Nano presets, multi-precision memory math.
4. Device compatibility: forward, backward, and generation on Apple Silicon MPS and CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from helios.model.config import HeliosConfig
from helios.model.parameter_inspector import (
    calculate_theoretical_parameters,
    count_parameters,
    inspect_parameters,
)
from helios.model.transformer import HeliosForCausalLM


@pytest.fixture
def test_config() -> HeliosConfig:
    """Configured small transformer for rapid adversarial stress-testing."""
    return HeliosConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ffn=128,
        max_seq_len=32,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )


@pytest.fixture
def test_model(test_config: HeliosConfig) -> HeliosForCausalLM:
    """Deterministic model instance."""
    torch.manual_seed(42)
    return HeliosForCausalLM(test_config).eval()


@pytest.mark.tier5
class TestAutoregressiveGenerationAdversarial:
    """Empirical adversarial verification of autoregressive generation."""

    def test_greedy_generation_determinism_and_step_equivalence(
        self, test_model: HeliosForCausalLM
    ) -> None:
        """Verify greedy generation is 100% deterministic and matches step argmax."""
        prompt = torch.tensor([[10, 20, 30]])

        # Determinism across repeated calls
        gen1 = test_model.generate(prompt, max_new_tokens=6, temperature=0.0)
        gen2 = test_model.generate(prompt, max_new_tokens=6, temperature=0.0)
        assert torch.equal(gen1, gen2), "Greedy generation is non-deterministic!"
        assert gen1.shape == (1, 9)

        # Boundary: max_new_tokens <= 0 returns prompt unchanged
        assert torch.equal(test_model.generate(prompt, max_new_tokens=0), prompt)
        assert torch.equal(test_model.generate(prompt, max_new_tokens=-5), prompt)

        # Step 0 token must match model's prefill argmax
        out = test_model(prompt)
        expected_first = torch.argmax(out["logits"][:, -1, :], dim=-1, keepdim=True)
        assert gen1[:, 3:4].equal(expected_first), (
            "First generated token mismatches prefill argmax!"
        )

    def test_stochastic_generation_and_topk_bounds(self, test_model: HeliosForCausalLM) -> None:
        """Verify stochastic sampling changes across seeds and strictly obeys top-k bounds."""
        prompt = torch.tensor([[10, 20, 30]])

        torch.manual_seed(101)
        stoch1 = test_model.generate(prompt, max_new_tokens=6, temperature=0.7, top_k=10)

        torch.manual_seed(202)
        stoch2 = test_model.generate(prompt, max_new_tokens=6, temperature=0.7, top_k=10)

        # Non-deterministic across distinct seeds
        assert not torch.equal(stoch1, stoch2), "Stochastic generation failed to produce variance!"

        # Invariant: test that _sample_token strictly confines samples to top-k
        for _ in range(50):
            logits = torch.randn(2, test_model.config.vocab_size)
            k = 7
            _, topk_indices = torch.topk(logits, k, dim=-1)
            sampled = test_model._sample_token(logits, temperature=0.7, top_k=k)
            for b in range(2):
                token_val = sampled[b].item()
                assert token_val in topk_indices[b].tolist(), (
                    f"Sampled token {token_val} outside top-{k} set {topk_indices[b].tolist()}"
                )

    def test_eos_early_termination_single_sequence(self, test_model: HeliosForCausalLM) -> None:
        """Verify generation halts immediately upon producing eos_token_id."""
        torch.manual_seed(123)
        prompt = torch.tensor([[10, 20]])

        # Generate 8 tokens under fixed seed to observe token sequence
        torch.manual_seed(999)
        full_seq = test_model.generate(prompt, max_new_tokens=8, temperature=0.7, top_k=10)[
            0
        ].tolist()

        # 1. Early stop at step 1: eos_token_id is the first generated token
        eos_step1 = full_seq[2]
        torch.manual_seed(999)
        stopped1 = test_model.generate(
            prompt, max_new_tokens=8, temperature=0.7, top_k=10, eos_token_id=eos_step1
        )[0].tolist()
        assert len(stopped1) == 3, f"Expected length 3 (prompt 2 + 1), got {len(stopped1)}"
        assert stopped1[-1] == eos_step1

        # 2. Early stop at step 3: eos_token_id is the third generated token
        eos_step3 = full_seq[4]
        torch.manual_seed(999)
        stopped3 = test_model.generate(
            prompt, max_new_tokens=8, temperature=0.7, top_k=10, eos_token_id=eos_step3
        )[0].tolist()
        assert len(stopped3) == 5, f"Expected length 5 (prompt 2 + 3), got {len(stopped3)}"
        assert stopped3[-1] == eos_step3

        # 3. No early stop if eos_token_id is never produced
        torch.manual_seed(999)
        no_eos = test_model.generate(
            prompt, max_new_tokens=8, temperature=0.7, top_k=10, eos_token_id=9999
        )[0].tolist()
        assert len(no_eos) == 10
        assert no_eos == full_seq

    def test_eos_batch_padding(self, test_model: HeliosForCausalLM) -> None:
        """Verify batch generation correctly pads finished sequences while others continue."""
        prompt_batch = torch.tensor([[10, 20], [30, 40]])

        torch.manual_seed(999)
        unstopped = test_model.generate(prompt_batch, max_new_tokens=6, temperature=0.7, top_k=10)

        # Choose first generated token in sequence 0 that is different from step 0 token
        step0_token = unstopped[0, 2].item()
        eos_idx = None
        for idx in range(3, unstopped.shape[1]):
            if unstopped[0, idx].item() != step0_token:
                eos_idx = idx
                break
        assert eos_idx is not None, "Failed to find distinct token in sequence 0"

        target_eos = unstopped[0, eos_idx].item()
        pad_id = 0

        torch.manual_seed(999)
        batch_stopped = test_model.generate(
            prompt_batch,
            max_new_tokens=6,
            temperature=0.7,
            top_k=10,
            eos_token_id=target_eos,
            pad_token_id=pad_id,
        )

        # Sequence 0 produced target_eos at eos_idx, followed by padding tokens
        assert batch_stopped[0, eos_idx].item() == target_eos
        for t in range(eos_idx + 1, batch_stopped.shape[1]):
            assert batch_stopped[0, t].item() == pad_id, (
                f"Token at pos {t} was {batch_stopped[0, t].item()}, expected pad {pad_id}"
            )

        # Sequence 1 continues without premature corruption
        assert batch_stopped[1, eos_idx].item() == unstopped[1, eos_idx].item()

    def test_prompt_boundary_conditions(self, test_config: HeliosConfig) -> None:
        """Verify single token prompt, max_seq_len boundaries, and context window guards."""
        model = HeliosForCausalLM(test_config).eval()

        # 1. Prompt of length 1 (scalar sequence)
        p1 = torch.tensor([[42]])
        g1 = model.generate(p1, max_new_tokens=4, temperature=0.0)
        assert g1.shape == (1, 5)
        assert g1[0, 0].item() == 42

        # 2. Batch prompt of length 1
        pb = torch.tensor([[1], [2], [3], [4]])
        gb = model.generate(pb, max_new_tokens=3, temperature=0.0)
        assert gb.shape == (4, 4)

        # 3. Prompt approaching max_seq_len (max_seq_len = 32)
        p_boundary = torch.randint(0, test_config.vocab_size, (1, 30))
        g_boundary = model.generate(p_boundary, max_new_tokens=2, temperature=0.0)
        assert g_boundary.shape == (1, 32)

        # 4. Context overflow during decode raises ValueError
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model.generate(p_boundary, max_new_tokens=4, temperature=0.0)

        # 5. Prompt already exceeding max_seq_len raises ValueError
        p_overflow = torch.randint(0, test_config.vocab_size, (1, 33))
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model.generate(p_overflow, max_new_tokens=1, temperature=0.0)


@pytest.mark.tier5
class TestGradientBackpropagationAdversarial:
    """Empirical verification that 100% of trainable parameters receive valid gradients."""

    @pytest.mark.parametrize("tie_word_embeddings", [True, False])
    def test_100_percent_trainable_parameters_receive_valid_gradients(
        self, tie_word_embeddings: bool
    ) -> None:
        """Verify loss.backward() flows to every parameter without NaN, Inf, or zero grads."""
        cfg = HeliosConfig(
            vocab_size=128,
            d_model=64,
            n_layers=3,
            n_heads=4,
            n_kv_heads=2,
            d_ffn=128,
            max_seq_len=32,
            tie_word_embeddings=tie_word_embeddings,
        )
        model = HeliosForCausalLM(cfg).train()
        torch.manual_seed(42)

        input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
        labels = input_ids.clone()

        out = model(input_ids, labels=labels)
        loss = out["loss"]
        assert loss is not None
        assert torch.isfinite(loss)
        loss.backward()

        total_trainable = 0
        valid_grads = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                total_trainable += 1
                assert param.grad is not None, f"Parameter {name} has None grad!"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} grad has NaN!"
                assert not torch.isinf(param.grad).any(), f"Parameter {name} grad has Inf!"
                grad_sum = param.grad.abs().sum().item()
                assert grad_sum > 0.0, f"Parameter {name} has dead/zero grad!"
                valid_grads += 1

        assert total_trainable > 0
        assert valid_grads == total_trainable, (
            f"Expected 100% valid gradients ({total_trainable}/{total_trainable}), "
            f"but got {valid_grads}/{total_trainable}"
        )

    def test_shifted_loss_alignment_and_masked_labels(self, test_config: HeliosConfig) -> None:
        """Verify shifted cross-entropy aligns token t to target t+1 and respects -100."""
        model = HeliosForCausalLM(test_config).train()
        input_ids = torch.tensor([[10, 20, 30, 40, 50]])
        labels = torch.tensor([[10, 20, 30, 40, 50]])

        out = model(input_ids, labels=labels)
        loss = out["loss"]
        assert loss is not None

        # Verify manual shifted cross entropy
        logits = out["logits"]
        shift_logits = logits[:, :-1, :].contiguous().view(-1, test_config.vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        expected_loss = nn.CrossEntropyLoss()(shift_logits, shift_labels)
        assert torch.allclose(loss, expected_loss, atol=1e-6)

        # Verify ignore_index=-100 masking
        masked_labels = torch.tensor([[10, -100, -100, 40, -100]])
        out_masked = model(input_ids, labels=masked_labels)
        assert torch.isfinite(out_masked["loss"])
        out_masked["loss"].backward()
        assert model.embed_tokens.weight.grad is not None


@pytest.mark.tier5
class TestParameterInspectorPrecisionAndDeduplication:
    """Empirical verification of parameter counting deduplication, formulas, and memory."""

    def test_embedding_weight_tying_deduplication(self) -> None:
        """Verify tied embeddings are deduplicated and untied models add exact vocab * d_model."""
        cfg_tied = HeliosConfig(
            vocab_size=500,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ffn=128,
            tie_word_embeddings=True,
        )
        model_tied = HeliosForCausalLM(cfg_tied)
        rep_tied = inspect_parameters(model_tied)
        assert rep_tied.tied_embeddings is True
        assert model_tied.embed_tokens.weight.data_ptr() == model_tied.lm_head.weight.data_ptr()

        cfg_untied = HeliosConfig(
            vocab_size=500,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ffn=128,
            tie_word_embeddings=False,
        )
        model_untied = HeliosForCausalLM(cfg_untied)
        rep_untied = inspect_parameters(model_untied)
        assert rep_untied.tied_embeddings is False
        assert model_untied.embed_tokens.weight.data_ptr() != model_untied.lm_head.weight.data_ptr()

        delta = rep_untied.total_parameters - rep_tied.total_parameters
        expected_delta = cfg_tied.vocab_size * cfg_tied.d_model
        assert delta == expected_delta, f"Expected delta {expected_delta}, got {delta}"

        counts = count_parameters(model_tied)
        assert counts["total_parameters"] == rep_tied.total_parameters
        assert counts["trainable_parameters"] == rep_tied.trainable_parameters

    @pytest.mark.parametrize(
        ("preset_name", "expected_params"),
        [
            ("nano_15m", 14_528_160),
            ("nano_25m", 24_877_440),
            ("nano_35m", 38_543_872),
        ],
    )
    def test_nano_presets_analytical_formula_and_envelope(
        self, preset_name: str, expected_params: int
    ) -> None:
        """Verify analytical closed-form counts match theoretical formulas and 10M-40M bound."""
        cfg = HeliosConfig.from_preset(preset_name)

        # Analytical theoretical calculation from first principles:
        # V: vocab_size, D: d_model, L: n_layers, H_q: n_heads, H_kv: n_kv_heads, D_ff: d_ffn
        v = cfg.vocab_size
        d = cfg.d_model
        num_layers = cfg.n_layers
        h_q = cfg.n_heads
        h_kv = cfg.n_kv_heads
        d_ff = cfg.d_ffn

        head_dim = d // h_q
        attn_per_layer = (d * h_q * head_dim) + (2 * d * h_kv * head_dim) + (h_q * head_dim * d)
        ffn_per_layer = 3 * d * d_ff
        norms_per_layer = 2 * d
        layer_params = attn_per_layer + ffn_per_layer + norms_per_layer
        final_norm = d
        tied_emb = v * d

        closed_form_total = tied_emb + (num_layers * layer_params) + final_norm
        assert closed_form_total == expected_params

        # Compare with inspector
        theo = calculate_theoretical_parameters(cfg)
        assert theo["total_parameters"] == expected_params
        assert 10_000_000 <= theo["total_parameters"] <= 40_000_000

    def test_multi_precision_memory_footprint_estimation(
        self, test_model: HeliosForCausalLM
    ) -> None:
        """Verify precision-aware memory footprint math across FP32, FP16, BF16, and INT8."""
        rep = inspect_parameters(test_model)
        mb = rep.memory_footprint_mb
        total_p = rep.total_parameters
        denom = 1024.0 * 1024.0

        assert abs(mb["fp32"] - (total_p * 4.0 / denom)) < 1e-9
        assert abs(mb["fp16"] - (total_p * 2.0 / denom)) < 1e-9
        assert abs(mb["bf16"] - (total_p * 2.0 / denom)) < 1e-9
        assert abs(mb["int8"] - (total_p * 1.0 / denom)) < 1e-9
        assert abs(mb["int4"] - (total_p * 0.5 / denom)) < 1e-9

        # Invariant ratios
        assert abs(mb["fp32"] - 2.0 * mb["fp16"]) < 1e-9
        assert abs(mb["fp16"] - mb["bf16"]) < 1e-9
        assert abs(mb["fp32"] - 4.0 * mb["int8"]) < 1e-9
        assert abs(mb["fp32"] - 8.0 * mb["int4"]) < 1e-9


@pytest.mark.tier5
class TestDeviceCompatibilityAdversarial:
    """Empirical verification on CPU and Apple Silicon MPS hardware targets."""

    def test_cpu_forward_backward_and_generate(self, test_config: HeliosConfig) -> None:
        """Verify complete training and inference execution loop on CPU."""
        model = HeliosForCausalLM(test_config).to("cpu").train()
        tokens = torch.randint(0, test_config.vocab_size, (2, 8), device="cpu")
        labels = tokens.clone()

        # Forward & loss
        out = model(tokens, labels=labels)
        assert out["logits"].device.type == "cpu"
        assert out["loss"].device.type == "cpu"

        # Backward
        out["loss"].backward()
        assert model.embed_tokens.weight.grad.device.type == "cpu"

        # Generate
        model.eval()
        gen = model.generate(tokens, max_new_tokens=4, temperature=0.0)
        assert gen.device.type == "cpu"
        assert gen.shape == (2, 12)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_forward_backward_and_generate(self, test_config: HeliosConfig) -> None:
        """Verify complete training and inference execution loop on Apple Silicon MPS."""
        device = torch.device("mps")
        model = HeliosForCausalLM(test_config).to(device).train()
        tokens = torch.randint(0, test_config.vocab_size, (2, 8), device=device)
        labels = tokens.clone()

        # Forward & loss
        out = model(tokens, labels=labels)
        assert out["logits"].device.type == "mps"
        assert out["loss"].device.type == "mps"

        # Backward
        out["loss"].backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"MPS: {name} grad is None"
                assert p.grad.device.type == "mps", f"MPS: {name} grad not on mps"
                assert not torch.isnan(p.grad).any(), f"MPS: {name} grad is NaN"
                assert p.grad.abs().sum().item() > 0.0, f"MPS: {name} grad is zero"

        # Generate (greedy and stochastic)
        model.eval()
        gen_greedy = model.generate(tokens, max_new_tokens=4, temperature=0.0)
        assert gen_greedy.device.type == "mps"
        assert gen_greedy.shape == (2, 12)

        gen_stoch = model.generate(tokens, max_new_tokens=4, temperature=0.7, top_k=5)
        assert gen_stoch.device.type == "mps"
        assert gen_stoch.shape == (2, 12)
