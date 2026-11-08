"""Helios Decoder-Only Causal Language Model.

Integrates token embeddings, HeliosBlock layers, final RMSNorm,
causal language model head with weight tying, and autoregressive generation
with pre-allocated KVCache and top-k sampling.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from helios.model.attention import KVCache
from helios.model.config import HeliosConfig
from helios.model.normalization import RMSNorm
from helios.model.transformer_block import HeliosBlock


class HeliosForCausalLM(nn.Module):
    """Helios Transformer for Causal Language Modeling.

    Implements a decoder-only architecture featuring RoPE, Pre-RMSNorm,
    GQA, SwiGLU, tied embeddings, and optimized KV caching.
    """

    def __init__(self, config: HeliosConfig, init_weights: bool = True) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([HeliosBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: alias lm_head weight directly to token embeddings
        tie = getattr(config, "tie_word_embeddings", getattr(config, "tie_embeddings", True))
        if tie:
            self.lm_head.weight = self.embed_tokens.weight

        # Parameter initialization (can be bypassed when restoring from checkpoints)
        if init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with Gaussian distribution scaled by initializer_range."""
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def get_input_embeddings(self) -> nn.Embedding:
        """Retrieve input token embedding module."""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set input token embedding module and re-tie weights if enabled."""
        self.embed_tokens = value
        tie = getattr(
            self.config, "tie_word_embeddings", getattr(self.config, "tie_embeddings", True)
        )
        if tie:
            self.lm_head.weight = self.embed_tokens.weight

    def get_output_embeddings(self) -> nn.Linear:
        """Retrieve output language model head module."""
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        start_pos: int = 0,
        kv_cache: list[KVCache] | None = None,
        kv_caches: list[KVCache] | None = None,
        use_cache: bool = False,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Execute forward pass for causal language modeling.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            labels: Optional ground-truth token IDs of shape (batch_size, seq_len).
            start_pos: Starting position index for rotary embeddings and KV cache cursor.
            kv_cache: Optional list of KVCache objects (one per layer).
            kv_caches: Alias for kv_cache.
            use_cache: If True and kv_caches is None, dynamically creates KVCache instances.
            inputs_embeds: Optional pre-embedded activations of shape (batch_size, seq_len, d_model).
            attention_mask: Optional attention mask tensor for prefix/multimodal masking.
            attn_mask: Alias for attention_mask.

        Returns:
            Dict containing:
              - 'logits': Output logits of shape (batch_size, seq_len, vocab_size).
              - 'loss': Scalar cross-entropy loss if labels provided, else None.
              - 'kv_caches': List of updated KVCache objects if used.
              - 'kv_cache': Alias for kv_caches for contract compatibility.
        """
        if inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            h = inputs_embeds
            device = inputs_embeds.device
            dtype = inputs_embeds.dtype
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
            h = self.embed_tokens(input_ids)
            device = input_ids.device
            dtype = self.embed_tokens.weight.dtype
        else:
            raise ValueError("Either input_ids or inputs_embeds must be specified.")

        active_caches = kv_caches or kv_cache

        # Initialize KV caches if requested and not provided
        if use_cache and active_caches is None:
            head_dim = self.config.d_model // self.config.n_heads
            active_caches = [
                KVCache(
                    max_batch_size=batch_size,
                    max_seq_len=self.config.max_seq_len,
                    n_kv_heads=self.config.n_kv_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(self.config.n_layers)
            ]

        mask = attention_mask if attention_mask is not None else attn_mask

        # Pass through all transformer blocks
        for i, layer in enumerate(self.layers):
            layer_cache = active_caches[i] if active_caches is not None else None
            h = layer(h, start_pos=start_pos, kv_cache=layer_cache, attn_mask=mask)

        # Final RMSNorm
        h = self.norm(h)

        # Project to vocabulary logits
        logits = self.lm_head(h)

        # Compute causal cross-entropy loss if labels are supplied
        loss = None
        if labels is not None:
            if seq_len > 1:
                # Shift so that tokens < n predict token n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                )
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    logits.view(-1, self.config.vocab_size),
                    labels.view(-1),
                )

        return {
            "logits": logits,
            "loss": loss,
            "kv_caches": active_caches,
            "kv_cache": active_caches,
        }

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively with stateful KV caching and top-k sampling.

        Args:
            prompt_tokens: Prompt token IDs of shape (batch_size, prompt_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = greedy argmax).
            top_k: Top-k filtering threshold (None or <=0 = disabled).
            eos_token_id: End-of-sequence token ID to halt generation.
            pad_token_id: Token ID used to pad halted sequences.

        Returns:
            Tensor of token IDs of shape (batch_size, prompt_len + generated_len).
        """
        self.eval()
        batch_size, prompt_len = prompt_tokens.shape
        device = prompt_tokens.device

        if max_new_tokens <= 0:
            return prompt_tokens

        total_len = prompt_len + max_new_tokens
        head_dim = self.config.d_model // self.config.n_heads

        # Pre-allocate contiguous KVCache per layer
        kv_caches = [
            KVCache(
                max_batch_size=batch_size,
                max_seq_len=min(total_len, self.config.max_seq_len),
                n_kv_heads=self.config.n_kv_heads,
                head_dim=head_dim,
                dtype=self.embed_tokens.weight.dtype,
                device=device,
            )
            for _ in range(self.config.n_layers)
        ]

        # 1. Prefill Step (step 0): Process entire prompt tokens at once
        out = self.forward(prompt_tokens, start_pos=0, kv_caches=kv_caches)
        next_token_logits = out["logits"][:, -1, :]

        next_token = self._sample_token(next_token_logits, temperature=temperature, top_k=top_k)
        generated_tokens = [next_token]

        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if eos_token_id is not None:
            is_finished |= next_token.squeeze(-1) == eos_token_id

        # 2. Incremental Decode Steps: Single-token generation with cached KV
        cur_token = next_token
        for step in range(1, max_new_tokens):
            if is_finished.all():
                break

            current_pos = prompt_len + step - 1
            out = self.forward(cur_token, start_pos=current_pos, kv_caches=kv_caches)
            next_token_logits = out["logits"][:, 0, :]

            cur_token = self._sample_token(next_token_logits, temperature=temperature, top_k=top_k)

            if eos_token_id is not None:
                if pad_token_id is not None:
                    cur_token = torch.where(is_finished.unsqueeze(-1), pad_token_id, cur_token)
                is_finished |= cur_token.squeeze(-1) == eos_token_id

            generated_tokens.append(cur_token)

        all_generated = torch.cat(generated_tokens, dim=-1)
        return torch.cat([prompt_tokens, all_generated], dim=-1)

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Sample a token ID from logits via greedy argmax or temperature / top-k."""
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        scaled_logits = logits / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            top_k_val = min(top_k, scaled_logits.shape[-1])
            kth_vals, _ = torch.topk(scaled_logits, top_k_val, dim=-1)
            min_val = kth_vals[:, -1].unsqueeze(-1)
            scaled_logits = torch.where(scaled_logits < min_val, -float("inf"), scaled_logits)

        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
