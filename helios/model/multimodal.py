"""Helios Multimodal Vision-Language Architecture (Feature 45).

Integrates a Vision Transformer (ViT) patch encoder (patch size 14x14, 224/336 res),
a 2-layer GELU MLP projection head, a unified 2D prefix multimodal attention mask,
and HeliosForCausalLM backbone with verified forward/backward gradient backpropagation.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from helios.model.attention import KVCache
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM

IMAGE_TOKEN_ID: int = 32001


@dataclass
class HeliosVisionConfig:
    """Configuration for Helios Vision Transformer patch encoder."""

    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02


@dataclass
class CausalLMOutputWithPast:
    """Container for multimodal causal language model outputs."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    past_key_values: list[Any] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class HeliosVisionBlock(nn.Module):
    """A single Transformer encoder block for vision features with Pre-LayerNorm."""

    def __init__(self, config: HeliosVisionConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_1 = self.ln_1(x)
        attn_out, _ = self.self_attn(norm_1, norm_1, norm_1, need_weights=False)
        h = x + attn_out
        return h + self.mlp(self.ln_2(h))


class HeliosVisionEncoder(nn.Module):
    """Vision Transformer (ViT) patch encoder with 2D bicubic position interpolation."""

    def __init__(self, config: HeliosVisionConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeliosVisionConfig()

        self.patch_embed = nn.Conv2d(
            in_channels=self.config.num_channels,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
            bias=False,
        )

        base_grid = self.config.image_size // self.config.patch_size
        num_patches = base_grid * base_grid
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches, self.config.hidden_size) * self.config.initializer_range
        )

        self.pre_layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.layers = nn.ModuleList(
            [HeliosVisionBlock(self.config) for _ in range(self.config.num_hidden_layers)]
        )
        self.post_layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)

    def interpolate_pos_encoding(self, h: int, w: int) -> torch.Tensor:
        """Dynamically interpolate 2D positional embeddings for arbitrary image resolutions."""
        grid_h = h // self.config.patch_size
        grid_w = w // self.config.patch_size
        target_patches = grid_h * grid_w
        if target_patches == self.position_embeddings.shape[1]:
            return self.position_embeddings

        orig_grid = int(math.isqrt(self.position_embeddings.shape[1]))
        pos_tokens = self.position_embeddings.reshape(
            1, orig_grid, orig_grid, self.config.hidden_size
        ).permute(0, 3, 1, 2)

        interpolated = F.interpolate(
            pos_tokens,
            size=(grid_h, grid_w),
            mode="bicubic",
            align_corners=False,
        )
        return interpolated.permute(0, 2, 3, 1).reshape(1, target_patches, self.config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode image tensor (B, 3, H, W) into patch token sequence (B, N, D_vis)."""
        _, _, h, w = pixel_values.shape
        x = self.patch_embed(pixel_values)  # (B, D_vis, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D_vis)

        pos_emb = self.interpolate_pos_encoding(h, w)
        x = self.pre_layernorm(x + pos_emb.to(device=x.device, dtype=x.dtype))

        for layer in self.layers:
            x = layer(x)

        return self.post_layernorm(x)


class HeliosMultiModalProjector(nn.Module):
    """2-layer GELU MLP projector mapping visual dimension to LLM hidden dimension."""

    def __init__(self, vision_hidden_size: int = 1024, llm_d_model: int = 2048) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden_size, llm_d_model)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(llm_d_model, llm_d_model)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(visual_tokens)))


def build_multimodal_attention_mask(
    seq_len: int,
    image_start: int,
    image_end: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build unified 2D prefix causal + bidirectional visual attention mask.

    Within [image_start, image_end), visual tokens attend bidirectionally (0.0).
    Preceding text tokens condition visual tokens; subsequent text tokens attend to all
    visual tokens; future tokens are masked with -inf.
    """
    q_idx = torch.arange(seq_len, device=device)[:, None]
    k_idx = torch.arange(seq_len, device=device)[None, :]

    mask = torch.where(
        k_idx <= q_idx,
        torch.zeros(1, device=device, dtype=dtype),
        torch.full((1,), float("-inf"), device=device, dtype=dtype),
    )

    if 0 <= image_start < image_end <= seq_len:
        mask[image_start:image_end, image_start:image_end] = 0.0

    return mask


class HeliosForMultimodalLM(nn.Module):
    """Helios Multimodal Vision-Language Model integrating ViT, Projector, and CausalLM."""

    IMAGE_TOKEN_ID: int = IMAGE_TOKEN_ID

    def __init__(
        self,
        config: HeliosConfig,
        vision_config: HeliosVisionConfig | None = None,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_config = vision_config or HeliosVisionConfig()

        self.language_model = HeliosForCausalLM(config, init_weights=init_weights)
        self.vision_encoder = HeliosVisionEncoder(self.vision_config)
        self.projector = HeliosMultiModalProjector(
            vision_hidden_size=self.vision_config.hidden_size,
            llm_d_model=config.d_model,
        )

    @classmethod
    def core_1b_multimodal(cls) -> HeliosForMultimodalLM:
        """Factory for Core-1.13B Multimodal model (D_vis=1024 -> D_model=2048)."""
        return cls(config=HeliosConfig.core_1b(), vision_config=HeliosVisionConfig())

    @classmethod
    def flagship_6b_multimodal(cls) -> HeliosForMultimodalLM:
        """Factory for Flagship-5.80B Multimodal model (D_vis=1024 -> D_model=4096)."""
        return cls(config=HeliosConfig.flagship_6b(), vision_config=HeliosVisionConfig())

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.language_model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        start_pos: int = 0,
        kv_cache: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutputWithPast:
        """Execute multimodal or text-only forward pass with gradient flow."""
        bsz, seq_len = input_ids.shape

        if pixel_values is not None:
            # 1. Vision feature extraction and projection
            vis_tokens = self.vision_encoder(pixel_values)
            proj_vis = self.projector(vis_tokens)
            n_vis = proj_vis.shape[1]

            # 2. Token sequence embedding merger
            safe_ids = torch.where(input_ids >= self.config.vocab_size, 0, input_ids)
            inputs_embeds = self.language_model.embed_tokens(safe_ids)

            if image_mask is not None:
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[image_mask] = proj_vis.reshape(-1, self.config.d_model)
                if attention_mask is None and bsz == 1:
                    indices = torch.where(image_mask[0])[0]
                    if len(indices) > 0:
                        v_start, v_end = int(indices[0].item()), int(indices[-1].item()) + 1
                        attention_mask = build_multimodal_attention_mask(
                            seq_len,
                            v_start,
                            v_end,
                            dtype=inputs_embeds.dtype,
                            device=inputs_embeds.device,
                        )
            else:
                has_image = input_ids == self.IMAGE_TOKEN_ID
                if has_image.any():
                    max_placeholders = int(has_image.sum(dim=-1).max().item())
                    if max_placeholders == n_vis:
                        # Case A: Sequence already has exact N placeholder tokens
                        inputs_embeds = inputs_embeds.clone()
                        inputs_embeds[has_image] = proj_vis.reshape(-1, self.config.d_model)
                        if attention_mask is None and bsz == 1:
                            indices = torch.where(has_image[0])[0]
                            v_start = int(indices[0].item())
                            v_end = int(indices[-1].item()) + 1
                            attention_mask = build_multimodal_attention_mask(
                                seq_len,
                                v_start,
                                v_end,
                                dtype=inputs_embeds.dtype,
                                device=inputs_embeds.device,
                            )
                    else:
                        # Case B: Single placeholder per batch item, spliced at index p
                        first_idx = int(torch.where(has_image[0])[0][0].item())
                        inputs_embeds = torch.cat(
                            [
                                inputs_embeds[:, :first_idx, :],
                                proj_vis,
                                inputs_embeds[:, first_idx + 1 :, :],
                            ],
                            dim=1,
                        )
                        if labels is not None:
                            neg_labels = torch.full(
                                (bsz, n_vis),
                                -100,
                                dtype=labels.dtype,
                                device=labels.device,
                            )
                            labels = torch.cat(
                                [
                                    labels[:, :first_idx],
                                    neg_labels,
                                    labels[:, first_idx + 1 :],
                                ],
                                dim=1,
                            )
                        if attention_mask is None and bsz == 1:
                            total_len = inputs_embeds.shape[1]
                            attention_mask = build_multimodal_attention_mask(
                                total_len,
                                first_idx,
                                first_idx + n_vis,
                                dtype=inputs_embeds.dtype,
                                device=inputs_embeds.device,
                            )
                else:
                    # Fallback: prepend projected visual tokens if no image tokens specified
                    inputs_embeds = torch.cat([proj_vis, inputs_embeds], dim=1)
                    if labels is not None:
                        neg_labels = torch.full(
                            (bsz, n_vis), -100, dtype=labels.dtype, device=labels.device
                        )
                        labels = torch.cat([neg_labels, labels], dim=1)
                    if attention_mask is None and bsz == 1:
                        total_len = inputs_embeds.shape[1]
                        attention_mask = build_multimodal_attention_mask(
                            total_len,
                            0,
                            n_vis,
                            dtype=inputs_embeds.dtype,
                            device=inputs_embeds.device,
                        )

            # 3. LLM Backbone Forward
            out = self.language_model(
                inputs_embeds=inputs_embeds,
                labels=labels,
                start_pos=start_pos,
                kv_cache=kv_cache,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
        else:
            # Text-only forward fallback
            out = self.language_model(
                input_ids=input_ids,
                labels=labels,
                start_pos=start_pos,
                kv_cache=kv_cache,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )

        return CausalLMOutputWithPast(
            loss=out["loss"],
            logits=out["logits"],
            past_key_values=out.get("kv_caches") or out.get("kv_cache"),
        )
