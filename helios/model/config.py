"""Model configuration for the Helios decoder-only transformer architecture."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class HeliosConfig:
    """Hyperparameter configuration for Helios decoder-only transformer models.

    Supports strongly typed presets (Nano-15M, Nano-25M, Nano-35M), parameter
    calculation, validation invariants, and JSON serialization.

    Attributes:
        vocab_size: Size of vocabulary including special tokens.
        d_model: Hidden dimension of the transformer representations.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention query heads.
        n_kv_heads: Number of key/value heads for Grouped-Query Attention (GQA).
        d_ffn: Intermediate hidden dimension of the SwiGLU feed-forward network.
        max_seq_len: Maximum sequence context window length.
        rope_theta: Base frequency constant theta for Rotary Position Embeddings.
        rms_norm_eps: Numerical stability constant epsilon in RMSNorm.
        tie_word_embeddings: Whether to tie input embeddings to output LM head weights.
        initializer_range: Standard deviation for weight initialization.
        dropout: Dropout rate for attention and residual connections.
    """

    vocab_size: int = 32000
    d_model: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    d_ffn: int = 768
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    dropout: float = 0.0
    model_family: str = "Helios"
    developer: str = "Arron Kian Parejas"
    developer_email: str = "arronkianparejas@gmail.com"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError(
                f"d_model and n_layers must be positive, "
                f"got d_model={self.d_model}, n_layers={self.n_layers}"
            )
        if self.d_ffn <= 0:
            raise ValueError(f"d_ffn must be positive, got {self.d_ffn}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads}) for GQA"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even for Rotary Position Embeddings"
            )
        if self.rms_norm_eps <= 0.0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}")
        if self.rope_theta <= 0.0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")

    @property
    def head_dim(self) -> int:
        """Dimension of each individual attention head."""
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        """Compatibility alias for d_ffn."""
        return self.d_ffn

    @property
    def rope_base(self) -> float:
        """Compatibility alias for rope_theta."""
        return self.rope_theta

    @property
    def tie_embeddings(self) -> bool:
        """Compatibility alias for tie_word_embeddings."""
        return self.tie_word_embeddings

    def count_parameters(self, tie_word_embeddings: bool | None = None) -> dict[str, int]:
        """Compute exact theoretical parameter breakdown for this configuration.

        Args:
            tie_word_embeddings: Override embedding tying flag (defaults to
                self.tie_word_embeddings).

        Returns:
            Dictionary containing exact parameter counts.
        """
        tied = self.tie_word_embeddings if tie_word_embeddings is None else tie_word_embeddings
        head_dim = self.head_dim
        emb_params = self.vocab_size * self.d_model

        # Attention sublayer per block: input norm + Wq + Wk + Wv + Wo
        attn_norm_params = self.d_model
        q_proj_params = self.d_model * (self.n_heads * head_dim)
        k_proj_params = self.d_model * (self.n_kv_heads * head_dim)
        v_proj_params = self.d_model * (self.n_kv_heads * head_dim)
        o_proj_params = (self.n_heads * head_dim) * self.d_model
        attn_params = q_proj_params + k_proj_params + v_proj_params + o_proj_params

        # SwiGLU FFN sublayer per block: post-attn norm + gate + up + down
        ffn_norm_params = self.d_model
        ffn_params = 3 * self.d_model * self.d_ffn

        per_layer_params = attn_norm_params + attn_params + ffn_norm_params + ffn_params
        total_transformer_layers = self.n_layers * per_layer_params
        final_norm_params = self.d_model
        lm_head_params = 0 if tied else emb_params

        total = emb_params + total_transformer_layers + final_norm_params + lm_head_params
        non_emb = total_transformer_layers + final_norm_params + (0 if tied else lm_head_params)

        return {
            "total_parameters": total,
            "non_embedding_parameters": non_emb,
            "embedding_parameters": emb_params,
            "per_layer_parameters": per_layer_params,
            "attention_per_layer": attn_params,
            "feed_forward_per_layer": ffn_params,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration instance to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeliosConfig:
        """Construct HeliosConfig from a dictionary with alias resolution."""
        payload = dict(data)
        if "d_ff" in payload and "d_ffn" not in payload:
            payload["d_ffn"] = payload.pop("d_ff")
        if "rope_base" in payload and "rope_theta" not in payload:
            payload["rope_theta"] = payload.pop("rope_base")
        if "tie_embeddings" in payload and "tie_word_embeddings" not in payload:
            payload["tie_word_embeddings"] = payload.pop("tie_embeddings")

        field_names = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in payload.items() if k in field_names}
        return cls(**filtered)

    def save_pretrained(self, save_directory: str | Path) -> str:
        """Export config to config.json within the specified directory.

        Args:
            save_directory: Target directory or file path.

        Returns:
            Absolute string path to written config.json.
        """
        target = Path(save_directory)
        if target.is_dir() or not target.suffix:
            target.mkdir(parents=True, exist_ok=True)
            out_file = target / "config.json"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            out_file = target

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return str(out_file.resolve())

    @classmethod
    def from_pretrained(cls, load_directory: str | Path) -> HeliosConfig:
        """Load HeliosConfig from config.json or JSON file path."""
        target = Path(load_directory)
        if target.is_dir():
            target = target / "config.json"
        if not target.is_file():
            raise FileNotFoundError(f"Configuration file not found at {target}")

        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def nano_15m(cls, **overrides: Any) -> HeliosConfig:
        """Helios Nano 15M target preset (~14.53M tied parameters)."""
        cfg = {
            "vocab_size": 32000,
            "d_model": 288,
            "n_layers": 6,
            "n_heads": 6,
            "n_kv_heads": 2,
            "d_ffn": 768,
            "max_seq_len": 2048,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": True,
        }
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def helios_nano_15m(cls, **overrides: Any) -> HeliosConfig:
        """Alias for nano_15m."""
        return cls.nano_15m(**overrides)

    @classmethod
    def nano_25m(cls, **overrides: Any) -> HeliosConfig:
        """Helios Nano 25M target preset (~24.88M tied parameters)."""
        cfg = {
            "vocab_size": 32000,
            "d_model": 384,
            "n_layers": 8,
            "n_heads": 6,
            "n_kv_heads": 2,
            "d_ffn": 1024,
            "max_seq_len": 2048,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": True,
        }
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def helios_nano_25m(cls, **overrides: Any) -> HeliosConfig:
        """Alias for nano_25m."""
        return cls.nano_25m(**overrides)

    @classmethod
    def nano_35m(cls, **overrides: Any) -> HeliosConfig:
        """Helios Nano 35M target preset (~38.54M tied parameters, 10M-40M target bound)."""
        cfg = {
            "vocab_size": 32000,
            "d_model": 512,
            "n_layers": 8,
            "n_heads": 8,
            "n_kv_heads": 2,
            "d_ffn": 1376,
            "max_seq_len": 2048,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": True,
        }
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def helios_nano_35m(cls, **overrides: Any) -> HeliosConfig:
        """Alias for nano_35m."""
        return cls.nano_35m(**overrides)

    @classmethod
    def nano_20m(cls, **overrides: Any) -> HeliosConfig:
        """Helios Nano 20M target preset (~19.4M tied parameters)."""
        cfg = {
            "vocab_size": 32000,
            "d_model": 320,
            "n_layers": 7,
            "n_heads": 8,
            "n_kv_heads": 2,
            "d_ffn": 864,
            "max_seq_len": 2048,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": True,
        }
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def helios_nano_20m(cls, **overrides: Any) -> HeliosConfig:
        """Alias for nano_20m."""
        return cls.nano_20m(**overrides)

    @classmethod
    def alpha_100m(cls, **overrides: Any) -> HeliosConfig:
        """Helios Alpha 100M stage preset (~100.1M tied parameters)."""
        cfg = {
            "vocab_size": 32000,
            "d_model": 768,
            "n_layers": 12,
            "n_heads": 12,
            "n_kv_heads": 4,
            "d_ffn": 2048,
            "max_seq_len": 2048,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": True,
        }
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def beta_300m(cls, **overrides: Any) -> HeliosConfig:
        """Helios Beta 300M stage preset (~302M tied parameters)."""
        cfg = {
            "vocab_size": 32000,
            "d_model": 1024,
            "n_layers": 24,
            "n_heads": 16,
            "n_kv_heads": 4,
            "d_ffn": 2816,
            "max_seq_len": 2048,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": True,
        }
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def core_1b(cls, **overrides: Any) -> HeliosConfig:
        """Helios Core 1B milestone preset (~1.06B tied parameters)."""
        cfg = {
            "vocab_size": 32000,
            "d_model": 2048,
            "n_layers": 24,
            "n_heads": 16,
            "n_kv_heads": 4,
            "d_ffn": 5504,
            "max_seq_len": 4096,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": True,
        }
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def pro_3b(cls, **overrides: Any) -> HeliosConfig:
        """Helios Pro 3B milestone preset (~3.1B tied parameters)."""
        cfg = {
            "vocab_size": 32000,
            "d_model": 3072,
            "n_layers": 32,
            "n_heads": 24,
            "n_kv_heads": 8,
            "d_ffn": 8192,
            "max_seq_len": 4096,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": True,
        }
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def flagship_6b(cls, **overrides: Any) -> HeliosConfig:
        """Helios Flagship 6B flagship milestone preset (~6.2B tied parameters)."""
        cfg = {
            "vocab_size": 32000,
            "d_model": 4096,
            "n_layers": 32,
            "n_heads": 32,
            "n_kv_heads": 8,
            "d_ffn": 11008,
            "max_seq_len": 4096,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": True,
        }
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def from_preset(cls, preset_name: str, **overrides: Any) -> HeliosConfig:
        """Retrieve HeliosConfig by preset name."""
        name = preset_name.lower().replace("-", "_")
        presets = {
            "helios_nano_15m": cls.nano_15m,
            "nano_15m": cls.nano_15m,
            "helios_nano_20m": cls.nano_20m,
            "nano_20m": cls.nano_20m,
            "helios_nano_25m": cls.nano_25m,
            "nano_25m": cls.nano_25m,
            "helios_nano_35m": cls.nano_35m,
            "nano_35m": cls.nano_35m,
            "helios_alpha_100m": cls.alpha_100m,
            "alpha_100m": cls.alpha_100m,
            "alpha": cls.alpha_100m,
            "helios_beta_300m": cls.beta_300m,
            "beta_300m": cls.beta_300m,
            "beta": cls.beta_300m,
            "helios_core_1b": cls.core_1b,
            "core_1b": cls.core_1b,
            "core": cls.core_1b,
            "helios_pro_3b": cls.pro_3b,
            "pro_3b": cls.pro_3b,
            "pro": cls.pro_3b,
            "helios_flagship_6b": cls.flagship_6b,
            "flagship_6b": cls.flagship_6b,
            "flagship": cls.flagship_6b,
        }
        if name not in presets:
            raise KeyError(f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}")
        return presets[name](**overrides)
