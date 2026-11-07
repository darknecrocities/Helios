"""Hugging Face Hub export engine and safetensors serialization for Helios.

Generates standard Hugging Face repository files:
- config.json (Hugging Face transformers format with auto_map)
- generation_config.json
- tokenizer.json, tokenizer_config.json, special_tokens_map.json
- model.safetensors (safe zero-copy weight serialization)
- README.md (Comprehensive model card with dataset citations & Arron Kian Parejas attribution)

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file

from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios.tokenizer.base import BaseTokenizer
from helios.tokenizer.bpe import ByteLevelBPETokenizer

logger = logging.getLogger("helios.export.hf")


def generate_hf_config_dict(config: HeliosConfig) -> dict[str, Any]:
    """Generate Hugging Face compliant configuration dictionary."""
    return {
        "architectures": ["HeliosForCausalLM"],
        "model_type": "helios",
        "auto_map": {
            "AutoConfig": "helios.model.config.HeliosConfig",
            "AutoModelForCausalLM": "helios.model.transformer.HeliosForCausalLM",
            "AutoTokenizer": "helios.tokenizer.base.HeliosTokenizer",
        },
        "vocab_size": config.vocab_size,
        "hidden_size": config.d_model,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads,
        "intermediate_size": config.d_ffn,
        "max_position_embeddings": config.max_seq_len,
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.rms_norm_eps,
        "tie_word_embeddings": config.tie_word_embeddings,
        "initializer_range": config.initializer_range,
        "dropout": config.dropout,
        "torch_dtype": "float32",
        "transformers_version": "4.40.0",
        "model_family": config.model_family,
        "developer": config.developer,
        "developer_email": config.developer_email,
    }


def generate_hf_generation_config(
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    pad_token_id: int = 3,
    max_length: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
) -> dict[str, Any]:
    """Generate standard Hugging Face generation configuration dictionary."""
    return {
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": True,
        "transformers_version": "4.40.0",
    }


def save_safetensors_weights(
    model: nn.Module,
    output_path: Path | str,
) -> int:
    """Serialize model weights into a safetensors file.

    Handles weight tying safely to prevent shared tensor storage errors in safetensors.

    Args:
        model: PyTorch model instance.
        output_path: Destination path for .safetensors file.

    Returns:
        Written file size in bytes.
    """
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    clean_dict: dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        # Ensure tensor is detached, on CPU, and contiguous
        t = v.detach().cpu().contiguous()
        clean_dict[k] = t

    # Handle tied embeddings or shared memory tensors:
    # safetensors rejects tensors sharing identical memory pointers
    seen_ptrs: set[int] = set()
    for k, v in list(clean_dict.items()):
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            clean_dict[k] = v.clone()
        else:
            seen_ptrs.add(ptr)

    save_file(clean_dict, str(target))
    size_bytes = target.stat().st_size
    logger.info("Saved safetensors weights to %s (%d bytes)", target, size_bytes)
    return size_bytes


def generate_hf_model_card(
    config: HeliosConfig,
    repo_id: str = "Darknecrocities/Helios",
    license_tag: str = "apache-2.0",
) -> str:
    """Generate Hugging Face Model Card README.md with YAML metadata and citations."""
    param_counts = config.count_parameters()
    total_m = param_counts["total_parameters"] / 1e6

    card = f"""---
license: {license_tag}
tags:
  - helios
  - open-weight
  - pytorch
  - causal-lm
  - rotary-embeddings
  - grouped-query-attention
  - swiglu
  - hierarchical-memory
datasets:
  - HuggingFaceFW/fineweb-edu
  - HuggingFaceFW/fineweb
  - mlfoundations/dCLM-baseline
  - HuggingFaceTB/cosmopedia-v2
  - open-web-math/open-web-math
  - EleutherAI/proof-pile-2
  - bigcode/the-stack-v2
  - bigcode/starcoderdata
  - uonlp/CulturaX
  - oscar-corpus/OSCAR-2301
  - statmt/cc-100
  - wikimedia/wikipedia
pipeline_tag: text-generation
language:
  - en
  - tl
---

# Helios Open-Weight Foundation Model ({config.model_family})

**Developed by: Arron Kian Parejas** (`arronkianparejas@gmail.com`)  
**GitHub Repository**: [https://github.com/darknecrocities/Helios](https://github.com/darknecrocities/Helios)  
**Hugging Face Hub**: [https://huggingface.co/{repo_id}](https://huggingface.co/{repo_id})  
**Ollama Registry**: `parejasarronkian/Helios`  
**License**: Apache-2.0 / MIT Dual License  

---

## 1. Overview & Architecture

Helios is a modular, transparent, open-weight foundation model ecosystem
engineered for efficient inference on consumer edge hardware and clusters.

### Core Architectural Specifications:
- **Architecture**: Decoder-only Causal Transformer
- **Parameters**: ~{total_m:.2f}M ({config.n_layers} layers, {config.d_model} hidden dim)
- **Attention**: Grouped-Query Attention (GQA, {config.n_heads} Q : {config.n_kv_heads} KV)
- **Position Representation**: Rotary Position Embeddings (RoPE, theta={config.rope_theta})
- **Normalization**: Pre-RMSNorm (eps={config.rms_norm_eps})
- **Feed-Forward**: SwiGLU Gated MLP (intermediate dim={config.d_ffn})
- **Context Length**: {config.max_seq_len} tokens
- **Memory Augmentation**: HMA with Working, Episodic, Semantic, and Procedural tiers.

---

## 2. Parameter Scaling Roadmap & Exit Gates

| Stage | Target Size | Hardware Target | Status | Exit Gates |
| :--- | :--- | :--- | :--- | :--- |
| **Nano** | 10M–40M | Apple Silicon M3 | **Active** | Deterministic resume, overfit tiny corpus |
| **Alpha** | 100M | Single GPU / MPS | Planned | Perplexity < 25, fertility validated |
| **Beta** | 300M | Single GPU (24GB) | Planned | GQA KV-cache efficiency, zero regressions |
| **Core** | 1B | Multi-GPU Node | Roadmap | FSDP2 sharding, cluster recipe |
| **Pro** | 3B | Cluster (8x H100) | Long-Term | Multi-node recovery, optimizer sharding |
| **Flagship**| 6B | Compute Cluster | Long-Term | Full training provenance, safety review |

---

## 3. Pretraining Datasets & Domain Allocation

Helios models are trained on an overtraining schedule using 12 high-quality datasets:

| Domain | Dataset Name | Hugging Face Dataset ID | Ratio |
| :--- | :--- | :--- | :--- |
| **Educational** | FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 30% |
| **General Web** | FineWeb & DCLM | `HuggingFaceFW/fineweb` | 15% |
| **Synthetic** | Cosmopedia v2 | `HuggingFaceTB/cosmopedia-v2` | 15% |
| **Code** | The Stack v2 | `bigcode/the-stack-v2` | 20% |
| **STEM/Math** | OpenWebMath | `open-web-math/open-web-math` | 10% |
| **Filipino/TL** | CulturaX (tl) | `uonlp/CulturaX` (tl) | 10% |

---

## 4. Quickstart Usage

### Using Helios CLI:
```bash
# Inspect model parameters
helios inspect -m nano-20m

# Run interactive generation
helios generate --prompt "Write a Python function to compute Fibonacci numbers."
```

### Using Transformers:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")

inputs = tokenizer("Hello Helios! Who developed you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 5. Citation & Attribution

```bibtex
@misc{{helios2026,
  author = {{Parejas, Arron Kian}},
  title = {{Helios: Transparent Open-Weight Foundation Model Ecosystem}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  journal = {{Hugging Face Hub Repository}},
  howpublished = {{\\url{{https://huggingface.co/{repo_id}}}}}
}}
```
"""
    return card.strip() + "\n"


def export_huggingface(
    output_dir: Path | str = "deploy/huggingface",
    config: HeliosConfig | None = None,
    model: nn.Module | None = None,
    tokenizer: BaseTokenizer | None = None,
    checkpoint_path: Path | str | None = None,
    repo_id: str = "Darknecrocities/Helios",
    generate_card: bool = True,
) -> dict[str, Any]:
    """Export complete Hugging Face repository package.

    Args:
        output_dir: Destination directory.
        config: Model configuration (defaults to nano-20m).
        model: Optional model instance.
        tokenizer: Optional tokenizer instance (defaults to ByteLevelBPETokenizer).
        checkpoint_path: Optional path to checkpoint .pt file or directory.
        repo_id: Target Hugging Face Hub repository identifier.
        generate_card: Whether to generate the model card README.md.

    Returns:
        Dictionary detailing exported artifact paths and sizes.
    """
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)

    cfg = config or HeliosConfig.nano_20m()
    tok = tokenizer or ByteLevelBPETokenizer()

    # 1. Config files
    config_dict = generate_hf_config_dict(cfg)
    config_file = target / "config.json"
    config_file.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    special_map = tok.special_tokens
    gen_config = generate_hf_generation_config(
        bos_token_id=special_map.get("<|begin_of_text|>", 1),
        eos_token_id=special_map.get("<|end_of_text|>", 2),
        pad_token_id=special_map.get("<|pad|>", 3),
        max_length=cfg.max_seq_len,
    )
    gen_config_file = target / "generation_config.json"
    gen_config_file.write_text(json.dumps(gen_config, indent=2), encoding="utf-8")

    # 2. Tokenizer artifacts
    tok.save_pretrained(target)

    # 3. Model weights (.safetensors)
    weights_path = target / "model.safetensors"
    if model is None:
        if checkpoint_path is not None:
            ckpt_p = Path(checkpoint_path)
            state_file = ckpt_p / "state.pt" if ckpt_p.is_dir() else ckpt_p
            payload = torch.load(state_file, map_location="cpu", weights_only=False)
            model = HeliosForCausalLM(cfg)
            model.load_state_dict(payload.get("model_state_dict", payload))
        else:
            model = HeliosForCausalLM(cfg)

    weights_bytes = save_safetensors_weights(model, weights_path)

    # 4. Model card
    card_path = target / "README.md"
    if generate_card:
        card_content = generate_hf_model_card(cfg, repo_id=repo_id)
        card_path.write_text(card_content, encoding="utf-8")

    return {
        "output_directory": str(target),
        "config_file": str(config_file),
        "generation_config_file": str(gen_config_file),
        "tokenizer_files": [
            str(target / "tokenizer.json"),
            str(target / "tokenizer_config.json"),
            str(target / "special_tokens_map.json"),
        ],
        "weights_file": str(weights_path),
        "weights_bytes": weights_bytes,
        "model_card_file": str(card_path),
        "repo_id": repo_id,
    }
