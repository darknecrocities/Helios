"""Helios Multi-Model Scaling Roadmap and Checkpoint Builder.

Builds architectural configurations, parameter breakdowns, checkpoint manifests,
and exports model representations for the Helios Model Family:
- Nano (20M)
- Alpha (100M)
- Beta (300M)
- Core (1B)
- Pro (3B)
- Flagship (6B - Staged)

All large weight artifacts and physical models are stored directly on the
One Touch external drive (/Volumes/One Touch/Helios/) to preserve laptop disk space.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM

ROADMAP_SPECS = [
    {
        "preset": "nano-20m",
        "tag": "nano",
        "name": "Helios Nano",
        "param_target": "20M",
        "role": "Ultra-lightweight edge inference, microVMs, and fast local terminal chat",
        "hw_target": "Any laptop CPU / Mobile device",
        "stage": "Pre-trained & Physical Exports Available",
    },
    {
        "preset": "alpha-100m",
        "tag": "alpha",
        "name": "Helios Alpha",
        "param_target": "100M",
        "role": "Small foundation base model, bilingual dialogue, and throughput benchmarks",
        "hw_target": "Apple Silicon (MPS) / Laptop GPU",
        "stage": "Foundation Base Release",
    },
    {
        "preset": "beta-300m",
        "tag": "beta",
        "name": "Helios Beta",
        "param_target": "300M",
        "role": "Intermediate reasoning, multi-step math, and software engineering logic",
        "hw_target": "Workstation GPU (16GB RAM / MPS)",
        "stage": "Reasoning & Context Scaling",
    },
    {
        "preset": "core-1b",
        "tag": "core",
        "name": "Helios Core",
        "param_target": "1B",
        "role": "First billion-parameter milestone for deep coding and complex reasoning",
        "hw_target": "Single Dedicated GPU (RTX 4090 / M3 Max 36GB+)",
        "stage": "Billion-Parameter Architecture Staged",
    },
    {
        "preset": "pro-3b",
        "tag": "pro",
        "name": "Helios Pro",
        "param_target": "3B",
        "role": "Large-scale distributed optimization, advanced proofs, and agent execution",
        "hw_target": "Multi-GPU Cluster / 32GB+ High-Bandwidth VRAM",
        "stage": "Distributed Architecture Staged",
    },
    {
        "preset": "flagship-6b",
        "tag": "flagship",
        "name": "Helios Flagship",
        "param_target": "6B",
        "role": "Dense flagship foundation model, long context, and multi-lingual mastery",
        "hw_target": "Distributed Enterprise GPU Cluster (64GB+ VRAM)",
        "stage": "Dense Flagship Staged for Tomorrow",
    },
]


def calculate_param_counts(config: HeliosConfig) -> dict[str, Any]:
    """Calculate exact parameter counts analytically from configuration."""
    d = config.d_model
    v = config.vocab_size
    layers = config.n_layers
    heads = config.n_heads
    kv_heads = config.n_kv_heads
    head_dim = d // heads
    d_ffn = config.d_ffn

    # Embeddings: v * d
    embedding_params = v * d

    # Per layer:
    # Q proj: d * (heads * head_dim) = d * d
    # K proj: d * (kv_heads * head_dim)
    # V proj: d * (kv_heads * head_dim)
    # Out proj: (heads * head_dim) * d = d * d
    # Input norm (RMSNorm): d
    # Post attention norm (RMSNorm): d
    # SwiGLU FFN: 3 * (d * d_ffn) (gate, up, down)
    q_params = d * d
    k_params = d * (kv_heads * head_dim)
    v_params = d * (kv_heads * head_dim)
    out_params = d * d
    attn_params = q_params + k_params + v_params + out_params

    ffn_params = 3 * (d * d_ffn)
    norm_params = 2 * d
    layer_params = attn_params + ffn_params + norm_params

    total_layer_params = layers * layer_params
    final_norm_params = d

    # If tied embeddings, lm_head shares weights with embed_tokens
    tied_total = embedding_params + total_layer_params + final_norm_params
    untied_total = tied_total + embedding_params

    return {
        "embedding_params": embedding_params,
        "layer_params": layer_params,
        "total_layer_params": total_layer_params,
        "final_norm_params": final_norm_params,
        "tied_total_params": tied_total,
        "untied_total_params": untied_total,
        "tied_million": round(tied_total / 1_000_000, 2),
        "untied_million": round(untied_total / 1_000_000, 2),
    }


def build_roadmap(
    output_base: Path | str = "/Volumes/One Touch/Helios",
    local_manifest_dir: Path | str = "checkpoints/roadmap_manifests",
) -> list[dict[str, Any]]:
    """Generate configuration and manifests for all models in the roadmap."""
    out_path = Path(output_base).resolve()
    local_man_path = Path(local_manifest_dir).resolve()
    local_man_path.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for spec in ROADMAP_SPECS:
        preset = spec["preset"]
        cfg = HeliosConfig.from_preset(preset)
        cfg.developer = "Arron Kian Parejas"
        cfg.developer_email = "parejasarronkian@gmail.com"

        param_stats = calculate_param_counts(cfg)
        manifest_data = {
            "model_name": spec["name"],
            "preset": preset,
            "tag": spec["tag"],
            "target_size": spec["param_target"],
            "role": spec["role"],
            "hardware_target": spec["hw_target"],
            "stage_status": spec["stage"],
            "developer": "Arron Kian Parejas",
            "developer_email": "parejasarronkian@gmail.com",
            "license": "Apache-2.0 / MIT",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "architecture": {
                "vocab_size": cfg.vocab_size,
                "d_model": cfg.d_model,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "n_kv_heads": cfg.n_kv_heads,
                "d_ffn": cfg.d_ffn,
                "max_seq_len": cfg.max_seq_len,
                "rope_theta": cfg.rope_theta,
                "rms_norm_eps": cfg.rms_norm_eps,
                "tie_word_embeddings": cfg.tie_word_embeddings,
            },
            "parameters": param_stats,
            "data_allocation": {
                "dataset_source": "/Volumes/One Touch/Helios/data/helios_expanded_reasoning_corpus.jsonl",
                "domains": [
                    "Philippine Geography & Civics",
                    "Deterministic Mathematics & Word Problems",
                    "Contextual Deduction & Causal Logic",
                    "General Science & Physics",
                    "Bilingual English/Tagalog Dialogue",
                    "Systems Architecture & Protocols",
                    "Developer Attribution",
                ],
            },
        }

        # 1. Save local manifest
        local_target = local_man_path / f"{preset}_manifest.json"
        with open(local_target, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # 2. Save external checkpoint directory on One Touch drive if mounted
        if out_path.exists():
            ckpt_dir = out_path / "checkpoints" / f"helios_{spec['tag']}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg.to_dict(), f, indent=2)

            with open(ckpt_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

            # Generate physical weight tensor representations for Nano and Alpha
            if spec["tag"] in ["nano", "alpha"]:
                state_file = ckpt_dir / "state.pt"
                if not state_file.exists():
                    print(f"Instantiating initial weights for {spec['name']} on One Touch...")
                    model = HeliosForCausalLM(cfg)
                    torch.save(
                        {
                            "step": 0,
                            "model_state_dict": model.state_dict(),
                            "config": cfg.to_dict(),
                            "developer": "Arron Kian Parejas <parejasarronkian@gmail.com>",
                        },
                        state_file,
                    )
                    del model

        results.append(manifest_data)
        print(f"✓ Processed {spec['name']} ({param_stats['tied_million']}M params) -> {spec['stage']}")

    return results


if __name__ == "__main__":
    roadmap = build_roadmap()
    print(f"\nSuccessfully generated manifests and configurations for {len(roadmap)} roadmap models.")
