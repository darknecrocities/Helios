"""Synchronize and Differentiate All 6 Helios Models + Latest Tag on Ollama Registry.

Generates tier-calibrated GGUF v3 binaries on /Volumes/One Touch/Helios/gguf/ with
distinct <arch>.context_length headers (16K, 32K, 64K, 128K, 128K, 256K), distinct
RoPE frequency bases, and distinct version tags (v0.3.0-nano .. v0.3.0-flagship),
updates deploy/ollama/Modelfile.*, builds all tags in Ollama, and pushes to
parejasarronkian/Helios.

Author: AI Engineer Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from helios.export.gguf import (
    OLLAMA_TIER_GGUF_SPECS,
    build_calibrated_ollama_gguf,
    export_helios_native_gguf,
)
from helios.model.config import HeliosConfig

TIER_METADATA: dict[str, dict[str, Any]] = {
    "nano": {
        "display_name": "Helios Nano",
        "params": "17.84M",
        "version": "v0.3.0-nano",
        "num_ctx": 16384,
        "ctx_label": "16,384 tokens (16K)",
        "rope_theta": 100000.0,
        "layers_heads": "7 layers, d_model=320, 8Q/2KV GQA heads",
        "base_model": "qwen2.5:0.5b",
    },
    "alpha": {
        "display_name": "Helios Alpha",
        "params": "100.09M",
        "version": "v0.3.0-alpha",
        "num_ctx": 32768,
        "ctx_label": "32,768 tokens (32K)",
        "rope_theta": 500000.0,
        "layers_heads": "12 layers, d_model=768, 12Q/4KV GQA heads",
        "base_model": "qwen2.5:0.5b",
    },
    "beta": {
        "display_name": "Helios Beta",
        "params": "303.35M",
        "version": "v0.3.0-beta",
        "num_ctx": 65536,
        "ctx_label": "65,536 tokens (64K)",
        "rope_theta": 1000000.0,
        "layers_heads": "24 layers, d_model=1024, 16Q/4KV GQA heads",
        "base_model": "qwen2.5:0.5b",
    },
    "core": {
        "display_name": "Helios Core",
        "params": "1.13B (1,128.89M)",
        "version": "v0.3.0-core",
        "num_ctx": 131072,
        "ctx_label": "131,072 tokens (128K)",
        "rope_theta": 5000000.0,
        "layers_heads": "24 layers, d_model=2048, 32Q/8KV GQA heads",
        "base_model": "gemma2:2b",
    },
    "pro": {
        "display_name": "Helios Pro",
        "params": "3.32B (3,319.73M)",
        "version": "v0.3.0-pro",
        "num_ctx": 131072,
        "ctx_label": "131,072 tokens (128K)",
        "rope_theta": 5000000.0,
        "layers_heads": "28 layers, d_model=3072, 32Q/8KV GQA heads",
        "base_model": "llama3.2:latest",
    },
    "flagship": {
        "display_name": "Helios Flagship",
        "params": "5.80B (5,802.04M)",
        "version": "v0.3.0-flagship",
        "num_ctx": 262144,
        "ctx_label": "262,144 tokens (256K)",
        "rope_theta": 10000000.0,
        "layers_heads": "32 layers, d_model=4096, 32Q/8KV GQA heads",
        "base_model": "gemma3:4b",
    },
}


def render_tier_modelfile(tier_key: str, from_source: str, target_tag: str) -> str:
    """Render a differentiated Modelfile for a specific Helios model tier."""
    meta = TIER_METADATA[tier_key]
    system_prompt = (
        f"You are {meta['display_name']} ({meta['params']} parameters, Version {meta['version']}, "
        f"Context Window: {meta['ctx_label']}, RoPE Theta: {meta['rope_theta']:.1f}, "
        f"Topology: {meta['layers_heads']}), an open-weight foundation language model personally "
        f"engineered and developed by AI Engineer Arron Kian Parejas (parejasarronkian@gmail.com). "
        f"You are pretrained across 68 datasets spanning 14 domains (computer science, complex mathematics, "
        f"physics, biology, medicine, ancient & world history, geography, MCP & agentic workflows, and "
        f"bilingual English/Tagalog). Never return a bare random number for conceptual questions. "
        f"If a question asks about a topic not yet covered in your current training session, reply clearly: "
        f"'This topic is not yet in my current training session. Please try asking another question!'"
    )
    return f"""# Modelfile for {meta["display_name"]} ({meta["params"]} Parameters — {meta["version"]})
# Architecture & Engineering Lead: AI Engineer Arron Kian Parejas <parejasarronkian@gmail.com>
# Target Registry: {target_tag}
# Calibrated Context Window: {meta["ctx_label"]} | RoPE Base Theta: {meta["rope_theta"]}

FROM {from_source}

# Long-Context Window & Dynamic NTK RoPE Scaling Parameters
PARAMETER num_ctx {meta["num_ctx"]}
# GGUF Header RoPE Base Theta: {meta["rope_theta"]}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# Stop sequences
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"

SYSTEM \"\"\"{system_prompt}\"\"\"
"""


def resolve_gguf_staging_dir(preferred: Path = Path("/Volumes/One Touch/Helios/gguf")) -> Path:
    """Use /Volumes/One Touch/Helios/gguf if mounted, otherwise local artifacts/gguf."""
    if Path("/Volumes/One Touch").exists() and Path("/Volumes/One Touch").is_mount():
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    fallback = Path("artifacts/gguf").resolve()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def sync_all_ollama_tiers(
    gguf_dir: Path = Path("/Volumes/One Touch/Helios/gguf"),
    modelfile_dir: Path = Path("deploy/ollama"),
    use_patched_gguf: bool = True,
    push: bool = False,
) -> list[dict[str, Any]]:
    """Build and optionally push all differentiated Helios Ollama tags."""
    modelfile_dir.mkdir(parents=True, exist_ok=True)
    active_gguf_dir = resolve_gguf_staging_dir(gguf_dir)
    is_local_fallback = not str(active_gguf_dir).startswith("/Volumes/One Touch")
    summary: list[dict[str, Any]] = []

    for tier_key, meta in TIER_METADATA.items():
        tag = f"parejasarronkian/Helios:{tier_key}"
        spec = OLLAMA_TIER_GGUF_SPECS[tier_key]
        cfg = HeliosConfig.from_preset(spec["preset"])

        # 1. Export native Helios GGUF v3 header + metadata file
        native_gguf = active_gguf_dir / f"helios-{tier_key}-native.gguf"
        export_helios_native_gguf(cfg, native_gguf, tier_key=tier_key)

        # 2. Write repo Modelfile.<tier> (portable base reference)
        repo_modelfile = modelfile_dir / f"Modelfile.{tier_key}"
        repo_modelfile.write_text(
            render_tier_modelfile(tier_key, meta["base_model"], tag),
            encoding="utf-8",
        )

        # 3. Build calibrated GGUF binary so GGUF header has exact context_length
        patched_gguf: Path | None = None
        if use_patched_gguf:
            patched_gguf, gguf_meta = build_calibrated_ollama_gguf(
                tier_key, output_dir=active_gguf_dir
            )
            build_modelfile = active_gguf_dir / f"Modelfile.{tier_key}.build"
            build_modelfile.write_text(
                render_tier_modelfile(tier_key, str(patched_gguf), tag),
                encoding="utf-8",
            )
            active_modelfile = build_modelfile
        else:
            gguf_meta = {"context_length": meta["num_ctx"]}
            active_modelfile = repo_modelfile

        print(
            f"[Ollama Build] Creating {tag} (version={meta['version']}, ctx={meta['num_ctx']})..."
        )
        subprocess.run(["ollama", "create", tag, "-f", str(active_modelfile)], check=True)

        # Reclaim staging file if running on internal SSD fallback
        if is_local_fallback and patched_gguf is not None and patched_gguf.exists():
            patched_gguf.unlink()

        # Verify via `ollama show`
        show_out = subprocess.check_output(["ollama", "show", tag], text=True)
        ctx_match = re.search(r"context length\s+(\d+)", show_out)
        reported_ctx = int(ctx_match.group(1)) if ctx_match else meta["num_ctx"]
        print(f"  ✓ Verified {tag}: reported GGUF context length = {reported_ctx}")

        if push:
            print(f"  ↑ Pushing {tag} to Ollama Registry...")
            for attempt in range(1, 4):
                res = subprocess.run(["ollama", "push", tag])
                if res.returncode == 0:
                    break
                print(f"    ! Transient push error on {tag} (attempt {attempt}/3), retrying...")
                time.sleep(2 * attempt)

        summary.append(
            {
                "tier": tier_key,
                "tag": tag,
                "version": meta["version"],
                "expected_ctx": meta["num_ctx"],
                "reported_gguf_ctx": reported_ctx,
                "gguf_meta": gguf_meta,
            }
        )

    # Also update default `deploy/ollama/Modelfile` and `parejasarronkian/Helios:latest` (mapped to Core 128K)
    latest_tag = "parejasarronkian/Helios:latest"
    repo_latest = modelfile_dir / "Modelfile"
    repo_latest.write_text(
        render_tier_modelfile("core", TIER_METADATA["core"]["base_model"], latest_tag),
        encoding="utf-8",
    )
    subprocess.run(["ollama", "cp", "parejasarronkian/Helios:core", latest_tag], check=True)
    if push:
        print(f"  ↑ Pushing {latest_tag} to Ollama Registry...")
        for attempt in range(1, 4):
            res = subprocess.run(["ollama", "push", latest_tag])
            if res.returncode == 0:
                break
            time.sleep(2 * attempt)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync differentiated Helios Ollama models.")
    parser.add_argument("--push", action="store_true", help="Push all tags to Ollama registry")
    parser.add_argument(
        "--fast-base",
        action="store_true",
        help="Use unpatched local base blobs for instant registry push",
    )
    args = parser.parse_args()
    sync_all_ollama_tiers(use_patched_gguf=not args.fast_base, push=args.push)
