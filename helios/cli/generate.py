"""Helios Autoregressive Generation Engine & CLI Command.

Provides high-throughput generation with continuous pre-allocated KVCache,
token-by-token streaming, top-k sampling, and telemetry tracking.

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

import sys
import time
from typing import Generator

import torch
import typer

from helios.cli.utils import (
    format_card,
    load_model_and_config,
    load_or_create_tokenizer,
    resolve_device,
)
from helios.model.attention import KVCache
from helios.model.transformer import HeliosForCausalLM
from helios.tokenizer.base import BaseTokenizer


def stream_generate(
    model: HeliosForCausalLM,
    tokenizer: BaseTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_k: int | None = 50,
    device: torch.device | None = None,
    telemetry: dict[str, float] | None = None,
) -> Generator[str, None, list[int]]:
    """Stream tokens autoregressively using pre-allocated contiguous KVCache.

    Yields:
        Decoded text chunks token-by-token.

    Returns:
        Full list of generated token IDs.
    """
    target_device = device or next(model.parameters()).device
    model.eval()

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if not prompt_ids:
        prompt_ids = [0]

    prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
    prompt_len = prompt_tokens.shape[1]
    total_seq_len = min(prompt_len + max_new_tokens + 16, model.config.max_seq_len)
    head_dim = model.config.d_model // model.config.n_heads

    # Pre-allocate contiguous KVCache per layer
    kv_caches = [
        KVCache(
            max_batch_size=1,
            max_seq_len=total_seq_len,
            n_kv_heads=model.config.n_kv_heads,
            head_dim=head_dim,
            dtype=next(model.parameters()).dtype,
            device=target_device,
        )
        for _ in range(model.config.n_layers)
    ]

    eos_token_id = tokenizer.special_tokens.get("<|end_of_text|>")
    if eos_token_id is None:
        eos_token_id = tokenizer.special_tokens.get("<|eot_id|>")

    generated_ids: list[int] = []

    # 1. Prefill Step
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.forward(prompt_tokens, start_pos=0, kv_caches=kv_caches)
        next_logits = out["logits"][:, -1, :]
        next_tok = model._sample_token(next_logits, temperature=temperature, top_k=top_k)

    t_prefill = time.perf_counter() - t0
    tok_id = int(next_tok.item())

    if eos_token_id is not None and tok_id == eos_token_id:
        if telemetry is not None:
            telemetry.update(
                {
                    "prefill_s": t_prefill,
                    "decode_s": 0.0,
                    "tokens_generated": 0,
                    "tokens_per_sec": 0.0,
                    "kv_mem_mb": sum(c.memory_footprint_mb() for c in kv_caches),
                }
            )
        return generated_ids

    generated_ids.append(tok_id)
    tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
    yield tok_str

    # 2. Incremental Decode Steps with O(1) KV caching
    t_decode_start = time.perf_counter()
    cur_tok = next_tok

    with torch.inference_mode():
        for step in range(1, max_new_tokens):
            cur_pos = prompt_len + step - 1
            if cur_pos >= model.config.max_seq_len - 1:
                break

            out = model.forward(cur_tok, start_pos=cur_pos, kv_caches=kv_caches)
            next_logits = out["logits"][:, 0, :]
            cur_tok = model._sample_token(next_logits, temperature=temperature, top_k=top_k)
            tok_id = int(cur_tok.item())

            if eos_token_id is not None and tok_id == eos_token_id:
                break

            generated_ids.append(tok_id)
            tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
            yield tok_str

    t_decode = time.perf_counter() - t_decode_start
    num_gen = len(generated_ids)
    tok_sec = num_gen / t_decode if t_decode > 0 else 0.0
    kv_mb = sum(c.memory_footprint_mb() for c in kv_caches)

    if telemetry is not None:
        telemetry.update(
            {
                "prefill_s": t_prefill,
                "decode_s": t_decode,
                "tokens_generated": float(num_gen),
                "tokens_per_sec": tok_sec,
                "kv_mem_mb": kv_mb,
            }
        )

    return generated_ids


def run_generate(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Input prompt text or code snippet"),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens", "-n", help="Max tokens to generate"),
    temperature: float = typer.Option(
        0.7, "--temperature", "-t", help="Sampling temperature (0=greedy)"
    ),
    top_k: int = typer.Option(50, "--top-k", "-k", help="Top-K sampling threshold (0=disabled)"),
    preset: str = typer.Option("nano-20m", "--preset", "-m", help="Helios model size preset"),
    model_path: str = typer.Option(
        None, "--model-path", help="Path to checkpoint directory or state.pt"
    ),
    tokenizer_path: str = typer.Option(
        None, "--tokenizer-path", help="Path to pretrained tokenizer directory"
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Hardware device (auto, mps, cuda, cpu)"
    ),
    stream: bool = typer.Option(
        True, "--stream/--no-stream", help="Stream generated tokens in real time"
    ),
) -> None:
    """Execute live autoregressive text and code generation with KV cache."""
    dev = resolve_device(device)
    typer.echo(f"Initializing Helios ({preset}) on {dev}...")

    model, config = load_model_and_config(preset=preset, model_path=model_path, device=dev)
    tokenizer = load_or_create_tokenizer(
        tokenizer_path=tokenizer_path, vocab_size=config.vocab_size
    )

    k_val = top_k if top_k > 0 else None
    telemetry: dict[str, float] = {}

    typer.echo(f"\nPrompt: {prompt}")
    typer.echo("Response: ", nl=False)

    generator = stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=k_val,
        device=dev,
        telemetry=telemetry,
    )

    full_response: list[str] = []
    for piece in generator:
        if stream:
            sys.stdout.write(piece)
            sys.stdout.flush()
        full_response.append(piece)

    if not stream:
        sys.stdout.write("".join(full_response))
        sys.stdout.flush()

    sys.stdout.write("\n\n")

    # Display performance summary card
    gen_tokens = int(telemetry.get("tokens_generated", 0))
    tok_sec = telemetry.get("tokens_per_sec", 0.0)
    kv_mb = telemetry.get("kv_mem_mb", 0.0)
    prefill_ms = telemetry.get("prefill_s", 0.0) * 1000
    decode_ms_per_tok = (telemetry.get("decode_s", 0.0) / max(gen_tokens, 1)) * 1000

    summary_fields = [
        ("Model Preset", f"{preset} ({config.developer})"),
        ("Hardware Device", str(dev)),
        ("Tokens Generated", f"{gen_tokens} tokens"),
        ("Prefill Latency", f"{prefill_ms:.2f} ms"),
        ("Decode Latency", f"{decode_ms_per_tok:.2f} ms/token"),
        ("Throughput", f"{tok_sec:.2f} tokens/sec"),
        ("KV Cache Footprint", f"{kv_mb:.4f} MB (O(1) in-place preallocated)"),
    ]
    typer.echo(format_card("Helios Generation Telemetry", summary_fields))
