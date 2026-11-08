"""Helios Performance & Perplexity Evaluation Suite.

Benchmarks generation speed (tokens/sec), continuous KVCache memory footprint (MB),
and multi-domain cross-entropy perplexity.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

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

DEFAULT_EVAL_TEXT = (
    "The Helios open-weight foundation model ecosystem is architected with Pre-RMSNorm, "
    "Grouped-Query Attention, and Rotary Position Embeddings. "
    "Pinatutunayan ng sistemang ito ang kahusayan sa pamamahala ng memorya at kasanayan. "
    "def fibonacci(n: int) -> int:\n"
    "    if n <= 1:\n"
    "        return n\n"
    "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
)


def benchmark_speed(
    model: HeliosForCausalLM,
    prompt_len: int = 64,
    gen_len: int = 64,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Measure prefill and decode generation latency and throughput."""
    dev = device or next(model.parameters()).device
    model.eval()

    prompt_tokens = torch.randint(
        low=1,
        high=min(1000, model.config.vocab_size),
        size=(1, prompt_len),
        dtype=torch.long,
        device=dev,
    )
    total_seq_len = min(prompt_len + gen_len + 16, model.config.max_seq_len)
    head_dim = model.config.d_model // model.config.n_heads

    kv_caches = [
        KVCache(
            max_batch_size=1,
            max_seq_len=total_seq_len,
            n_kv_heads=model.config.n_kv_heads,
            head_dim=head_dim,
            dtype=next(model.parameters()).dtype,
            device=dev,
        )
        for _ in range(model.config.n_layers)
    ]

    # Warmup pass
    with torch.inference_mode():
        _ = model.forward(prompt_tokens[:, :8], start_pos=0)

    # 1. Prefill benchmark
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.forward(prompt_tokens, start_pos=0, kv_caches=kv_caches)
        next_tok = model._sample_token(out["logits"][:, -1, :], temperature=0.0)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    t_prefill = time.perf_counter() - t0

    # 2. Autoregressive decode benchmark
    t1 = time.perf_counter()
    cur_tok = next_tok
    with torch.inference_mode():
        for step in range(1, gen_len):
            cur_pos = prompt_len + step - 1
            out = model.forward(cur_tok, start_pos=cur_pos, kv_caches=kv_caches)
            cur_tok = model._sample_token(out["logits"][:, 0, :], temperature=0.0)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    t_decode = time.perf_counter() - t1

    prefill_tok_sec = prompt_len / t_prefill if t_prefill > 0 else 0.0
    decode_tok_sec = gen_len / t_decode if t_decode > 0 else 0.0
    decode_ms_per_tok = (t_decode / max(gen_len, 1)) * 1000

    return {
        "prefill_ms": t_prefill * 1000,
        "prefill_tok_sec": prefill_tok_sec,
        "decode_ms_per_tok": decode_ms_per_tok,
        "decode_tok_sec": decode_tok_sec,
        "total_time_s": t_prefill + t_decode,
    }


def benchmark_kv_cache(
    model: HeliosForCausalLM,
    max_seq_len: int = 2048,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Calculate and verify contiguous pre-allocated KV cache footprint."""
    dev = device or next(model.parameters()).device
    head_dim = model.config.d_model // model.config.n_heads

    sample_cache = KVCache(
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        n_kv_heads=model.config.n_kv_heads,
        head_dim=head_dim,
        dtype=next(model.parameters()).dtype,
        device=dev,
    )
    layer_mb = sample_cache.memory_footprint_mb()
    total_mb = layer_mb * model.config.n_layers

    return {
        "layers": model.config.n_layers,
        "n_kv_heads": model.config.n_kv_heads,
        "max_seq_len": max_seq_len,
        "layer_mb": layer_mb,
        "total_mb": total_mb,
    }


def evaluate_perplexity(
    model: HeliosForCausalLM,
    tokenizer: BaseTokenizer,
    text: str = DEFAULT_EVAL_TEXT,
    chunk_size: int = 128,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Compute exact cross-entropy loss and perplexity across text chunks."""
    dev = device or next(model.parameters()).device
    model.eval()

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < 4:
        token_ids = token_ids * 4

    total_loss = 0.0
    total_chunks = 0
    total_evaluated_tokens = 0

    with torch.inference_mode():
        for i in range(0, len(token_ids), chunk_size):
            chunk = token_ids[i : i + chunk_size]
            if len(chunk) < 2:
                continue

            input_tensor = torch.tensor([chunk], dtype=torch.long, device=dev)
            out = model.forward(input_tensor, labels=input_tensor)
            loss_val = float(out["loss"].item())

            total_loss += loss_val * len(chunk)
            total_evaluated_tokens += len(chunk)
            total_chunks += 1

    avg_loss = total_loss / max(total_evaluated_tokens, 1)
    ppl = math.exp(min(avg_loss, 50.0))

    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "evaluated_tokens": float(total_evaluated_tokens),
        "chunks": float(total_chunks),
    }


def run_eval(
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
    prompt_len: int = typer.Option(
        64, "--prompt-len", "-p", help="Synthetic benchmark prompt length"
    ),
    gen_len: int = typer.Option(64, "--gen-len", "-g", help="Generated tokens benchmark length"),
    eval_text: str = typer.Option(
        None, "--eval-text", help="Custom text or file path for perplexity eval"
    ),
    benchmark: str = typer.Option(
        None, "--benchmark", "-b", help="Benchmark suite ('2000' or '500')"
    ),
    tier: str = typer.Option(
        None, "--tier", "-t", help="Helios tier alias (e.g. nano, alpha, beta, core, pro, flagship)"
    ),
    mode: str = typer.Option(
        "fast", "--mode", help="Execution mode ('fast' for HMA Memory Mode or 'neural' for autoregressive decoding)"
    ),
    output: str = typer.Option(
        None, "--output", "-o", help="Custom output directory or file path for evaluation reports"
    ),
    limit: int = typer.Option(
        None, "--limit", "-l", help="Limit number of questions to evaluate"
    ),
) -> None:
    """Benchmark generation speed, KV cache memory footprint, perplexity, or evaluation suites."""
    dev = resolve_device(device)

    # 1. Dispatched benchmark evaluation suites (2,000 or 500 questions)
    if benchmark is not None:
        b_clean = benchmark.strip().lower()
        if b_clean in ("2000", "helios-2000", "2k"):
            from helios.evaluation.benchmark_2000 import load_benchmark_questions
            from helios.evaluation.evaluator_2000 import HeliosEvaluator2000

            eval_tier = tier or preset
            evaluator = HeliosEvaluator2000(
                mode=mode,
                preset=eval_tier,
                checkpoint_path=model_path,
                output_path=output,
                device=device,
            )
            items = None
            if limit is not None:
                all_q = load_benchmark_questions()
                items = all_q[:limit]

            summary = evaluator.run_suite(questions=items)
            fields = [
                ("Benchmark Suite", "Helios 2,000-Question Multi-Subject Evaluation"),
                ("Execution Mode", summary["mode"].upper()),
                ("Model Tier", f"{summary['preset']}"),
                ("Total Questions", f"{summary['total_questions']:,}"),
                ("Passed Questions", f"{summary['passed']:,}"),
                ("Failed Questions", f"{summary['failed']:,}"),
                ("Accuracy", f"{summary['accuracy_percent']:.2f}%"),
                ("Total Wall Time", f"{summary['total_wall_time_s']:.3f} s"),
                ("Average Latency", f"{summary['average_latency_ms']:.2f} ms/question"),
            ]
            typer.echo(format_card("Helios 2,000-Question Benchmark Results", fields))
            typer.echo("\n--- Pillar Breakdown ---")
            for p_name, p_stat in summary.get("pillar_breakdown", {}).items():
                typer.echo(
                    f"  * {p_name}: {p_stat['passed']}/{p_stat['total']} ({p_stat['accuracy_percent']:.1f}%)"
                )
            out_dest = output or "reports/"
            typer.echo(f"\n✓ Reports successfully exported to {out_dest}")
            return

        elif b_clean in ("500", "helios-500", "500q"):
            from helios.evaluation.benchmark_500 import ALL_500_QUESTIONS
            from helios.evaluation.evaluator_500 import HeliosEvaluator500

            eval_tier = tier or preset
            evaluator = HeliosEvaluator500(preset=eval_tier, checkpoint_path=model_path)
            items = ALL_500_QUESTIONS[:limit] if limit is not None else ALL_500_QUESTIONS
            summary = evaluator.run_suite(questions=items)
            tot_q = summary.get("total_questions", summary.get("total", len(items)))
            pas_q = summary.get("passed", 0)
            fai_q = summary.get("failed", 0)
            acc_pct = summary.get("accuracy_percent", summary.get("accuracy_pct", 0.0))
            wall_t = summary.get("total_wall_time_s", summary.get("wall_time_s", 0.0))
            fields = [
                ("Benchmark Suite", "Helios 500-Question Non-Biased Benchmark"),
                ("Model Preset", f"{eval_tier}"),
                ("Total Questions", f"{tot_q:,}"),
                ("Passed Questions", f"{pas_q:,}"),
                ("Failed Questions", f"{fai_q:,}"),
                ("Accuracy", f"{acc_pct:.2f}%"),
                ("Total Wall Time", f"{wall_t:.3f} s"),
            ]
            typer.echo(format_card("Helios 500-Question Benchmark Results", fields))
            return
        else:
            typer.echo(f"Error: Unknown benchmark suite '{benchmark}'. Valid options: '2000', '500'")
            raise typer.Exit(code=1)

    # 2. Standard performance & perplexity evaluation
    typer.echo(f"Benchmarking Helios ({preset}) on {dev}...")

    model, config = load_model_and_config(preset=preset, model_path=model_path, device=dev)
    tokenizer = load_or_create_tokenizer(
        tokenizer_path=tokenizer_path, vocab_size=config.vocab_size
    )

    # 1. Benchmark speed
    speed = benchmark_speed(model=model, prompt_len=prompt_len, gen_len=gen_len, device=dev)

    # 2. Benchmark KV cache footprint
    kv = benchmark_kv_cache(model=model, max_seq_len=config.max_seq_len, device=dev)

    # 3. Benchmark perplexity
    target_text = DEFAULT_EVAL_TEXT
    if eval_text is not None:
        p = Path(eval_text)
        if p.is_file():
            target_text = p.read_text(encoding="utf-8")
        else:
            target_text = eval_text

    ppl_res = evaluate_perplexity(model=model, tokenizer=tokenizer, text=target_text, device=dev)

    # Output formatted report card
    fields = [
        ("Model Preset", f"{preset} ({config.developer})"),
        ("Model Parameters", f"{config.count_parameters()['total_parameters']:,} params"),
        ("Hardware Device", str(dev)),
        ("", ""),
        ("Prefill Latency", f"{speed['prefill_ms']:.2f} ms ({speed['prefill_tok_sec']:.2f} tok/s)"),
        (
            "Decode Latency",
            f"{speed['decode_ms_per_tok']:.2f} ms/token ({speed['decode_tok_sec']:.2f} tok/s)",
        ),
        ("Total Gen Time", f"{speed['total_time_s']:.3f} s"),
        ("", ""),
        ("KV Cache Layers", f"{kv['layers']} layers (GQA 3:1)"),
        ("Max Context Window", f"{kv['max_seq_len']} tokens"),
        ("Total KV Footprint", f"{kv['total_mb']:.4f} MB"),
        ("KV Allocation Strategy", "Contiguous Pre-allocated Fixed Buffers"),
        ("", ""),
        ("Evaluated Tokens", f"{int(ppl_res['evaluated_tokens'])} tokens"),
        ("Cross-Entropy Loss", f"{ppl_res['loss']:.4f}"),
        ("Perplexity (PPL)", f"{ppl_res['perplexity']:.2f}"),
    ]

    typer.echo(format_card("Helios Performance & Perplexity Report", fields))
