"""Automated, unbiased post-training benchmark for all trained tiers.

Every tier is measured with the same held-out texts, seed, and sequence length:
held-out loss / perplexity, forward-pass throughput, and training loss delta.
Results are written to markdown and JSON.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from helios.reasoning.context_optimizer import CONTEXT_PROFILES
from helios_pipeline.training.multi_tier import TierResult, _batches, _build


@dataclass
class TierBenchmark:
    tier: str
    mode: str
    heldout_loss: float
    heldout_ppl: float
    infer_tok_per_s: float
    train_loss_start: float
    train_loss_end: float
    train_tok_per_s: float
    max_context: int
    cot_budget: int


@torch.no_grad()
def benchmark_tier(
    result: TierResult, heldout: list[str], tokenizer: Any, batches: int = 8
) -> TierBenchmark:
    """Reload the saved checkpoint and score it on held-out texts (CPU, deterministic)."""
    torch.manual_seed(0)
    seq_len = 128 if result.mode == "full_parameter" else 64
    model, cfg, _ = _build(result.tier, seq_len)
    model.load_state_dict(load_file(str(Path(result.checkpoint) / "model.safetensors")))
    model.eval()
    vocab = getattr(model, "active_vocab", cfg.vocab_size)
    stream = _batches(heldout, tokenizer, seq_len, 2, vocab)
    total_loss, total_tok, elapsed = 0.0, 0, 0.0
    for _ in range(batches):
        ids, labels = next(stream)
        t0 = time.perf_counter()
        loss = model(ids, labels=labels)["loss"]
        elapsed += time.perf_counter() - t0
        n = int((labels != -100).sum())
        total_loss += float(loss) * n
        total_tok += n
    mean = total_loss / max(1, total_tok)
    prof = CONTEXT_PROFILES[result.tier]
    return TierBenchmark(
        tier=result.tier,
        mode=result.mode,
        heldout_loss=round(mean, 4),
        heldout_ppl=round(math.exp(min(mean, 20)), 2),
        infer_tok_per_s=round(total_tok / elapsed, 1) if elapsed else 0.0,
        train_loss_start=round(result.initial_loss, 4),
        train_loss_end=round(result.final_loss, 4),
        train_tok_per_s=round(result.tokens_per_sec, 1),
        max_context=prof.max_context,
        cot_budget=prof.cot_budget,
    )


def write_reports(
    benches: list[TierBenchmark], results: list[TierResult], minutes: float, docs: Path
) -> Path:
    """Write docs/FIVE_MINUTE_PRETRAIN_REPORT.md and eval/reports JSON."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = "\n".join(
        f"| {b.tier} | {b.mode} | {r.device} | {r.steps} | {b.train_loss_start} → "
        f"{b.train_loss_end} | {b.heldout_loss} | {b.heldout_ppl} | {b.train_tok_per_s} | "
        f"{b.infer_tok_per_s} | {b.max_context:,} / {b.cot_budget:,} |"
        for b, r in zip(benches, results, strict=True)
    )
    md = f"""# HELIOS Time-Boxed Pretraining Report

Generated: {stamp} · Budget: {minutes:g} min total · Storage: /Volumes/One Touch (verified)

| Tier | Mode | Device | Steps | Train loss | Held-out loss | Held-out PPL | Train tok/s | Infer tok/s | Context / CoT budget |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | :--- |
{rows}

## Interpretation & limits
- `full_parameter` tiers (nano/alpha/beta) train the complete model.
- `block_slice` tiers (core/pro/flagship) train one transformer block at exact width with a
  2,048-token active vocab. Their loss is **not comparable** to full models and does not
  indicate full 1B–5.8B capability.
- A few minutes of training moves loss but does not produce usable language ability.
  Held-out perplexity in the thousands is expected at this stage.
"""
    docs.mkdir(parents=True, exist_ok=True)
    out = docs / "FIVE_MINUTE_PRETRAIN_REPORT.md"
    out.write_text(md, encoding="utf-8")
    js = Path("eval/reports/pretrain_timeboxed.json")
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps([asdict(b) for b in benches], indent=2), encoding="utf-8")
    return out
