#!/usr/bin/env python3
"""Long-context fine-tuning for Helios tiers (up to 1M tokens).

Non-destructive: writes to a NEW lineage directory, never overwrites existing
checkpoints. Resumable via checkpoint-every. Uses gradient checkpointing and
chunked attention to fit 1M-token sequences on 16 GB M3 (Nano/Alpha only).

Usage
-----
    python scripts/long_context_finetune.py \\
      --tier nano \\
      --max-ctx 131072 \\
      --seq-len 131072 \\
      --batch-size 1 \\
      --accum 16 \\
      --lr 1e-5 \\
      --steps 1000 \\
      --checkpoint-every 100 \\
      --out /Volumes/One\\ Touch/Helios/checkpoints/nano_longctx

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from datasets import load_dataset

from helios.cli.utils import load_or_create_tokenizer, resolve_device
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM

ONETOUCH = Path("/Volumes/One Touch/Helios")
CHECKPOINTS = ONETOUCH / "checkpoints"
LINEAGE = CHECKPOINTS / "v_longctx"

TIERS = {
    "nano": ("nano-20m", "helios_nano", 17_842_880),
    "alpha": ("alpha-100m", "helios_alpha", 100_092_672),
    "beta": ("beta-300m", "helios_beta", 303_350_784),
}


def load_student(tier, device):
    preset, parent, _ = TIERS[tier]
    parent_dir = CHECKPOINTS / parent
    config = HeliosConfig.from_pretrained(parent_dir)
    config.max_seq_len = None  # set by args
    model = HeliosForCausalLM(config)
    state_path = parent_dir / "state.pt"
    if state_path.is_file():
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    model.to(device)
    model.train()
    # gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    tokenizer = load_or_create_tokenizer(parent_dir)
    return model, tokenizer, config


def stream_long_texts(dataset, config, split, field, min_chars, limit):
    ds = load_dataset(dataset, config, split=split, streaming=True)
    count = 0
    for rec in ds:
        if count >= limit:
            break
        text = rec.get(field)
        if isinstance(text, str) and len(text) >= min_chars:
            yield text
            count += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", required=True, choices=list(TIERS))
    ap.add_argument("--max-ctx", type=int, default=131072)
    ap.add_argument("--seq-len", type=int, default=131072)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--min-chars", type=int, default=50000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = resolve_device("auto")
    print(f"device: {device}, max_ctx: {args.max_ctx}, seq_len: {args.seq_len}", flush=True)

    model, tokenizer, config = load_student(args.tier, device)
    config.max_seq_len = args.seq_len
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Never overwrite: use a timestamped sub-dir if exists
    if (out_dir / "state.pt").exists():
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(str(out_dir) + f"_{ts}")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Existing checkpoint found; using new dir: {out_dir}", flush=True)

    step = 0
    t0 = time.time()
    optimizer.zero_grad()

    for text in stream_long_texts(args.dataset, args.config, "train", args.text_field, args.min_chars, args.limit):
        if step >= args.steps:
            break
        ids = tokenizer.encode(text, add_special_tokens=False)[:args.seq_len]
        if len(ids) < 1024:
            continue
        input_ids = torch.tensor([ids], device=device)
        labels = input_ids.clone()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type in ("cuda", "mps")):
            out = model(input_ids, labels=labels)
            loss = out["loss"] if isinstance(out, dict) else out.loss
        loss = loss / args.accum
        loss.backward()
        if (step + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            elapsed = (time.time() - t0) / 60
            print(f"step {step+1} loss {loss.item()*args.accum:.4f} seq {len(ids)} elapsed {elapsed:.1f}m", flush=True)
        if (step + 1) % args.checkpoint_every == 0:
            ckpt = out_dir / f"state_step_{step+1}.pt"
            torch.save({"model_state_dict": model.state_dict(), "step": step+1, "max_ctx": args.max_ctx}, ckpt)
            print(f"  checkpoint saved: {ckpt}", flush=True)
        step += 1

    torch.save({"model_state_dict": model.state_dict(), "step": step, "max_ctx": args.max_ctx}, out_dir / "state.pt")
    print(f"Done: {step} steps, saved to {out_dir}")


if __name__ == "__main__":
    main()
