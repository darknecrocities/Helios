#!/usr/bin/env python3
"""Distill a Helios student from a (stronger) Hugging Face teacher.

This is the "other way" to make local models capable without full pretraining of
the large tiers: use an open teacher (e.g. Qwen2.5-1.5B-Instruct) and train the
Helios student to match its output distribution on a streamed corpus.

Feasible tiers on a 16 GB M3:
  - Nano (17.8M) and Alpha (100M): full distillation, fp16/bf16.
  - Beta (303M): distillation with gradient checkpointing + accumulation.
  - Core/Pro/Flagship: QLoRA-only adapter distillation (4-bit teacher/student).

Usage
-----
    python scripts/distill_from_teacher.py \\
      --teacher Qwen/Qwen2.5-1.5B-Instruct \\
      --student-tier nano \\
      --dataset HuggingFaceFW/fineweb-edu \\
      --config sample-10BT \\
      --text-field text \\
      --limit 200000 \\
      --seq-len 512 \\
      --batch-size 4 \\
      --accum 8 \\
      --lr 2e-5 \\
      --temperature 2.0 \\
      --alpha 0.7 \\
      --out /Volumes/One\\ Touch/Helios/checkpoints/nano_distilled

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from helios.cli.utils import resolve_device
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM

ONETOUCH = Path("/Volumes/One Touch/Helios")
CHECKPOINTS = ONETOUCH / "checkpoints"

TIERS = {
    "nano": ("nano-20m", "helios_nano"),
    "alpha": ("alpha-100m", "helios_alpha"),
    "beta": ("beta-300m", "helios_beta"),
}


def load_student(tier: str, device: torch.device):
    preset, parent = TIERS[tier]
    parent_dir = CHECKPOINTS / parent
    config = HeliosConfig.from_pretrained(parent_dir)
    model = HeliosForCausalLM(config)
    state_path = parent_dir / "state.pt"
    if state_path.is_file():
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    model.to(device)
    from helios.cli.utils import load_or_create_tokenizer
    tokenizer = load_or_create_tokenizer(parent_dir)
    return model, tokenizer, config


def kd_loss(student_logits, teacher_logits, labels, temperature, alpha):
    ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    t = temperature
    s_logp = F.log_softmax(student_logits / t, dim=-1)
    t_prob = F.softmax(teacher_logits / t, dim=-1)
    kl = F.kl_div(s_logp, t_prob, reduction="batchmean") * (t * t)
    return (1 - alpha) * ce + alpha * kl, ce.detach(), kl.detach()


def stream_texts(name, config, split, field, limit):
    ds = load_dataset(name, config, split=split, streaming=True)
    count = 0
    for rec in ds:
        if count >= limit:
            break
        text = rec.get(field)
        if isinstance(text, str) and text.strip():
            yield text
            count += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student-tier", required=True, choices=list(TIERS))
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = resolve_device("auto")
    print(f"device: {device}", flush=True)

    teacher_tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float16)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student, student_tok, config = load_student(args.student_tier, device)
    student.train()

    if student_tok.vocab_size != teacher_tok.vocab_size:
        raise SystemExit(
            f"Vocab mismatch: teacher {teacher_tok.vocab_size} vs student {student_tok.vocab_size}. "
            "Use a 32k-vocab teacher such as TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    accum_loss = 0.0
    t0 = time.time()
    optimizer.zero_grad()

    for text in stream_texts(args.dataset, args.config, args.split, args.text_field, args.limit):
        if step >= args.max_steps:
            break
        ids = teacher_tok(text, truncation=True, max_length=args.seq_len, return_tensors="pt")["input_ids"].to(device)
        if ids.size(1) < 8:
            continue
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type in ("cuda","mps")):
                t_out = teacher(input_ids=ids)
            t_logits = t_out["logits"] if isinstance(t_out, dict) else t_out.logits
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type in ("cuda","mps")):
            out = student(input_ids=ids)
        s_logits = out["logits"] if isinstance(out, dict) else out.logits
        labels = ids.clone()
        loss, ce, kl = kd_loss(s_logits, t_logits, labels, args.temperature, args.alpha)
        loss = loss / args.accum
        loss.backward()
        accum_loss += loss.item()
        if (step + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            elapsed = (time.time() - t0) / 60
            print(f"step {step+1} loss {accum_loss:.4f} ce {ce:.4f} kl {kl:.4f} elapsed {elapsed:.1f}m", flush=True)
            accum_loss = 0.0
        if (step + 1) % args.checkpoint_every == 0:
            ckpt = out_dir / f"state_step_{step+1}.pt"
            torch.save({"model_state_dict": student.state_dict(), "step": step+1, "teacher": args.teacher}, ckpt)
            print(f"  checkpoint saved: {ckpt}", flush=True)
        step += 1

    import torch as _t
    _t.save({"model_state_dict": student.state_dict(), "step": step, "teacher": args.teacher}, out_dir / "state.pt")
    print(f"Saved distilled student to {out_dir} after {step} steps")


if __name__ == "__main__":
    main()
