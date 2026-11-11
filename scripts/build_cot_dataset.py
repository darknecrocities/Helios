#!/usr/bin/env python3
"""Build a deduplicated chain-of-thought SFT dataset from streaming HF sources.

Sources
-------
- ``open-thoughts/OpenThoughts-114k``: mixed code/math reasoning traces.
- ``AI-MO/NuminaMath-CoT``: math problems with step-by-step solutions.
- ``open-r1/OpenR1-Math-220k``: math reasoning traces with final answers.

Every record is normalized into:

    prompt, reasoning, answer, text, source

where ``text`` uses the Helios chat format and a ``<thought>...</thought>``
block. Prompt-level exact and MinHash deduplication removes the same problem
appearing in multiple datasets. Prompts that overlap GSM8K test, MATH-500 test,
or Helios benchmark questions are rejected as contaminated.

Usage
-----
    python scripts/build_cot_dataset.py \
        --out-dir "/Volumes/One Touch/Helios/datasets/helios_cot_v1" \
        --open-thoughts-limit 20000 \
        --numina-limit 20000 \
        --openr1-limit 10000

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from scripts.build_pretrain_shards import (
    DedupStore,
    EvalContaminationIndex,
    _hash64,
    minhash_signature,
    normalize_text,
)

BEGIN_THOUGHT = "<|begin_of_thought|>"
END_THOUGHT = "<|end_of_thought|>"
BEGIN_SOLUTION = "<|begin_of_solution|>"
END_SOLUTION = "<|end_of_solution|>"
BOXED_RE = re.compile(r"\\boxed\{")
CODE_BLOCK_RE = re.compile(r"```(?:python|py|python3)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
FINAL_RE = re.compile(
    r"(?:final answer|the answer is|answer\s*[:：])\s*(.+)$", re.IGNORECASE | re.DOTALL
)


@dataclass
class CotStats:
    """Build counters."""

    seen: int = 0
    accepted: int = 0
    train: int = 0
    val: int = 0
    reasoning_tokens_chars: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    per_source: dict[str, dict[str, int]] = field(default_factory=dict)

    def reject(self, source: str, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1
        counters = self.per_source.setdefault(source, {"seen": 0, "accepted": 0, "rejected": 0})
        counters["rejected"] += 1


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def extract_boxed(text: str) -> str | None:
    """Extract the last balanced ``\\boxed{...}`` expression."""
    matches = list(BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    index = start
    while index < len(text) and depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    if depth:
        return None
    value = text[start : index - 1].strip()
    return value or None


def extract_final_answer(text: str) -> str | None:
    """Extract a final answer from boxed notation or the last non-empty line."""
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    match = FINAL_RE.search(text)
    if match:
        value = match.group(1).strip().splitlines()[0].strip()
        if value:
            return value
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1][:1000]
    return None


def extract_code_block(text: str) -> str | None:
    """Extract the last fenced code block, which is the answer for code tasks."""
    blocks = CODE_BLOCK_RE.findall(text)
    if not blocks:
        return None
    code = blocks[-1].strip()
    return code or None


def split_open_thoughts(content: str) -> tuple[str, str]:
    """Split an OpenThoughts assistant message into reasoning and solution."""
    reasoning = ""
    solution = ""
    if BEGIN_THOUGHT in content and END_THOUGHT in content:
        reasoning = content.split(BEGIN_THOUGHT, 1)[1].split(END_THOUGHT, 1)[0].strip()
    if BEGIN_SOLUTION in content and END_SOLUTION in content:
        solution = content.split(BEGIN_SOLUTION, 1)[1].split(END_SOLUTION, 1)[0].strip()
    if not solution and END_THOUGHT in content:
        solution = content.split(END_THOUGHT, 1)[1].strip()
    if not reasoning:
        reasoning = content
    if not solution:
        solution = content
    return reasoning, solution


def conversations_to_parts(record: dict[str, Any]) -> tuple[str, str]:
    """Extract the first user prompt and concatenated assistant content."""
    conversations = record.get("conversations") or []
    prompts: list[str] = []
    answers: list[str] = []
    for turn in conversations:
        role = str(turn.get("from", turn.get("role", ""))).lower()
        value = str(turn.get("value", turn.get("content", "")))
        if role == "user":
            prompts.append(value)
        elif role == "assistant":
            answers.append(value)
    return "\n\n".join(prompts).strip(), "\n\n".join(answers).strip()


def format_record(prompt: str, reasoning: str, answer: str) -> str:
    """Render a Helios chat record with a visible thought block."""
    return (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"<thought>\n{reasoning}\n</thought>\n{answer}<|eot_id|>"
    )


def source_records(name: str, config: str | None, split: str, limit: int) -> Iterator[dict[str, Any]]:
    """Stream a Hugging Face dataset source."""
    from datasets import load_dataset

    dataset = load_dataset(name, config, split=split, streaming=True)
    for index, record in enumerate(dataset):
        if index >= limit:
            break
        yield record


def parse_record(source: str, record: dict[str, Any]) -> tuple[str, str, str] | None:
    """Normalize one source record into (prompt, reasoning, answer)."""
    if source == "open-thoughts":
        prompt, content = conversations_to_parts(record)
        if not content:
            return None
        reasoning, solution = split_open_thoughts(content)
        answer = extract_code_block(solution) or extract_final_answer(solution) or solution[:1000]
        return prompt, reasoning, answer
    if source == "numina":
        prompt = str(record.get("problem", "")).strip()
        solution = str(record.get("solution", "")).strip()
        answer = extract_final_answer(solution) or ""
        return prompt, solution, answer
    if source == "openr1":
        prompt = str(record.get("problem", "")).strip()
        reasoning = str(record.get("solution", "")).strip()
        answer = str(record.get("answer", "")).strip() or (extract_final_answer(reasoning) or "")
        return prompt, reasoning, answer
    return None


def quality_reason(prompt: str, reasoning: str, answer: str) -> str | None:
    """Reject malformed or low-information CoT records."""
    if len(prompt) < 10:
        return "prompt_too_short"
    if len(prompt) > 20_000:
        return "prompt_too_long"
    if len(reasoning) < 100:
        return "reasoning_too_short"
    if len(reasoning) > 40_000:
        return "reasoning_too_long"
    if not answer or len(answer) > 8_000:
        return "bad_answer"
    if len(prompt) + len(reasoning) + len(answer) > 48_000:
        return "record_too_long"
    return None


def load_eval_prompts() -> list[str]:
    """Load GSM8K test, MATH-500 test, and Helios benchmark prompts."""
    from datasets import load_dataset

    texts: list[str] = []
    try:
        gsm = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
        for record in gsm:
            texts.append(str(record.get("question", "")))
    except Exception as exc:
        print(f"warning: GSM8K contamination index unavailable: {exc}", flush=True)
    try:
        math500 = load_dataset("HuggingFaceH4/MATH-500", split="test", streaming=True)
        for record in math500:
            texts.append(str(record.get("problem", "")))
    except Exception as exc:
        print(f"warning: MATH-500 contamination index unavailable: {exc}", flush=True)
    try:
        from helios.evaluation.benchmark_500 import ALL_500_QUESTIONS

        for question in ALL_500_QUESTIONS:
            texts.append(str(getattr(question, "prompt", "")))
    except Exception:
        pass
    return [text for text in texts if text.strip()]


def build(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dedup = DedupStore(out_dir / "dedup.sqlite")
    state = dedup.load_state()
    stats = CotStats(**state.get("stats", {}))
    train_path = out_dir / "cot_train.jsonl"
    val_path = out_dir / "cot_val.jsonl"
    train_handle = open(train_path, "a", encoding="utf-8")
    val_handle = open(val_path, "a", encoding="utf-8")

    eval_prompts = load_eval_prompts()
    contamination = EvalContaminationIndex(eval_prompts) if eval_prompts else None
    print(
        f"contamination index: {len(eval_prompts)} prompts, "
        f"{len(contamination.hashes) if contamination else 0} 13-grams",
        flush=True,
    )

    sources = [
        ("open-thoughts", "open-thoughts/OpenThoughts-114k", None, "train", args.open_thoughts_limit),
        ("numina", "AI-MO/NuminaMath-CoT", None, "train", args.numina_limit),
        ("openr1", "open-r1/OpenR1-Math-220k", None, "train", args.openr1_limit),
    ]
    start_source = int(state.get("source_index", 0))
    start_offset = int(state.get("source_offset", 0))
    started = time.time()

    for source_index, (tag, name, config, split, limit) in enumerate(sources):
        if source_index < start_source:
            continue
        offset = start_offset if source_index == start_source else 0
        print(f"\n=== {tag}: {name} limit={limit:,} offset={offset:,} ===", flush=True)
        for index, record in enumerate(source_records(name, config, split, limit)):
            if index < offset:
                continue
            stats.seen += 1
            counters = stats.per_source.setdefault(tag, {"seen": 0, "accepted": 0, "rejected": 0})
            counters["seen"] += 1
            parsed = parse_record(tag, record)
            if parsed is None:
                stats.reject(tag, "parse_failed")
                continue
            prompt, reasoning, answer = parsed
            prompt = normalize_text(prompt)
            reasoning = normalize_text(reasoning)
            answer = normalize_text(answer)
            reason = quality_reason(prompt, reasoning, answer)
            if reason:
                stats.reject(tag, reason)
                continue
            if contamination and contamination.is_contaminated(prompt):
                stats.reject(tag, "benchmark_contamination")
                continue

            exact = hashlib.blake2b(prompt.encode("utf-8"), digest_size=16).digest()
            signature = minhash_signature(prompt)
            duplicate = dedup.is_duplicate(exact, signature, args.near_duplicate_threshold)
            if duplicate:
                stats.reject(tag, f"duplicate_{duplicate}")
                continue

            record_out = {
                "source": tag,
                "prompt": prompt,
                "reasoning": reasoning,
                "answer": answer,
                "text": format_record(prompt, reasoning, answer),
                "sha256": exact.hex(),
            }
            if _hash64(exact) % args.val_every == 0:
                val_handle.write(json.dumps(record_out, ensure_ascii=False) + "\n")
                stats.val += 1
            else:
                train_handle.write(json.dumps(record_out, ensure_ascii=False) + "\n")
                stats.train += 1
            stats.accepted += 1
            stats.reasoning_tokens_chars += len(reasoning)
            counters["accepted"] += 1
            dedup.add(exact, signature)

            if stats.accepted % args.checkpoint_every == 0:
                train_handle.flush()
                val_handle.flush()
                dedup.save_state(
                    {
                        "source_index": source_index,
                        "source_offset": index + 1,
                        "stats": stats.__dict__,
                        "updated_at": utc_now(),
                    }
                )
                dedup.commit()
                print(
                    f"  seen={stats.seen:,} accepted={stats.accepted:,} "
                    f"train={stats.train:,} val={stats.val:,} "
                    f"elapsed={(time.time() - started) / 60:.1f}m",
                    flush=True,
                )

        dedup.save_state(
            {
                "source_index": source_index + 1,
                "source_offset": 0,
                "stats": stats.__dict__,
                "updated_at": utc_now(),
            }
        )
        dedup.commit()
        print(f"  source done: accepted={counters['accepted']:,}", flush=True)

    train_handle.close()
    val_handle.close()
    dedup.close()

    manifest = {
        "dataset": "helios_cot_v1",
        "created_at": utc_now(),
        "sources": [
            {"tag": tag, "name": name, "config": config, "split": split, "limit": limit}
            for tag, name, config, split, limit in sources
        ],
        "counts": {
            "seen": stats.seen,
            "accepted": stats.accepted,
            "train": stats.train,
            "val": stats.val,
            "reasoning_chars": stats.reasoning_tokens_chars,
        },
        "rejected": stats.rejected,
        "per_source": stats.per_source,
        "filters": {
            "min_prompt_chars": 10,
            "min_reasoning_chars": 100,
            "max_reasoning_chars": 40_000,
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "val_every": args.val_every,
            "contamination": "GSM8K test + MATH-500 test + Helios benchmarks, 13-word n-grams",
        },
        "format": "Helios chat template with <thought>...</thought> before the final answer",
        "not_a_capability_claim": "CoT SFT data does not by itself establish strong chain-of-thought.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_data_card(out_dir, manifest)
    return manifest


def write_data_card(out_dir: Path, manifest: dict[str, Any]) -> None:
    """Write a human-readable CoT data card."""
    counts = manifest["counts"]
    lines = [
        "# Helios CoT v1 — Data Card",
        "",
        f"Created: {manifest['created_at']}",
        "",
        "Chain-of-thought SFT data in the Helios chat format. Not a model release.",
        "",
        "## Counts",
        "",
        f"- Seen: {counts['seen']:,}",
        f"- Accepted: {counts['accepted']:,}",
        f"- Train: {counts['train']:,}",
        f"- Validation: {counts['val']:,}",
        f"- Reasoning characters: {counts['reasoning_chars']:,}",
        "",
        "## Sources",
        "",
        "| Tag | Dataset | Limit | Seen | Accepted |",
        "|---|---|---:|---:|---:|",
    ]
    for source in manifest["sources"]:
        stats = manifest["per_source"].get(source["tag"], {})
        lines.append(
            f"| {source['tag']} | {source['name']} | {source['limit']:,} | "
            f"{stats.get('seen', 0):,} | {stats.get('accepted', 0):,} |"
        )
    lines += [
        "",
        "## Gates",
        "",
        "- Prompt exact dedup: BLAKE2b-128.",
        "- Prompt near dedup: 16-hash MinHash, 4 bands × 4 hashes.",
        "- Contamination: 13-word overlap with GSM8K test, MATH-500 test, and Helios benchmarks.",
        "- Quality: prompt/reasoning/answer length and presence checks.",
        "",
        "## Non-claims",
        "",
        "This dataset contains reasoning traces, but training on it does not guarantee strong "
        "chain-of-thought. Accuracy and format adherence must be measured on held-out benchmarks.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--open-thoughts-limit", type=int, default=20_000)
    parser.add_argument("--numina-limit", type=int, default=20_000)
    parser.add_argument("--openr1-limit", type=int, default=10_000)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()

    manifest = build(args)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"\nManifest: {Path(args.out_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
