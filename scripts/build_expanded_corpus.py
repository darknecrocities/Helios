"""Master Dataset Synthesizer for Helios Expanded Reasoning Corpus.

Aggregates multi-domain reasoning, science, math, Philippines, systems,
and identity pairs into a standardized JSONL pre-training and fine-tuning corpus.
Mirrors directly to One Touch external drive (/Volumes/One Touch/Helios/data/).

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.dataset_builders.dialogue_data import get_dialogue_pairs
from scripts.dataset_builders.identity_data import get_identity_pairs
from scripts.dataset_builders.math_data import get_math_pairs
from scripts.dataset_builders.philippines_data import get_philippines_pairs
from scripts.dataset_builders.reasoning_data import get_reasoning_pairs
from scripts.dataset_builders.science_data import get_science_pairs
from scripts.dataset_builders.systems_data import get_systems_pairs


def collect_all_pairs() -> list[tuple[str, str]]:
    """Aggregate pairs from all domain modules with deduplication."""
    all_pairs: list[tuple[str, str]] = []
    seen_prompts: set[str] = set()

    sources = [
        ("Identity", get_identity_pairs()),
        ("Philippines", get_philippines_pairs()),
        ("Math", get_math_pairs()),
        ("Reasoning", get_reasoning_pairs()),
        ("Science", get_science_pairs()),
        ("Dialogue", get_dialogue_pairs()),
        ("Systems", get_systems_pairs()),
    ]

    # Also load original contextual reasoning pairs if present
    orig_path = Path("data/helios_contextual_reasoning.jsonl")
    if orig_path.exists():
        with open(orig_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    p, r = d.get("prompt"), d.get("response")
                    if p and r:
                        norm = p.strip().lower()
                        if norm not in seen_prompts:
                            seen_prompts.add(norm)
                            all_pairs.append((p.strip(), r.strip()))

    for _, pairs in sources:
        for p, r in pairs:
            norm = p.strip().lower()
            if norm not in seen_prompts:
                seen_prompts.add(norm)
                all_pairs.append((p.strip(), r.strip()))

    return all_pairs


def build_corpus(
    output_path: Path | str = "data/helios_expanded_reasoning_corpus.jsonl",
    external_dir: Path | str = "/Volumes/One Touch/Helios/data",
) -> int:
    """Generate formatted JSONL and synchronize with external One Touch drive."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    pairs = collect_all_pairs()
    records = []

    for user_prompt, assistant_response in pairs:
        formatted_text = (
            f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n{assistant_response}<|eot_id|>"
        )
        record = {
            "prompt": user_prompt,
            "response": assistant_response,
            "text": formatted_text,
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
        }
        records.append(record)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✓ Generated {len(records)} records at {out_file}")

    # Mirror directly to One Touch external drive
    ext_dir = Path(external_dir).resolve()
    if ext_dir.exists():
        ext_target = ext_dir / out_file.name
        shutil.copy2(out_file, ext_target)
        print(f"✓ Mirrored to One Touch Drive at {ext_target}")
    else:
        print(f"! External drive path {ext_dir} not mounted or not found. Local copy kept.")

    return len(records)


if __name__ == "__main__":
    count = build_corpus()
    print(f"Total verified dataset count: {count} high-quality multi-domain records.")
