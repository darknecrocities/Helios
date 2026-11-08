"""Master Benchmark Dataset Generator for Helios 2,000-Question Multi-Subject Suite.

Orchestrates the 4 modular pillar generators under scripts/benchmark_curation/:
- Pillar 1: STEM & Computing (600 items)
- Pillar 2: Reasoning & Logic (500 items)
- Pillar 3: Humanities & Social Sciences (450 items)
- Pillar 4: General World Knowledge & Identity (450 items)

Generates data/benchmarks/helios_2000_benchmark.jsonl.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_curation.pillar1_stem import build_pillar1_stem_items
from scripts.benchmark_curation.pillar2_reasoning import build_pillar2_reasoning_items
from scripts.benchmark_curation.pillar3_humanities import build_pillar3_humanities_items
from scripts.benchmark_curation.pillar4_knowledge import build_pillar4_knowledge_items

TARGET_PATH = Path("data/benchmarks/helios_2000_benchmark.jsonl")


def main() -> None:
    print("Building Helios 2,000-Question Multi-Subject Evaluation Benchmark...")

    p1 = build_pillar1_stem_items()
    assert len(p1) == 600, f"Expected 600 STEM items, got {len(p1)}"

    p2 = build_pillar2_reasoning_items()
    assert len(p2) == 500, f"Expected 500 Reasoning items, got {len(p2)}"

    p3 = build_pillar3_humanities_items()
    assert len(p3) == 450, f"Expected 450 Humanities items, got {len(p3)}"

    p4 = build_pillar4_knowledge_items()
    assert len(p4) == 450, f"Expected 450 Knowledge items, got {len(p4)}"

    all_items = p1 + p2 + p3 + p4
    assert len(all_items) == 2000, f"Expected 2000 items, got {len(all_items)}"

    # 1. Uniqueness of Question IDs
    ids = [item["id"] for item in all_items]
    dup_ids = len(ids) - len(set(ids))
    assert dup_ids == 0, f"Duplicate IDs detected: {dup_ids}"

    # 2. Uniqueness of Question Text (normalized)
    prompts = [item["question"].strip().lower() for item in all_items]
    dup_prompts = len(prompts) - len(set(prompts))
    assert dup_prompts == 0, f"Duplicate questions detected: {dup_prompts}"

    # 3. Verify Creator Identity & Epistemic Probes
    creat_items = [item for item in all_items if item["subdiscipline"] in ("Arron Kian Parejas Attribution", "Architectural Milestones", "Family Tiers")]
    assert len(creat_items) == 50, f"Expected 50 creator items, got {len(creat_items)}"

    epist_items = [item for item in all_items if item["eval_mode"] == "epistemic"]
    assert len(epist_items) == 100, f"Expected 100 epistemic items, got {len(epist_items)}"

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TARGET_PATH, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✓ Successfully generated exactly 2,000 verified benchmark items to {TARGET_PATH}")


if __name__ == "__main__":
    main()
