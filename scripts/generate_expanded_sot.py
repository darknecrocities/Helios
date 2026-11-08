"""Generate Expanded Source of Truth (SOT) Dataset Catalog (160+ Datasets).

Covers all 16 foundational domains with authoritative links, Hugging Face IDs,
and license provenance.

Author: AI Engineer Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent.parent / "sot" / "datasets_catalog.json"
DATASETS_160_PLUS: list[dict[str, str]] = []
if CATALOG_PATH.exists():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        DATASETS_160_PLUS = json.load(f)


def generate_sot_catalog() -> dict:
    catalog = {
        "sot_title": "Helios Foundation Model Single Source of Truth (SOT) Dataset Catalog",
        "version": "0.3.0",
        "developer": "Arron Kian Parejas",
        "developer_email": "parejasarronkian@gmail.com",
        "primary_storage": "/Volumes/One Touch/Helios/datasets/",
        "total_external_datasets": len(DATASETS_160_PLUS),
        "total_domains": 16,
        "active_training_snapshots": [
            {
                "snapshot_id": "helios_pretrain_corpus_v3",
                "storage_path": "/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl",
                "total_records": 120616,
                "estimated_tokens": 90376089,
                "storage_tier": "One Touch Drive Storage Only (0 bytes Laptop SSD)",
                "verification_status": "Pretrained Across All 6 Models & 500-Benchmark Verified",
            },
        ],
        "consolidated_corpora": [
            {
                "name": "Helios Massive Multi-Category Omni-Domain Pretraining Corpus",
                "path": "/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl",
                "records": 120616,
                "tokens": 90376089,
                "breakdown": {
                    "core_omni_domain_knowledge": 1248,
                    "mcp_and_agentic_skills": 4,
                    "coding_and_algorithms": 20016,
                    "python_software_engineering": 18612,
                    "mmlu_history_and_systems": 4465,
                    "biology_genetics_medical_science": 10000,
                    "nature_ecology_earth_science": 11679,
                    "tech_history_places_coding_documentation": 10000,
                    "complex_mathematical_reasoning": 7473,
                    "scientific_research_and_reasoning": 1119,
                    "magicoder_oss_instruct": 15000,
                    "agentic_function_calling": 10000,
                    "pubmed_clinical_qa": 1000,
                    "metamath_advanced_proofs": 10000,
                },
            },
            {
                "name": "Helios In-Repository Omni-Domain Grounding Seed Corpus",
                "path": "data/helios_expanded_reasoning_corpus.jsonl",
                "records": 1248,
            },
        ],
        "datasets": [{"index": idx, **ds} for idx, ds in enumerate(DATASETS_160_PLUS, 1)],
    }
    return catalog


def generate_sot_readme_markdown(cat: dict) -> str:
    """Generate Markdown text for sot/README.md."""
    lines = [
        "# Helios Dataset Single Source of Truth (SOT)",
        "",
        f"**Lead AI Engineer & Architect**: **{cat['developer']}** (`{cat['developer_email']}`)  ",
        f"**Total External Datasets Cataloged & Integrated**: **{cat['total_external_datasets']} Datasets** across **{cat['total_domains']} Foundational Knowledge & Reasoning Domains**  ",
        "**Consolidated Active Training Corpora**: Massive Multi-Category Omni-Domain Pretraining Corpus stored on `/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl`, and seed grounding corpus in `data/helios_expanded_reasoning_corpus.jsonl`",
        "",
        "---",
        "",
        "## 1. Executive Summary of Knowledge Domains in Helios",
        "",
        "| Category # | Domain Category | Datasets Count | Primary Focus & Provenance | Storage Location |",
        "| :---: | :--- | :---: | :--- | :--- |",
    ]

    domain_counts: dict[str, int] = {}
    for ds in cat["datasets"]:
        c = ds["category"]
        domain_counts[c] = domain_counts.get(c, 0) + 1

    for idx, (dom, count) in enumerate(domain_counts.items(), 1):
        lines.append(
            f"| **{idx}** | **{dom}** | {count} | Comprehensive Curated Streams & Foundation Pretraining | `/Volumes/One Touch/Helios/` |"
        )

    lines.append(
        f"| **TOTAL** | **{cat['total_domains']} Domains** | **{cat['total_external_datasets']} Datasets** | **Massive Multi-Domain Harvest & Checkpoints** | **`/Volumes/One Touch/Helios/`** |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Complete Registry of All External Datasets & Direct Links")
    lines.append("")
    lines.append("| # | Domain Category | Dataset / Platform Name | Identifier | Direct Link |")
    lines.append("| :---: | :--- | :--- | :--- | :--- |")

    for ds in cat["datasets"]:
        lines.append(
            f"| {ds['index']} | **{ds['category']}** | {ds['name']} | `{ds['hf_id']}` | [{ds['url']}]({ds['url']}) |"
        )

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    cat = generate_sot_catalog()
    sot_json_path = Path("sot/datasets_sot.json")
    with open(sot_json_path, "w", encoding="utf-8") as f:
        json.dump(cat, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(cat['datasets'])} datasets in {sot_json_path}")

    readme_md = generate_sot_readme_markdown(cat)
    readme_path = Path("sot/README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_md)
    print(f"Generated {readme_path}")
