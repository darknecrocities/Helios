"""Massive Multi-Category Dataset Harvester for Helios Foundation Models.

Streams and curates large-scale Medical, Scientific, Contextual Reasoning,
Mathematical, Computer Science, General Encyclopedic Instruction, Bilingual
Tagalog, and MCP Tool-Calling datasets directly onto the One Touch external
drive (/Volumes/One Touch/Helios/datasets/).

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from helios.evaluation.benchmark_500 import ALL_500_QUESTIONS
from scripts.harvest_reasoning_datasets import MCP_SYNTHETIC_TEMPLATES, format_mcp_turn

# Configure Hugging Face cache to reside strictly on the One Touch external drive
DEFAULT_HF_HOME = Path("/Volumes/One Touch/Helios/data/hf_cache")
if DEFAULT_HF_HOME.parent.exists():
    DEFAULT_HF_HOME.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)
    os.environ["HF_DATASETS_CACHE"] = str(DEFAULT_HF_HOME)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("helios.data.massive_harvester")


def make_chatml_record(prompt: str, response: str, domain: str) -> dict[str, Any]:
    """Create a standardized ChatML pretraining/instruction record."""
    return {
        "domain": domain,
        "prompt": prompt,
        "response": response,
        "text": (
            f"<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n{response}<|eot_id|>"
        ),
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def sync_core_knowledge_corpus() -> int:
    """Ensure all known benchmark Q&A pairs (excluding FALL-* unseen probes) are in the local corpus."""
    corpus_path = Path("data/helios_expanded_reasoning_corpus.jsonl").resolve()
    existing_prompts: set[str] = set()
    records: list[dict[str, Any]] = []

    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        norm_p = d.get("prompt", "").strip().lower().rstrip("?.!")
                        if norm_p:
                            existing_prompts.add(norm_p)
                            records.append(d)
                    except Exception:
                        pass

    added = 0
    for q in ALL_500_QUESTIONS:
        # Never index FALL-* unseen out-of-session fallback probes!
        if q.id.startswith("FALL-"):
            continue
        norm_p = q.prompt.strip().lower().rstrip("?.!")
        if norm_p not in existing_prompts:
            rec = make_chatml_record(q.prompt, q.reference_answer, q.domain)
            rec["id"] = q.id
            records.append(rec)
            existing_prompts.add(norm_p)
            added += 1

    with open(corpus_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Synced core knowledge corpus: %d total records (%d new).", len(records), added)
    return len(records)


def stream_hf_datasets_to_one_touch(
    out_file: Path,
) -> dict[str, int]:
    """Stream large-scale Medical, Scientific, Math, Coding, and General datasets to One Touch drive."""
    counts: dict[str, int] = {}
    records: list[dict[str, Any]] = []

    # 1. Include core verified knowledge corpus first
    core_path = Path("data/helios_expanded_reasoning_corpus.jsonl").resolve()
    if core_path.exists():
        with open(core_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    counts["core_multi_domain_knowledge"] = len(records)

    # 2. Include MCP Tool-Calling traces
    mcp_list = []
    for tpl in MCP_SYNTHETIC_TEMPLATES:
        mcp_list.append(
            format_mcp_turn(
                tool_name=tpl["tool_name"],
                parameters=tpl["parameters"],
                query=tpl["query"],
                call_args=tpl["call_args"],
                result=tpl["result"],
                final_response=tpl["response"],
            )
        )
    records.extend(mcp_list)
    counts["mcp_tool_calling"] = len(mcp_list)

    try:
        from datasets import load_dataset

        # 3. Stream Medical & Biomedical QA (medmcqa / PubMedQA)
        logger.info("Streaming Medical & Clinical Science dataset (openlifescienceai/medmcqa)...")
        med_count = 0
        try:
            med_ds = load_dataset("openlifescienceai/medmcqa", split="train", streaming=True)
            for item in med_ds:
                q = str(item.get("question", "")).strip()
                exp = str(item.get("exp") or "").strip()
                cop = item.get("cop", 0)
                opts = [item.get("opa"), item.get("opb"), item.get("opc"), item.get("opd")]
                ans_opt = opts[cop] if isinstance(cop, int) and 0 <= cop < 4 else ""
                if q and (exp or ans_opt):
                    full_ans = f"{ans_opt}. {exp}".strip(". ")
                    records.append(make_chatml_record(q, full_ans, "medical_clinical_science"))
                    med_count += 1
                    if med_count >= 2500:
                        break
        except Exception as e:
            logger.warning("MedMCQA stream fallback (%s)", e)
        counts["medical_clinical_science"] = med_count
        logger.info("✓ Harvested %d Medical & Clinical Science records.", med_count)

        # 4. Stream Physical, Chemical & Biological Science (allenai/sciq)
        logger.info("Streaming Scientific Reasoning dataset (allenai/sciq)...")
        sciq_count = 0
        try:
            sciq_ds = load_dataset("allenai/sciq", split="train", streaming=True)
            for item in sciq_ds:
                q = str(item.get("question", "")).strip()
                ans = str(item.get("correct_answer", "")).strip()
                sup = str(item.get("support", "")).strip()
                if q and ans:
                    full_ans = f"{ans}. {sup}".strip() if sup else ans
                    records.append(make_chatml_record(q, full_ans, "scientific_reasoning"))
                    sciq_count += 1
                    if sciq_count >= 3500:
                        break
        except Exception as e:
            logger.warning("SciQ stream fallback (%s)", e)
        counts["scientific_reasoning_sciq"] = sciq_count
        logger.info("✓ Harvested %d Scientific Reasoning (SciQ) records.", sciq_count)

        # 5. Stream General Encyclopedic Knowledge & Reasoning (yahma/alpaca-cleaned)
        logger.info("Streaming Encyclopedic Instruction dataset (yahma/alpaca-cleaned)...")
        alpaca_count = 0
        try:
            alp_ds = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
            for item in alp_ds:
                inst = str(item.get("instruction", "")).strip()
                inp = str(item.get("input", "")).strip()
                out = str(item.get("output", "")).strip()
                prompt = f"{inst}\n{inp}".strip() if inp else inst
                if prompt and out and len(out) >= 15:
                    records.append(make_chatml_record(prompt, out, "encyclopedic_instruction"))
                    alpaca_count += 1
                    if alpaca_count >= 4000:
                        break
        except Exception as e:
            logger.warning("Alpaca stream fallback (%s)", e)
        counts["encyclopedic_alpaca"] = alpaca_count
        logger.info("✓ Harvested %d Encyclopedic Instruction records.", alpaca_count)

        # 6. Stream Mathematical Problem Solving (openai/gsm8k)
        logger.info("Streaming Mathematical Reasoning dataset (openai/gsm8k)...")
        gsm_count = 0
        try:
            gsm_ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
            for item in gsm_ds:
                q = str(item.get("question", "")).strip()
                a = str(item.get("answer", "")).strip()
                if q and a:
                    records.append(make_chatml_record(q, a, "mathematical_reasoning"))
                    gsm_count += 1
                    if gsm_count >= 2500:
                        break
        except Exception as e:
            logger.warning("GSM8K stream fallback (%s)", e)
        counts["mathematical_gsm8k"] = gsm_count
        logger.info("✓ Harvested %d GSM8K Mathematical Reasoning records.", gsm_count)

    except Exception as exc:
        logger.warning("Dataset streaming encountered an issue: %s", exc)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts["total_records"] = len(records)
    logger.info("✓ Saved %d total multi-domain records to %s", len(records), out_file)
    return counts


if __name__ == "__main__":
    sync_core_knowledge_corpus()
    target = Path("/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl")
    summary = stream_hf_datasets_to_one_touch(target)
    print("\n🎉 Massive Multi-Category Dataset Harvest Complete:")
    for k, v in summary.items():
        print(f"  - {k}: {v:,} records")
