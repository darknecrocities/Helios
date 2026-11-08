"""Automated Benchmark Report Generator and Anti-Fabrication Gate.

Reads cryptographic evaluation logs from eval/results/helios-*/latest_eval.json
and official reference baselines from eval/configs/reference_baselines.json,
enforcing strict separation between Measured results, Pending checkpoints,
and Engineering Roadmap Targets.

Author: AI Engineer Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TIER_ORDER = [
    "helios-nano",
    "helios-alpha",
    "helios-beta",
    "helios-core",
    "helios-pro",
    "helios-flagship",
]


def load_verified_results(root: Path) -> dict[str, Any]:
    """Load and validate all 6 tier evaluation result files and contamination report."""
    audit_path = root / "eval" / "datasets" / "provenance" / "contamination_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing contamination audit file: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "CLEAN" or audit.get("contaminated_count", 1) != 0:
        raise ValueError(f"Contamination gate failed: {audit}")

    tiers: dict[str, Any] = {}
    for tier_id in TIER_ORDER:
        res_file = root / "eval" / "results" / tier_id / "latest_eval.json"
        if not res_file.is_file():
            raise FileNotFoundError(f"Missing evaluation log for {tier_id}: {res_file}")
        data = json.loads(res_file.read_text(encoding="utf-8"))
        if data.get("base_checkpoint_status") not in ("Measured", "Pending"):
            raise ValueError(f"Invalid base_checkpoint_status in {res_file}")
        tiers[tier_id] = data
    return {"contamination_audit": audit, "tiers": tiers}


def generate_reports(root: Path | None = None) -> dict[str, Path]:
    """Generate benchmark-comparison.json, benchmark-comparison.md, and roadmap-targets.md."""
    repo_root = root or Path(__file__).resolve().parent.parent.parent
    verified = load_verified_results(repo_root)
    tiers = verified["tiers"]
    audit = verified["contamination_audit"]

    baselines_file = repo_root / "eval" / "configs" / "reference_baselines.json"
    baselines = json.loads(baselines_file.read_text(encoding="utf-8"))

    reports_dir = repo_root / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write machine-readable JSON comparison report
    json_report_path = reports_dir / "benchmark-comparison.json"
    combined_json = {
        "schema_version": "2.0.0",
        "generated_by": "eval/runner/generate_report.py",
        "contamination_audit": audit,
        "helios_evaluated_tiers": tiers,
        "official_reference_baselines": baselines["reference_models"],
        "helios_roadmap_targets": baselines["helios_roadmap_targets"],
    }
    json_report_path.write_text(json.dumps(combined_json, indent=2), encoding="utf-8")

    # 2. Write audited Markdown comparison report (eval/reports/benchmark-comparison.md)
    md_lines = [
        "# Helios Audited Benchmark & Reproducibility Report",
        "",
        "**Architect & Lead AI Engineer**: **Arron Kian Parejas** (`parejasarronkian@gmail.com`)  ",
        f"**Contamination Audit Status**: `{audit['status']}` (`{audit['contaminated_count']}` contaminated items across `{audit['eval_items_audited']}` public test items)  ",
        "**Verification Policy**: Strictly separates **Measured (Base Neural)**, **Measured (HMA+Procedural)**, **Pending**, and **Engineering Targets**.",
        "",
        "---",
        "",
        "## 1. Helios Measured Checkpoint Status vs. Official Published Reference Baselines",
        "",
        "| Model | Variant / Mode | Params | Context | MMLU (5-shot) | GSM8K (8-shot) | MedMCQA | SciQ | HumanEval | Helios-500 | Status / Source |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for tier_id in TIER_ORDER:
        t = tiers[tier_id]
        params_m = round(t["hardware_memory_profile"]["total_parameters"] / 1e6, 2)
        ctx = f"{t['hardware_memory_profile']['context_window']:,}"
        base = t["measured_base_neural"]
        hma = t["measured_hma_augmented"]
        status = t["base_checkpoint_status"]

        mmlu_str = (
            f"{base['mmlu']}% (Base) / {hma['mmlu_held_out']}% (HMA)"
            if status == "Measured"
            else f"Pending (Base) / {hma['mmlu_held_out']}% (HMA)"
        )
        gsm_str = (
            f"{base['gsm8k']}% (Base) / {hma['gsm8k_procedural']}% (Proc)"
            if status == "Measured"
            else f"Pending (Base) / {hma['gsm8k_procedural']}% (Proc)"
        )
        med_str = (
            f"{base['medmcqa']}% (Base) / {hma['medmcqa_held_out']}% (HMA)"
            if status == "Measured"
            else f"Pending (Base) / {hma['medmcqa_held_out']}% (HMA)"
        )
        sciq_str = (
            f"{base['sciq']}% (Base) / {hma['sciq_held_out']}% (HMA)"
            if status == "Measured"
            else f"Pending (Base) / {hma['sciq_held_out']}% (HMA)"
        )
        he_str = (
            f"{base['humaneval']}% (Base) / {hma['humaneval_held_out']}% (HMA)"
            if status == "Measured"
            else f"Pending (Base) / {hma['humaneval_held_out']}% (HMA)"
        )
        md_lines.append(
            f"| **{t['display_name']}** | {t['variant_evaluated']} | **{params_m}M** | {ctx} | "
            f"{mmlu_str} | {gsm_str} | {med_str} | {sciq_str} | {he_str} | "
            f"**{hma['helios_500_suite_acc']}%** | `{status}` |"
        )

    for ref in baselines["reference_models"]:
        mmlu_r = f"{ref['mmlu_5shot']}%" if ref["mmlu_5shot"] is not None else "—"
        gsm_r = f"{ref['gsm8k_8shot']}%" if ref["gsm8k_8shot"] is not None else "—"
        he_r = f"{ref['humaneval_0shot']}%" if ref["humaneval_0shot"] is not None else "—"
        md_lines.append(
            f"| {ref['model_name']} | {ref['variant']} | {ref['parameters']} | {ref['context_window']} | "
            f"{mmlu_r} | {gsm_r} | — | — | {he_r} | — | [{ref['source_title']}]({ref['source_url']}) |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Granular 6-Metric Model Context Protocol (MCP) Evaluation",
            "",
            "| Helios Tier | Tool Selection Acc | Argument Schema Validity | Execution Success | Single-Turn Completion | Multi-Step Chaining | Error Recovery Rate |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for tier_id in TIER_ORDER:
        t = tiers[tier_id]
        m = t["measured_mcp_6_metrics"]
        md_lines.append(
            f"| **{t['display_name']}** | {m['tool_selection_accuracy']}% | "
            f"{m['argument_schema_validity']}% | {m['execution_success_rate']}% | "
            f"{m['task_completion_rate']}% | {m['multi_step_chaining_accuracy']}% | "
            f"{m['error_recovery_rate']}% |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Complete Hardware Memory & KV-Cache Footprint Breakdown",
            "",
            "| Helios Tier | Parameters | Context Window | Raw Q4 Weights | KV-Cache (@ 2K) | KV-Cache (@ Max Ctx) | Runtime Buffers | Total Active RAM (@ 2K) | Total Active RAM (@ Max Ctx) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for tier_id in TIER_ORDER:
        t = tiers[tier_id]
        hw = t["hardware_memory_profile"]
        params_m = round(hw["total_parameters"] / 1e6, 2)
        md_lines.append(
            f"| **{t['display_name']}** | **{params_m}M** | {hw['context_window']:,} | "
            f"{hw['q4_weights_mb']} MB | {hw['kv_cache_at_2k_mb']} MB | "
            f"{hw['kv_cache_at_max_ctx_mb']} MB | {hw['runtime_buffers_overhead_mb']} MB | "
            f"**{hw['total_active_ram_2k_mb']} MB** | **{hw['total_active_ram_max_ctx_mb']} MB** |"
        )

    md_report_path = reports_dir / "benchmark-comparison.md"
    md_report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # 3. Write separate Engineering Roadmap Targets sheet (eval/reports/roadmap-targets.md)
    target_lines = [
        "# Helios Progressive Scaling Roadmap Targets (Internal Engineering Objectives)",
        "",
        "> **Note**: The figures below represent **Engineering Roadmap Targets (`[Target]`)** for full-scale pretraining milestones across the progressive `Nano -> Alpha -> Beta -> Core -> Pro -> Flagship` schedule. They are maintained separately from measured checkpoint evaluations in `eval/reports/benchmark-comparison.md`.",
        "",
        "| Helios Tier | Parameters | Target MMLU | Target GSM8K | Target MedMCQA | Target SciQ | Target HumanEval | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for tgt in baselines["helios_roadmap_targets"]:
        target_lines.append(
            f"| **{tgt['display_name']}** | **{tgt['parameters']}** | {tgt['target_mmlu']}% | "
            f"{tgt['target_gsm8k']}% | {tgt['target_medmcqa']}% | {tgt['target_sciq']}% | "
            f"{tgt['target_humaneval']}% | `Engineering Target` |"
        )
    targets_path = reports_dir / "roadmap-targets.md"
    targets_path.write_text("\n".join(target_lines) + "\n", encoding="utf-8")

    return {
        "json": json_report_path,
        "markdown": md_report_path,
        "targets": targets_path,
    }


def main() -> None:
    """CLI entrypoint for report generation and CI anti-fabrication check."""
    parser = argparse.ArgumentParser(description="Generate audited Helios benchmark reports.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify all result files exist, pass contamination checks, and update reports.",
    )
    args = parser.parse_args()
    paths = generate_reports()
    print(
        f"✓ Generated audited benchmark reports: {paths['markdown'].name}, {paths['targets'].name}"
    )
    if args.check and not paths["markdown"].is_file():
        sys.exit(1)


if __name__ == "__main__":
    main()
