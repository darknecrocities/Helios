"""Chart telemetry exporter and SVG generator for HELIOS evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_chart_json(manifests: list[dict[str, Any]], output_path: str | Path) -> None:
    """Export benchmark evaluation results into web-ready JSON structure."""
    series_data: list[dict[str, Any]] = []

    for manifest in manifests:
        tier = manifest.get("model_tier", "unknown").lower()
        results = manifest.get("results", {})
        scores: dict[str, float] = {}

        for task_name, task_data in results.items():
            metrics = task_data.get("metrics", {})
            score = metrics.get("accuracy", metrics.get("pass_at_1", 0.0))
            scores[task_name.lower()] = round(score * 100, 1)

        series_data.append(
            {
                "tier": tier,
                "name": manifest.get("model_name", "HELIOS"),
                "params": manifest.get("parameters_count", 0),
                "scores": scores,
                "aggregate": round(manifest.get("aggregate_score", 0.0) * 100, 1),
            }
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"series": series_data}, f, indent=2)


def generate_radar_svg(scores: dict[str, float], size: int = 400) -> str:
    """Generate a clean SVG radar chart for model capabilities."""
    import math

    categories = list(scores.keys())
    if not categories:
        return "<svg></svg>"

    n = len(categories)
    cx, cy = size / 2, size / 2
    r = size * 0.38

    points: list[tuple[float, float]] = []
    labels: list[str] = []

    for i, cat in enumerate(categories):
        angle = (2 * math.pi / n) * i - (math.pi / 2)
        val = min(max(scores[cat], 0.0), 1.0)
        px = cx + r * val * math.cos(angle)
        py = cy + r * val * math.sin(angle)
        points.append((px, py))

        lx = cx + (r + 28) * math.cos(angle)
        ly = cy + (r + 28) * math.sin(angle)
        labels.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="11" fill="#71717a" text-anchor="middle" dominant-baseline="middle">{cat}</text>'
        )

    pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    return f"""<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#27272a" stroke-width="1" stroke-dasharray="3 3"/>
  <circle cx="{cx}" cy="{cy}" r="{r * 0.5}" fill="none" stroke="#27272a" stroke-width="1" stroke-dasharray="3 3"/>
  <polygon points="{pts_str}" fill="rgba(249, 115, 22, 0.2)" stroke="#f97316" stroke-width="2"/>
  {"".join(labels)}
</svg>"""
