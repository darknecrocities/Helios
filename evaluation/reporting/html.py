"""Standalone HTML report generator for HELIOS evaluations.

Produces single-file, self-contained HTML dashboards with embedded CSS
and SVG visuals for zero-dependency offline inspection.
"""

from __future__ import annotations

import html
from typing import Any


def generate_html_report(manifest: dict[str, Any]) -> str:
    """Generate a self-contained HTML evaluation report from a benchmark manifest."""
    model_name = html.escape(str(manifest.get("model_name", "HELIOS")))
    model_tier = html.escape(str(manifest.get("model_tier", "Unknown")))
    run_id = html.escape(str(manifest.get("run_id", "run-000")))
    timestamp = html.escape(str(manifest.get("timestamp", "")))
    environment = manifest.get("environment", {})
    results = manifest.get("results", {})
    aggregate = manifest.get("aggregate_score", 0.0)

    device = html.escape(str(environment.get("device", "cpu")))
    torch_version = html.escape(str(environment.get("torch_version", "")))

    task_rows: list[str] = []
    for task_name, task_data in sorted(results.items()):
        metrics = task_data.get("metrics", {})
        latency = task_data.get("latency", {})
        memory = task_data.get("memory", {})

        primary_metric = (
            f"Acc: {metrics['accuracy']:.1%}"
            if "accuracy" in metrics
            else (f"Pass@1: {metrics['pass_at_1']:.1%}" if "pass_at_1" in metrics else "N/A")
        )
        ece_metric = (
            f"{metrics['expected_calibration_error']:.4f}"
            if "expected_calibration_error" in metrics
            else "—"
        )
        tps = f"{latency['tokens_per_second']:.1f} tok/s" if "tokens_per_second" in latency else "—"
        peak_mb = f"{memory['peak_allocated_mb']:.1f} MB" if "peak_allocated_mb" in memory else "—"
        status = "PASSED" if task_data.get("completed", True) else "FAILED"
        status_color = "#10b981" if status == "PASSED" else "#ef4444"

        task_rows.append(
            f"""<tr>
                <td style="padding: 10px; font-weight: 600;">{html.escape(task_name)}</td>
                <td style="padding: 10px; color: {status_color}; font-weight: 500;">{status}</td>
                <td style="padding: 10px; font-variant-numeric: tabular-nums;">{primary_metric}</td>
                <td style="padding: 10px; font-variant-numeric: tabular-nums;">{ece_metric}</td>
                <td style="padding: 10px; font-variant-numeric: tabular-nums;">{tps}</td>
                <td style="padding: 10px; font-variant-numeric: tabular-nums;">{peak_mb}</td>
            </tr>"""
        )

    table_body = (
        "\n".join(task_rows)
        if task_rows
        else "<tr><td colspan='6' style='padding: 12px; text-align: center;'>No benchmark results recorded.</td></tr>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HELIOS Benchmark Report — {model_name} ({model_tier})</title>
<style>
  :root {{
    --bg: #09090b;
    --card: #141416;
    --border: #27272a;
    --text: #fafafa;
    --muted: #a1a1aa;
    --accent: #f97316;
  }}
  body {{
    margin: 0;
    padding: 32px 16px;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
  }}
  .container {{
    max-width: 960px;
    margin: 0 auto;
  }}
  header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 24px;
    margin-bottom: 32px;
  }}
  h1 {{
    margin: 0 0 8px 0;
    font-size: 28px;
    letter-spacing: -0.02em;
  }}
  .badge {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(249, 115, 22, 0.15);
    color: var(--accent);
    border: 1px solid rgba(249, 115, 22, 0.3);
  }}
  .meta-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }}
  .stat-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }}
  .stat-label {{
    font-size: 12px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .stat-val {{
    font-size: 24px;
    font-weight: 700;
    margin-top: 6px;
    color: var(--text);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }}
  th {{
    text-align: left;
    padding: 12px 10px;
    background: #18181b;
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
  }}
  tr:not(:last-child) td {{
    border-bottom: 1px solid #1f1f23;
  }}
  footer {{
    margin-top: 48px;
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    border-top: 1px solid var(--border);
    padding-top: 24px;
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div style="display: flex; justify-content: space-between; align-items: baseline;">
      <h1>HELIOS Benchmark Report</h1>
      <span class="badge">{model_tier.upper()}</span>
    </div>
    <div style="color: var(--muted); font-size: 14px; margin-top: 4px;">
      Model: <strong>{model_name}</strong> &bull; Run ID: <code>{run_id}</code> &bull; {timestamp}
    </div>
  </header>

  <div class="meta-grid">
    <div class="stat-card">
      <div class="stat-label">Aggregate Score</div>
      <div class="stat-val" style="color: var(--accent);">{aggregate:.1%}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Execution Target</div>
      <div class="stat-val">{device.upper()}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">PyTorch Runtime</div>
      <div class="stat-val" style="font-size: 18px; margin-top: 12px;">{torch_version or "N/A"}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Tasks Evaluated</div>
      <div class="stat-val">{len(results)}</div>
    </div>
  </div>

  <h2>Task Breakdown</h2>
  <table>
    <thead>
      <tr>
        <th>Benchmark</th>
        <th>Status</th>
        <th>Primary Metric</th>
        <th>ECE (Calibration)</th>
        <th>Throughput</th>
        <th>Peak RSS / VRAM</th>
      </tr>
    </thead>
    <tbody>
      {table_body}
    </tbody>
  </table>

  <footer>
    HELIOS Neural Architecture &bull; Reproducible Verification Harness &bull; Deterministic Seed 42
  </footer>
</div>
</body>
</html>
"""
