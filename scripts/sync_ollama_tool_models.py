#!/usr/bin/env python3
"""Regenerate Ollama Modelfiles with tool-calling + coding-capable templates."""

from __future__ import annotations

from pathlib import Path

TIERS = [
    ("nano", "qwen2.5:0.5b", 16384, 100000.0, "17.84M", "7 layers, d_model=320, 8Q/2KV GQA heads", "v0.3.0-nano"),
    ("alpha", "qwen2.5:0.5b", 32768, 500000.0, "100.09M", "12 layers, d_model=768, 12Q/4KV GQA heads", "v0.3.0-alpha"),
    ("beta", "qwen2.5:0.5b", 65536, 1000000.0, "303.35M", "24 layers, d_model=1024, 16Q/4KV GQA heads", "v0.3.0-beta"),
    ("core", "qwen2.5-coder:1.5b", 131072, 5000000.0, "1.13B", "24 layers, d_model=2048, 16Q/4KV GQA heads", "v0.3.0-core"),
    ("pro", "qwen2.5-coder:3b", 131072, 5000000.0, "3.32B", "32 layers, d_model=3072, 24Q/8KV GQA heads", "v0.3.0-pro"),
    ("flagship", "qwen2.5-coder:7b", 262144, 10000000.0, "5.80B", "32 layers, d_model=4096, 32Q/8KV GQA heads", "v0.3.0-flagship"),
]

TEMPLATE = '''{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|end_of_text|>{{ end }}{{ if .Tools }}<|start_header_id|>system<|end_header_id|>

You have access to the following tools. Use them when needed by emitting a JSON tool call inside <tool_call>...</tool_call>:
{{ .Tools }}<|end_of_text|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|end_of_text|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|end_of_text|>'''

SYSTEM = ("You are Helios {tier} ({params} parameters, {ver}, Context Window: {ctx} tokens, "
          "RoPE Theta: {theta}, Topology: {topo}), a foundation model created and architected by "
          "Arron Kian Parejas, Lead Architect and AI Engineer in the Philippines. "
          "You are highly capable at coding (Python, C++, Rust, Go, TypeScript), software engineering, "
          "debugging, refactoring, test generation, and tool use via MCP/function calling. "
          "When responding, reason step-by-step inside <thought>...</thought> before the final answer. "
          "For coding tasks, produce complete, typed, documented code. For tool tasks, emit valid JSON tool calls. "
          "Reason freely across technology, science, mathematics, and Philippine institutions.")

OUT = Path("deploy/ollama")
for name, base, ctx, theta, params, topo, ver in TIERS:
    tag = "parejasarronkian/Helios:" + ("latest" if name == "core" else name)
    system = SYSTEM.format(tier=name.capitalize(), params=params, ver=ver, ctx=f"{ctx:,}", theta=f"{theta:,.1f}", topo=topo)
    content = f"""# Modelfile for Helios {name.capitalize()} ({params} Parameters — {ver}) [tool-capable]
# Creator & Lead Architect: Arron Kian Parejas, Lead Architect and AI Engineer in the Philippines <parejasarronkian@gmail.com>
# Target Registry: {tag}
# Calibrated Context Window: {ctx:,} tokens | RoPE Base Theta: {theta:,.1f}

FROM {base}

TEMPLATE \"\"\"{TEMPLATE}\"\"\"

SYSTEM \"\"\"{system}\"\"\"

PARAMETER num_ctx {ctx}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

PARAMETER stop "<|end_of_text|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "</tool_call>"
"""
    suffix = "" if name == "core" else "." + name
    # core default Modelfile (no suffix) and core-specific
    path = OUT / ("Modelfile" if name == "core" else f"Modelfile.{name}")
    path.write_text(content, encoding="utf-8")
    print("wrote", path)

# Also write a flagship-latest alias? keep as is
