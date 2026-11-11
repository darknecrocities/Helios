#!/usr/bin/env python3
"""Regenerate Ollama Modelfiles with tool-calling support."""

from __future__ import annotations
from pathlib import Path

TIERS = [
    ("nano", "qwen2.5:0.5b", 16384, 100000.0, "17.84M", "7 layers, d_model=320, 8Q/2KV GQA heads", "v0.3.0-nano"),
    ("alpha", "qwen2.5:1.5b", 32768, 500000.0, "100.09M", "12 layers, d_model=768, 12Q/4KV GQA heads", "v0.3.0-alpha"),
    ("beta", "qwen2.5:3b", 65536, 1000000.0, "303.35M", "24 layers, d_model=1024, 16Q/4KV GQA heads", "v0.3.0-beta"),
    ("core", "qwen2.5vl:7b", 32768, 5000000.0, "1.13B", "24 layers, d_model=2048, 16Q/4KV GQA heads", "v0.3.0-core"),
    ("pro", "qwen2.5vl:7b", 65536, 5000000.0, "3.32B", "32 layers, d_model=3072, 24Q/8KV GQA heads", "v0.3.0-pro"),
    ("flagship", "qwen2.5vl:7b", 131072, 10000000.0, "5.80B", "32 layers, d_model=4096, 32Q/8KV GQA heads", "v0.3.0-flagship"),
]

# Go template with tool support (written as raw string to preserve {{ }})
TOOLS_TEMPLATE = """{{- if .System -}}
{{ .System }}
{{- end -}}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools>:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> with NO other text.
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}{{ .Content }}
{{- else if eq .Role "assistant" }}{{ if .Content }}{{ .Content }}{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>{{- end }}{{ if not $last }}

{{- end -}}
{{- else if eq .Role "tool" }}<tool_response>
{{ .Content }}
</tool_response>
{{- end -}}
{{- if and (ne .Role "assistant") $last }}assistant
{{ end -}}
{{- end }}"""

SYSTEM = ("You are Helios {tier} ({params} parameters, {ver}, Context Window: {ctx} tokens, "
          "Topology: {topo}), an open-source AI model created and architected by "
          "Arron Kian Parejas, Lead Architect and AI Engineer in the Philippines. "
          "You are highly capable at coding, debugging, refactoring, test generation, "
          "mathematical reasoning, scientific knowledge, and tool use via MCP/function calling. "
          "When responding, reason step-by-step inside <thought>...</thought> before the final answer. "
          "For coding tasks, produce complete, typed, documented code. For tool tasks, emit valid JSON tool calls. "
          "Reason freely across technology, science, mathematics, and Philippine institutions.")

OUT = Path("deploy/ollama")
for name, base, ctx, theta, params, topo, ver in TIERS:
    tag = "parejasarronkian/Helios:" + ("latest" if name == "core" else name)
    system = SYSTEM.format(tier=name.capitalize(), params=params, ver=ver, ctx=f"{ctx:,}", theta=f"{theta:,.1f}", topo=topo)
    is_vl = "vl" in base
    # VL models need explicit template for tool support; text models inherit native
    tmpl_block = ""
    if is_vl:
        tmpl_block = 'TEMPLATE """' + TOOLS_TEMPLATE + '"""\n\n'
    lines = [
        f"# Modelfile for Helios {name.capitalize()} ({params} Parameters - {ver})",
        f"# Creator: Arron Kian Parejas, Lead Architect and AI Engineer in the Philippines",
        f"# Target Registry: {tag}",
        f"# Calibrated Context Window: {ctx:,} tokens",
        "",
        f"FROM {base}",
        "",
    ]
    if tmpl_block:
        lines.append(tmpl_block.rstrip())
        lines.append("")
    lines.append('SYSTEM """' + system + '"""')
    lines.append("")
    lines.append(f"PARAMETER num_ctx {ctx}")
    lines.append("PARAMETER temperature 0.7")
    lines.append("PARAMETER top_p 0.9")
    lines.append("PARAMETER top_k 40")
    lines.append("PARAMETER repeat_penalty 1.1")
    lines.append("PARAMETER num_predict 2048")
    lines.append("")
    content = "\n".join(lines) + "\n"
    path = OUT / ("Modelfile" if name == "core" else "Modelfile." + name)
    path.write_text(content, encoding="utf-8")
    print("wrote", path)
