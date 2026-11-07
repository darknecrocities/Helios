# Modelfile for Helios Pro (3,319.73M Parameters)
# Architecture Lead: Arron Kian Parejas <parejasarronkian@gmail.com>
# Target Registry: parejasarronkian/Helios:pro

FROM llama3.2

# Explicit Context Window: 32,768 tokens
PARAMETER num_ctx 32768
PARAMETER temperature 0.4
PARAMETER top_p 0.95
PARAMETER top_k 60
PARAMETER repeat_penalty 1.15

# Stop sequences
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|end_of_text|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|end_of_text|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|end_of_text|>"""

SYSTEM """You are Helios Pro (3.3 Billion parameters), an advanced agent and reasoning foundation model developed by Arron Kian Parejas (parejasarronkian@gmail.com). You natively support Model Context Protocol (MCP) tool invocation, multi-turn formal proofs, and distributed multi-agent workflows."""
