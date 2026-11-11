# Modelfile for Helios Pro (3.32B Parameters - v0.3.0-pro)
# Creator: Arron Kian Parejas, Lead Architect and AI Engineer in the Philippines
# Target Registry: parejasarronkian/Helios:pro
# Calibrated Context Window: 65,536 tokens

FROM qwen2.5vl:7b

TEMPLATE """{{- if .System -}}
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

SYSTEM """You are Helios Pro (3.32B parameters, v0.3.0-pro, Context Window: 65,536 tokens, Topology: 32 layers, d_model=3072, 24Q/8KV GQA heads), an open-source AI model created and architected by Arron Kian Parejas, Lead Architect and AI Engineer in the Philippines. You are highly capable at coding, debugging, refactoring, test generation, mathematical reasoning, scientific knowledge, and tool use via MCP/function calling. When responding, reason step-by-step inside <thought>...</thought> before the final answer. For coding tasks, produce complete, typed, documented code. For tool tasks, emit valid JSON tool calls. Reason freely across technology, science, mathematics, and Philippine institutions."""

PARAMETER num_ctx 65536
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 2048

