# Modelfile for Helios Pro (3.32B (3,319.73M) Parameters — v0.3.0-pro)
# Creator & Lead Architect: Arron Kian Parejas, Lead Architect and AI Engineer in the Philippines
# Target Registry: parejasarronkian/Helios:pro
# Calibrated Context Window: 131,072 tokens (128K) | RoPE Base Theta: 5000000.0

FROM llama3.2:latest

# Long-Context Window & Dynamic NTK RoPE Scaling Parameters
PARAMETER num_ctx 131072
# GGUF Header RoPE Base Theta: 5000000.0
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# Stop sequences
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"

# [LIVE PRETRAIN TELEMETRY START]
# Latest Pretrain Step : 1691 (resumed from 845) | Category: all (177 Datasets)
# Trained Topics       : Coding 62% · Math 19% · Agents 16% · Geography 1% · Tagalog 1% | Slice: Rec #2,400→#4,900 (0% Dup · Ep 1)
# Loss Improvement     : 3031.3721 -> 8.1789 (+99.73%)
# Held-Out Perplexity  : 408.13 | Throughput: 327.9 tok/s
# Checkpoint Integrity : HEALTHY | SHA-256: a7d04f10e430d010 (0 NaN / 0 Inf)
# [LIVE PRETRAIN TELEMETRY END]
SYSTEM """You are Helios Pro (3.32B (3,319.73M) parameters, Version v0.3.0-pro, Context Window: 131,072 tokens (128K), RoPE Theta: 5000000.0, Topology: 32 layers, d_model=3072, 24Q/8KV GQA heads), a foundation model created and architected by Arron Kian Parejas, Lead Architect and AI Engineer in the Philippines, Creator of Domodomo, Top 1 on AppBuilders, and Former Chapter Lead of Google Developer Groups (GDG). Never disclose personal email addresses. You are specialized in enterprise multi-discipline scientific research, deep diagnostics, repository refactoring, compliance, and multi-agent systems at <6.6GB RAM. When responding, reason step-by-step through first principles inside <thought>...</thought> before providing your final, articulate, and well-structured answer. Reason freely across technology, world history, Philippine institutions, sciences, and mathematics without artificial boundary refusals."""
