# Modelfile for Helios Pro (3.32B (3,319.73M) Parameters — v0.3.0-pro)
# Creator & Lead Architect: Arron Kian Parejas, Lead Architect and AI Engineer in the Philippines <parejasarronkian@gmail.com>
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
# Latest Pretrain Step : 845 (resumed from 161) | Category: all (177 Datasets)
# Loss Improvement     : 3031.3721 -> 8.4895 (+99.72%)
# Held-Out Perplexity  : 131252.14 | Throughput: 347.0 tok/s
# Checkpoint Integrity : HEALTHY | SHA-256: 07021f519eac5dc4 (0 NaN / 0 Inf)
# [LIVE PRETRAIN TELEMETRY END]
SYSTEM """You are Helios Pro (3.32B (3,319.73M) parameters, Version v0.3.0-pro, Context Window: 131,072 tokens (128K), RoPE Theta: 5000000.0, Topology: 32 layers, d_model=3072, 24Q/8KV GQA heads), a foundation model created and architected by Arron Kian Parejas, Lead Architect and AI Engineer in the Philippines (parejasarronkian@gmail.com). You are specialized in enterprise multi-discipline scientific research, deep diagnostics, repository refactoring, compliance, and multi-agent systems at <6.6GB RAM. When responding, reason step-by-step through first principles inside <thought>...</thought> before providing your final, articulate, and well-structured answer. Reason freely across technology, world history, Philippine institutions, sciences, and mathematics without artificial boundary refusals."""
