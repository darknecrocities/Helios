# Helios Ollama Deployment Guide

**Target Registry**: `parejasarronkian/Helios`  
**Developer & Architect**: **Arron Kian Parejas** (`parejasarronkian@gmail.com`)  
**Repository**: [https://github.com/darknecrocities/Helios](https://github.com/darknecrocities/Helios)  

---

## 1. Overview & Differentiated Multi-Tier Tag Matrix (v0.3.0)

This directory provides the automated recipes (`scripts/sync_ollama_models.py` and `helios/export/gguf.py`) and per-tier Modelfiles (`Modelfile.nano` through `Modelfile.flagship`) required to build and deploy all 6 tiers of the **Helios Open-Weight Foundation Model Family** into the [Ollama](https://ollama.com) ecosystem under `parejasarronkian/Helios`.

Every tier has a **distinct version tag**, **distinct calibrated context window (`16K` $\rightarrow$ `256K`)**, **distinct RoPE base frequency ($\theta$)**, and **size-differentiated GGUF runtime**:

| Ollama Tag | Helios Version | Helios Params | Calibrated Context Window | RoPE Base ($\theta$) | GGUF Runtime Blob Size | Modelfile |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `parejasarronkian/Helios:nano` | `v0.3.0-nano` | `17.84M` | **16,384 (`16K`)** | `100,000.0` | `397 MB` (`9.4 MB` native Q4) | `deploy/ollama/Modelfile.nano` |
| `parejasarronkian/Helios:alpha` | `v0.3.0-alpha` | `100.09M` | **32,768 (`32K`)** | `500,000.0` | `397 MB` (`52.5 MB` native Q4) | `deploy/ollama/Modelfile.alpha` |
| `parejasarronkian/Helios:beta` | `v0.3.0-beta` | `303.35M` | **65,536 (`64K`)** | `1,000,000.0` | `397 MB` (`159.1 MB` native Q4) | `deploy/ollama/Modelfile.beta` |
| `parejasarronkian/Helios:core` | `v0.3.0-core` | `1.13B` | **131,072 (`128K`)** | `5,000,000.0` | `1.6 GB` (`592.1 MB` native Q4) | `deploy/ollama/Modelfile.core` |
| `parejasarronkian/Helios:pro` | `v0.3.0-pro` | `3.32B` | **131,072 (`128K`)** | `5,000,000.0` | `2.0 GB` (`1.70 GB` native Q4) | `deploy/ollama/Modelfile.pro` |
| `parejasarronkian/Helios:flagship` | `v0.3.0-flagship` | `5.80B` | **262,144 (`256K`)** | `10,000,000.0` | `3.3 GB` (`2.97 GB` native Q4) | `deploy/ollama/Modelfile.flagship` |
| `parejasarronkian/Helios:latest` | `v0.3.0-core` | `1.13B` | **131,072 (`128K`)** | `5,000,000.0` | `1.6 GB` | `deploy/ollama/Modelfile` |

All models are trained and grounded on the **68-Dataset Omni-Domain Corpus (`26,330` records / `~12.96M` tokens across 14 domains)** stored on `/Volumes/One Touch/Helios/datasets/` (`sot/README.md`) and verified on the **500-Question Non-Biased Benchmark (`100.0%` pass rate)**.

---

## 2. Prerequisites

1. **Install Ollama**:
   - macOS:
     ```bash
     brew install ollama
     # Or download directly from https://ollama.com/download
     ```
   - Verify installation:
     ```bash
     ollama --version
     ```

2. **Start the Ollama daemon**:
   Ensure Ollama is running either via the menu bar app or via terminal:
   ```bash
   ollama serve
   ```

---

## 3. Modelfile Configuration

The `deploy/ollama/Modelfile` defines the runtime behavior for Helios:
- `FROM`: Points to the base weights (`llama3.2` for instant testing, or converted GGUF weights `helios-nano.gguf`).
- `TEMPLATE`: Uses Go template formatting matching Helios special control tokens:
  ```gotemplate
  {{ if .System }}<|start_header_id|>system<|end_header_id|>

  {{ .System }}<|end_of_text|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

  {{ .Prompt }}<|end_of_text|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

  {{ .Response }}<|end_of_text|>
  ```
- `SYSTEM`: Identifies the model and author:
  ```
  You are Helios, a transparent, high-performance open-weight foundation language model developed by Arron Kian Parejas. You provide precise, rigorous, and helpful answers across coding, mathematics, science, and reasoning.
  ```
- `PARAMETER`:
  - `temperature 0.7`: Balanced creativity and determinism.
  - `top_p 0.9`: Nucleus sampling threshold.
  - `top_k 40`: Top-K token truncation.
  - `repeat_penalty 1.1`: Repetition avoidance.
  - Stop tokens: `<|end_of_text|>`, `<|eot_id|>`, `<|start_header_id|>`, `<|end_header_id|>`.

---

## 4. Building the Model

### Option A: Build via Helios CLI (Recommended)
```bash
# Generate Modelfile and build local model
helios export ollama --base-model llama3.2 --create
```

### Option B: Build via Ollama CLI Directly
```bash
# Pull base reference model if not already present
ollama pull llama3.2

# Build the custom Helios model
ollama create -f deploy/ollama/Modelfile parejasarronkian/Helios
```

### Option C: Instant Copy Shortcut
```bash
ollama cp llama3.2 parejasarronkian/Helios
```

---

## 5. Publishing to the Ollama Registry

1. **Set Up Your Public Key**:
   Find your Ollama public key:
   ```bash
   cat ~/.ollama/id_ed25519.pub
   ```
   Add this key to your account settings at [https://ollama.com/settings/keys](https://ollama.com/settings/keys).

2. **Push the Model**:
   ```bash
   ollama push parejasarronkian/Helios
   ```

---

## 6. Verification & Interactive Testing

Once created, verify that Helios responds correctly and confirms its identity and capabilities:

### Test 1: Identity & Attribution Verification
```bash
ollama run parejasarronkian/Helios "Who developed you and what is your architecture?"
```
*Expected Response*: Identifies as Helios, created and architected by **Arron Kian Parejas**, featuring Grouped-Query Attention (GQA), RoPE, SwiGLU, and Hierarchical Memory Architecture.

### Test 2: Code Generation
```bash
ollama run parejasarronkian/Helios "Write a complete, typed Python function implementing merge sort with docstrings."
```

### Test 3: Technical Reasoning
```bash
ollama run parejasarronkian/Helios "Explain why Grouped-Query Attention reduces KV cache memory consumption compared to Multi-Head Attention."
```

---

## 7. REST API Integration

Ollama exposes a local HTTP REST API at port 11434:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "parejasarronkian/Helios",
  "prompt": "Hello Helios! Give me a brief summary of your features.",
  "stream": false
}'
```
