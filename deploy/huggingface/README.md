# Helios Hugging Face Hub Deployment Guide

**Target Repository**: [https://huggingface.co/Darknecrocities/Helios](https://huggingface.co/Darknecrocities/Helios)  
**Lead Developer & Architect**: **Arron Kian Parejas** (`arronkianparejas@gmail.com`)  
**GitHub Source**: [https://github.com/darknecrocities/Helios](https://github.com/darknecrocities/Helios)  

---

## 1. Overview & Architecture

Helios publishes its open-weight model weights, tokenizer vocabularies, configurations, and evaluation artifacts to Hugging Face Hub under the canonical repository `Darknecrocities/Helios`.

The exported repository adheres strictly to standard Hugging Face formats:
- `config.json`: Strongly-typed `HeliosConfig` with `auto_map` for seamless `AutoModelForCausalLM` discovery.
- `generation_config.json`: Deterministic and sampling parameters (`temperature: 0.7`, `top_p: 0.9`, `top_k: 40`).
- `model.safetensors`: Memory-safe, zero-copy weight tensors.
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`: Full byte-level BPE tokenizer artifacts.
- `README.md`: Interactive model card with metadata tags, architecture breakdown, parameter scaling roadmap, and pretraining dataset citations.

---

## 2. Prerequisites & Tooling

### A. Install Git-Xet (Hugging Face High-Speed Large-File Extension)
Hugging Face supports `git-xet` for accelerated deduplicated storage transfers of large tensor files:
```bash
# Option 1: Official installer script
curl -fsSL https://xethub.com/install.sh | bash

# Option 2: Verify git-xet integration
git xet install
```
*Note: If `git-xet` is not available on your platform, standard `git-lfs` (`/opt/homebrew/bin/git-lfs`) functions as the standard fallback.*

### B. Configure Hugging Face SSH Keys
1. Generate an SSH key if you do not already have one:
   ```bash
   ssh-keygen -t ed25519 -C "arronkianparejas@gmail.com"
   ```
2. Display your public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
3. Copy the key and add it to your user settings at [https://huggingface.co/settings/keys](https://huggingface.co/settings/keys).
4. Verify SSH connectivity:
   ```bash
   ssh -T git@hf.co
   # Expected: Hi <username>, you have successfully authenticated!
   ```

---

## 3. Cloning the Model Repository

Clone the target Hugging Face Hub repository locally:
```bash
# Clone via SSH
git clone git@hf.co:Darknecrocities/Helios

# Navigate into the cloned repository
cd Helios
```

---

## 4. Exporting Helios Artifacts to the Cloned Repository

Using the Helios CLI export engine, stage all Hugging Face artifacts into the cloned folder:

```bash
# From the helios project root:
helios export hf --preset nano-20m --output-dir ./Helios --repo-id Darknecrocities/Helios
```

This command automatically generates:
1. `config.json`
2. `generation_config.json`
3. `tokenizer.json`
4. `tokenizer_config.json`
5. `special_tokens_map.json`
6. `model.safetensors`
7. `README.md` (Rich Model Card)

---

## 5. Staging, Committing, and Pushing to Hugging Face

```bash
cd Helios

# Stage all artifacts
git add .

# Verify status
git status

# Commit changes with descriptive conventional message
git commit -m "feat: release Helios open-weight foundation model weights and configuration

- Architecture: Pre-RMSNorm, RoPE, SwiGLU, Grouped-Query Attention (GQA)
- Model Preset: Helios Nano (~20M parameters)
- Weights: model.safetensors
- Tokenizer: Byte-Level BPE (32K vocabulary, bilingual EN/Filipino)
- Lead Architect: Arron Kian Parejas"

# Push to Hugging Face Hub
git push origin main
```

---

## 6. Remote Verification & Testing

Verify that the model can be loaded and executed directly from Hugging Face Hub:

### Python `transformers` Verification:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "Darknecrocities/Helios"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id)

# Test question answering
prompt = "<|start_header_id|>user<|end_header_id|>\n\nWho developed Helios and what are its key features?<|end_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

## 7. Model Card Schema & Dataset Citations

The generated `README.md` in the repository includes full metadata tags:
- `license: apache-2.0`
- `tags: [helios, open-weight, pytorch, causal-lm, grouped-query-attention, swiglu, hierarchical-memory]`
- `datasets:`
  - `HuggingFaceFW/fineweb-edu` (30% allocation)
  - `HuggingFaceFW/fineweb` & `mlfoundations/dCLM-baseline` (15% allocation)
  - `HuggingFaceTB/cosmopedia-v2` (15% allocation)
  - `bigcode/the-stack-v2` & `bigcode/starcoderdata` (20% allocation)
  - `open-web-math/open-web-math` & `EleutherAI/proof-pile-2` (10% allocation)
  - `uonlp/CulturaX` (tl) & `oscar-corpus/OSCAR-2301` (tl) (10% allocation)
- Explicit Attribution: **Developed by: Arron Kian Parejas**.
