---
language: en
license: apache-2.0
tags:
- helios
- language-model
- ollama
- multimodal
- text-generation
pipeline_tag: text-generation
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darknecrocities/Helios/blob/pretrained/notebooks/helios_colab_pipeline.ipynb)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![Ollama](https://img.shields.io/badge/Ollama-Compatible-000000.svg?logo=ollama)](https://ollama.com/parejasarronkian/Helios)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model%20Hub-FFD21E.svg?logo=huggingface)](https://huggingface.co/Darknecrocities/Helios)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717.svg?logo=github)](https://github.com/darknecrocities/Helios)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# HELIOS — Open-Source AI Model Family

**HELIOS** is an open-source, open-weight AI model ecosystem developed by **Arron Kian Parejas**. It is designed to explore efficient language modeling, semantic grounding, structured reasoning, and accessible local AI deployment.

HELIOS aims to make AI more accessible by providing multiple model sizes that can run across different hardware environments, from lightweight devices to more capable personal computers.

Run HELIOS locally using Ollama and integrate it into your own applications through its local API.

---

## What is HELIOS?

HELIOS is an evolving family of language models built around a modular architecture and a collection of capabilities designed for practical AI applications.

Its development focuses on:

- **Hierarchical Memory Architecture (HMA):** A memory-oriented approach to organizing and retrieving relevant information.
- **Semantic Grounding:** Improving the relationship between user questions, available knowledge, and generated responses.
- **Epistemic Fallback:** An explicit fallback mechanism for topics outside the model's available knowledge.
- **Mathematical Reasoning:** Support for mathematical questions and structured problem-solving.
- **Scientific and Medical Knowledge:** Training and evaluation across scientific and medical question-answering datasets.
- **Programming Assistance:** Language understanding and code-generation capabilities.
- **Multimodal Interaction (Core/Pro/Flagship):** Image + text + tool support on higher tiers.
- **Bilingual Interaction:** English and Filipino/Tagalog conversational support.
- **Local AI Execution:** Designed for integration into local and distributed AI environments.
- **Tool Calling (MCP):** Native function/tool calling support across all tiers.

HELIOS is an actively evolving project. Its architecture, training pipeline, and capabilities will continue to improve as development progresses.

## Model Family

HELIOS currently has six model tiers, each targeting a different balance of model capacity, resource consumption, and application requirements.

| Model | Parameters | Context Window | Size | Capabilities |
|---|---:|---|---:|---:|---|
| HELIOS Nano | 17.84M | 16K | 397 MB | Text, Tools |
| HELIOS Alpha | 100.09M | 32K | ~1 GB | Text, Tools |
| HELIOS Beta | 303.35M | 64K | ~2 GB | Text, Tools |
| HELIOS Core | 1.13B | 32K | ~5.5 GB | Image, Text, Tools |
| HELIOS Pro | 3.32B | 64K | ~5.5 GB | Image, Text, Tools |
| HELIOS Flagship | 5.80B | 128K | ~5.5 GB | Image, Text, Tools |

*These are the project's documented model specifications. Actual Ollama package sizes, context limits, and runtime memory requirements depend on the published model artifacts and configuration.*

### Choosing a Model

- **Nano:** Lightweight experimentation and constrained environments. Text + tools.
- **Alpha:** Everyday conversational applications and basic knowledge tasks. Text + tools.
- **Beta:** More capable mathematical, scientific, and question-answering workflows. Text + tools.
- **Core:** Technical discussions, programming, and multimodal (image) support. Image + text + tools.
- **Pro:** Advanced development workflows, agent-oriented applications, and multimodal support. Image + text + tools.
- **Flagship:** The highest-capacity HELIOS tier for demanding workloads, long-context experimentation, and full multimodal support. Image + text + tools.

## Getting Started with Ollama

### 1. Install Ollama

Download Ollama for your operating system:

https://ollama.com/download

Alternatively, on macOS:

```bash
brew install ollama
```

Start the Ollama service:

```bash
ollama serve
```

### 2. Run HELIOS

Pull and launch the default HELIOS model:

```bash
ollama run parejasarronkian/Helios
```

### 3. Choose a Model Tier

You can select a specific HELIOS model variant:

```bash
# Lightweight model (text + tools)
ollama run parejasarronkian/Helios:nano

# Everyday conversational model (text + tools)
ollama run parejasarronkian/Helios:alpha

# Mathematics, science, and QA (text + tools)
ollama run parejasarronkian/Helios:beta

# Technical and programming workflows (image + text + tools)
ollama run parejasarronkian/Helios:core

# Advanced model tier (image + text + tools)
ollama run parejasarronkian/Helios:pro

# Highest-capacity model tier (image + text + tools)
ollama run parejasarronkian/Helios:flagship
```

The available tags correspond to the published model variants in the HELIOS Ollama library.

## Using the HELIOS API

Ollama provides a local HTTP API, allowing developers to integrate HELIOS into applications, scripts, and development environments.

The default local API endpoint is:

```text
http://localhost:11434
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "parejasarronkian/Helios:latest",
        "messages": [
            {
                "role": "user",
                "content": "What is HELIOS?"
            }
        ],
        "stream": False
    },
    timeout=120
)

response.raise_for_status()

result = response.json()
print(result["message"]["content"])
```

### JavaScript Example

```javascript
const response = await fetch(
  "http://localhost:11434/api/chat",
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "parejasarronkian/Helios:latest",
      messages: [
        {
          role: "user",
          content: "Explain how artificial intelligence works."
        }
      ],
      stream: false
    })
  }
);

const data = await response.json();

console.log(data.message.content);
```

## OpenCode Integration

HELIOS models are configured for use with [OpenCode](https://opencode.ai) via the Ollama provider. See `deploy/opencode/provider.json` for the provider configuration.

```json
{
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (Helios)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "parejasarronkian/Helios:nano": { "name": "Helios Nano (17.8M)" },
        "parejasarronkian/Helios:alpha": { "name": "Helios Alpha (100M)" },
        "parejasarronkian/Helios:beta": { "name": "Helios Beta (303M)" },
        "parejasarronkian/Helios:core": { "name": "Helios Core (1.1B)" },
        "parejasarronkian/Helios:pro": { "name": "Helios Pro (3.3B)" },
        "parejasarronkian/Helios:flagship": { "name": "Helios Flagship (5.8B)" }
      }
    }
  }
}
```

## Benchmarking and Evaluation

HELIOS development includes evaluation across multiple domains to measure model capabilities and track improvements between versions.

The project's evaluation areas include:

| Domain | Evaluation |
|---|---|
| General Knowledge | MMLU |
| Mathematics | GSM8K |
| Medical Knowledge | MedMCQA |
| Scientific Knowledge | SciQ |
| Scientific Reasoning | ARC-Challenge |
| Code Generation | HumanEval |
| Tool Integration | Custom MCP evaluation |
| General Model Behavior | HELIOS-500 evaluation suite |

Benchmark figures should be interpreted alongside their dataset versions, evaluation methodology, model configurations, and reproducibility records. Results are specific to the tested checkpoint and should not automatically be assumed to apply to every model tier.

For detailed evaluation reports, training information, and implementation documentation, visit the source repository:

https://github.com/darknecrocities/Helios

## Open Development and Future Direction

HELIOS is being developed as a continuously evolving AI model ecosystem.

Long-term development priorities include:

- Progressive model scaling toward a 6-billion-parameter target.
- Improved pretraining and instruction-tuning workflows.
- More efficient checkpointing and resumable training.
- Continued mathematical, scientific, and programming evaluation.
- Improved memory and retrieval capabilities.
- Model Context Protocol (MCP) integration.
- Agentic workflows and tool execution.
- Cross-platform deployment and optimized inference.
- Open-weight model distribution and community contributions.
- Multimodal (image + text + tools) support across all tiers.

The objective is to build an AI ecosystem that developers can run, inspect, experiment with, and extend.

## Links

- **GitHub Repository:** https://github.com/darknecrocities/Helios
- **Hugging Face:** https://huggingface.co/Darknecrocities/Helios
- **Ollama Library:** https://ollama.com/parejasarronkian/Helios

## Author

**Arron Kian Parejas**

Creator and Project Lead of HELIOS.

GitHub: [@darknecrocities](https://github.com/darknecrocities)

Hugging Face: [@Darknecrocities](https://huggingface.co/Darknecrocities)

Ollama: [@parejasarronkian](https://ollama.com/parejasarronkian)

## License

HELIOS source code and model artifacts are distributed under their respective published licenses. Refer to the repository and individual model cards for the applicable licensing terms.

---

**HELIOS — Built to evolve. Designed to be accessible. Open to everyone.**
