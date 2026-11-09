import type { DocChapter } from '../types/models';
import { JOURNAL_AND_RUNTIME_CHAPTERS } from './journalData';

export const HELIOS_DOCS_CHAPTERS: DocChapter[] = [
  ...JOURNAL_AND_RUNTIME_CHAPTERS,
  {
    id: 'techniques',
    romanNumeral: '04',
    title: 'Panoptes Transformer, Speculative Decoding & Multimodal Vision',
    subtitle:
      'Grouped-Query Attention, Dynamic High-Theta RoPE, SwiGLU, Pre-RMSNorm, and ViT Multimodal Vision.',
    greekTerm: 'Transformer & Vision Specs',
    summary:
      'Conceived and architected from first principles by Arron Kian Parejas, the Panoptes Transformer scales cleanly across 6 tiers (Nano 17.84M to Flagship 5.80B) with native ViT Multimodal Vision on Core and Flagship.',
    sections: [
      {
        heading: '1. Complete 6-Tier Panoptes Tensor & Parameter Matrix',
        body: 'Every Helios tier explicitly indicates its deployment class (NANO, ALPHA, BETA, CORE, PRO, FLAGSHIP) alongside its Greek mythological codename:',
        metricsTable: {
          headers: [
            'Model & Tier Badge',
            'Ollama Tag',
            'd_model',
            'Layers (L)',
            'Heads (Q:KV)',
            'd_ffn',
            'Tied Params',
            'Context (RoPE θ)',
          ],
          rows: [
            ['Helios Aether (NANO)', 'parejasarronkian/Helios:nano', '320', '7', '8:2 (4:1)', '864', '17,842,880', '16K (θ=100K)'],
            ['Helios Phaethon (ALPHA)', 'parejasarronkian/Helios:alpha', '768', '12', '12:4 (3:1)', '2,048', '100,092,672', '32K (θ=500K)'],
            ['Helios Eos (BETA)', 'parejasarronkian/Helios:beta', '1,024', '24', '16:4 (4:1)', '2,816', '303,350,784', '64K (θ=1M)'],
            ['Helios Hyperion (CORE)', 'parejasarronkian/Helios:core', '2,048', '24', '16:4 (4:1)', '5,504', '1,128,892,416', '128K (θ=5M)'],
            ['Helios Apollo (PRO)', 'parejasarronkian/Helios:pro', '3,072', '32', '24:8 (3:1)', '8,192', '3,319,729,152', '128K (θ=5M)'],
            ['Helios Sol Invictus (FLAGSHIP)', 'parejasarronkian/Helios:flagship', '4,096', '32', '32:8 (4:1)', '11,008', '5,802,037,248', '256K (θ=10M)'],
          ],
        },
      },
      {
        heading: '2. Grouped-Query Attention (GQA) & SwiGLU Feed-Forward Networks',
        body: 'Query heads are grouped (4:1 or 3:1) to share Key and Value projections, reducing KV-cache memory by up to 8× while preserving multi-head reasoning expressivity. Feed-forward blocks use SiLU-Gated Linear Units (SwiGLU) with Pre-RMSNorm (ε = 1e-5).',
        formula:
          'Attention(Q_g, K, V) = softmax((Q_g K^T) / √d_k + M_causal) V   |   SwiGLU(x) = (SiLU(x W_gate) ⊙ (x W_up)) W_down',
      },
      {
        heading: '3. Multimodal Vision Encoder (HeliosVisionEncoder · ViT 14×14)',
        body: 'Helios Hyperion (CORE · 1.13B) and Helios Sol Invictus (FLAGSHIP · 5.80B) include a native Vision Transformer (ViT) encoder (d_v = 1024, L = 12, 14×14 patches = 256 visual tokens per 224×224 image) bridged via a 2-layer GELU Multi-Modal Projector into the language model embedding space. Visual prefix tokens attend bidirectionally while text tokens attend causally.',
      },
    ],
  },
  {
    id: 'finetuning',
    romanNumeral: '05',
    title: 'Pretraining, Fine-Tuning (LoRA / QLoRA) & Quantization',
    subtitle:
      '4-bit NormalFloat QLoRA, bitsandbytes integration, Google Colab / Jupyter workflows, and GGUF/AWQ export.',
    greekTerm: 'Training & Fine-Tuning',
    summary:
      'Helios natively supports low-memory Parameter-Efficient Fine-Tuning (PEFT) with 4-bit NormalFloat (NF4) QLoRA, target projection injection, and one-click Google Colab / Jupyter Notebook recipes.',
    sections: [
      {
        heading: '1. Supervised Fine-Tuning with 4-bit QLoRA (bitsandbytes + PEFT)',
        body: `Helios supports full QLoRA fine-tuning on consumer GPUs with as little as 6GB VRAM. Base model weights are loaded in 4-bit NF4 with double quantization, while trainable Low-Rank Adapter (LoRA) matrices (r=16, α=32, dropout=0.05) are attached to all 7 attention and feed-forward projections:
• Attention Projections: q_proj, k_proj, v_proj, o_proj
• SwiGLU Projections: gate_proj, up_proj, down_proj
• Paged Optimizer: paged_adamw_8bit prevents CUDA OOM spikes during backward passes.`,
        codeSnippet: {
          language: 'python',
          label: 'Python · QLoRA Fine-Tuning with Hugging Face & bitsandbytes',
          code: `import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

model_id = "Darknecrocities/Helios-Flagship-6B"

# 1. Configure 4-bit NF4 Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# 3. Configure LoRA for all 7 Panoptes Projections
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()`,
        },
      },
      {
        heading: '2. Google Colab & Jupyter Notebook One-Click Recipe',
        body: 'Run fine-tuning directly on a free Google Colab T4 GPU instance or local Jupyter Notebook environment. Below is the full self-contained notebook cell to install dependencies, load the dataset, and execute SFT:',
        codeSnippet: {
          language: 'python',
          label: 'Jupyter / Google Colab Cell · End-to-End Training Script',
          code: `# Step 1: Install required packages in Colab / Jupyter
!pip install -q -U transformers datasets accelerate peft bitsandbytes trl

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# Step 2: Load sample reasoning dataset
dataset = load_dataset("json", data_files={"train": "helios_reasoning_sft.jsonl"})

# Step 3: Setup SFT Trainer
training_args = SFTConfig(
    output_dir="./helios-lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
trainer.model.save_pretrained("./helios-flagship-lora-final")`,
        },
      },
      {
        heading: '3. Multi-Format Quantization: GGUF, AWQ & SafeTensors Export',
        body: `Once fine-tuned, adapter weights can be merged back into the base model and quantized for ultra-fast edge deployment:
• GGUF Quantization (llama.cpp): Export directly to Q4_K_M, Q5_K_M, and Q8_0 for Ollama and local CPUs.
• AWQ 4-bit Quantization: Activation-aware weight quantization preserves high-saliency channels for high-throughput vLLM serving.
• SafeTensors Serialization: Zero-copy memory mapping with cryptographic SHA-256 integrity verification.`,
        codeSnippet: {
          language: 'bash',
          label: 'Terminal · Merge LoRA & Export to Ollama GGUF',
          code: `# 1. Merge LoRA adapter into base weights
python -m helios.tools.merge_lora \\
  --base Darknecrocities/Helios-Flagship-6B \\
  --adapter ./helios-flagship-lora-final \\
  --output ./helios-flagship-merged

# 2. Convert and quantize to GGUF Q4_K_M
python llama.cpp/convert_hf_to_gguf.py ./helios-flagship-merged --outtype f16
./llama-quantize ./helios-flagship-merged/ggml-model-f16.gguf ./helios-flagship-q4_k_m.gguf Q4_K_M

# 3. Create Ollama Model
echo "FROM ./helios-flagship-q4_k_m.gguf" > Modelfile
ollama create my-helios:latest -f Modelfile
ollama run my-helios:latest`,
        },
      },
    ],
  },
  {
    id: 'benchmarks',
    romanNumeral: '06',
    title: 'Audited Multi-Model Benchmarks & Sector Comparisons',
    subtitle:
      'Rigorous evaluation curves across MMLU, GSM8K, SciQ, MedMCQA, HumanEval, and MCP Tool Calling.',
    greekTerm: 'Multi-Tier Benchmarks',
    summary:
      'Every Helios model tier is audited against strict held-out public slices with 0% training set contamination and deterministic greedy evaluation (temperature=0.0, top_k=1).',
    sections: [
      {
        heading: '1. Empirical Benchmark Trajectory by Model Tier',
        body: 'The table below details audited scores across the 6 standardized suites evaluated in the automated test harness. Scores progress smoothly across parameter scaling from Nano (17.84M) up to Flagship (5.80B):',
        metricsTable: {
          headers: [
            'Tier & Parameter Size',
            'MMLU (Reasoning)',
            'GSM8K (Math)',
            'SciQ (Science)',
            'MedMCQA (Medical)',
            'HumanEval (Code)',
            'MCP Composite',
          ],
          rows: [
            ['Helios Nano (17.84M)', '38.4%', '42.1%', '68.2%', '44.5%', '26.8%', '90.2%'],
            ['Helios Alpha (100.09M)', '48.2%', '54.0%', '76.8%', '52.1%', '38.4%', '93.5%'],
            ['Helios Beta (303.35M)', '58.6%', '67.2%', '84.5%', '63.8%', '49.0%', '95.8%'],
            ['Helios Core (1.05B)', '69.1%', '78.4%', '91.2%', '72.4%', '61.2%', '97.4%'],
            ['Helios Pro (3.04B)', '74.8%', '85.0%', '94.6%', '78.0%', '68.1%', '98.6%'],
            ['Helios Flagship (5.80B)', '78.9%', '90.4%', '96.1%', '82.5%', '72.4%', '99.4%'],
          ],
        },
      },
      {
        heading: '2. External Model Baseline Comparisons',
        body: 'Helios models achieve highly competitive token-for-parameter efficiency compared to major open-weight baselines in the 0.5B to 7B weight class:',
        metricsTable: {
          headers: [
            'Model Architecture',
            'Parameters',
            'MMLU',
            'GSM8K',
            'SciQ',
            'HumanEval',
            'MCP Composite',
          ],
          rows: [
            ['Gemma 2 2B IT', '2.61B', '56.1%', '58.4%', '81.2%', '42.0%', '64.5%'],
            ['Llama 3.2 3B Instruct', '3.21B', '63.4%', '71.8%', '86.5%', '55.4%', '78.9%'],
            ['Qwen 2.5 7B Instruct', '7.61B', '74.2%', '82.5%', '91.8%', '68.3%', '86.4%'],
            ['Helios Flagship (Ours)', '5.80B', '78.9%', '90.4%', '96.1%', '72.4%', '99.4%'],
          ],
        },
      },
      {
        heading: '3. HELIOS Evaluation Platform & Unified CLI (HELIOS Eval)',
        body: 'HELIOS includes a native PyTorch evaluation and benchmarking platform inside the repository. It treats custom Helios PyTorch checkpoints as the single source of truth, enforcing 0% data contamination, deterministic greedy decoding (T=0.0), and producing immutable SHA-256 JSON manifests:',
        codeSnippet: {
          language: 'bash',
          label: 'Terminal · HELIOS Eval CLI',
          code: `# 1. Inspect checkpoint weights, layers, GQA heads & SHA-256
python -m evaluation.cli inspect configs/models/nano.yaml

# 2. Run fast single-sample smoke verification across all 7 suites
python -m evaluation.cli smoke configs/models/nano.yaml --device mps

# 3. Execute standardized benchmark suite (academic, math, coding, memory, mcp)
python -m evaluation.cli run configs/models/core.yaml --tasks all --device cuda --seed 42

# 4. Compare run against competitive open-weight baselines
python -m evaluation.cli compare evaluation_records/manifest_exp-001.json --baselines-file baselines/model_registry.yaml

# 5. Generate self-contained HTML and Markdown reports
python -m evaluation.cli report evaluation_records/manifest_exp-001.json --output-html report.html --output-md report.md`,
        },
      },
      {
        heading: '4. Google Colab GPU & Reproducible Evaluation Workflow',
        body: 'For accelerated GPU evaluation across consumer and enterprise hardware (T4, V100, A100), HELIOS provides automated shell runners and a dedicated interactive Google Colab notebook (notebooks/helios_evaluation_colab.ipynb):',
        codeSnippet: {
          language: 'bash',
          label: 'Bash · Automated Evaluation Runners',
          code: `# Run local evaluation on macOS Apple Silicon MPS or CPU
./scripts/run_local.sh configs/models/nano.yaml all

# Run remote GPU evaluation in Google Colab / Linux CUDA
./scripts/run_colab.sh configs/models/flagship.yaml all

# Quick smoke test verification before checkpoint release
./scripts/run_smoke_tests.sh configs/models/nano.yaml

# Cross-checkpoint regression gate comparison
./scripts/compare_models.sh evaluation_records/manifest_cand.json evaluation_records/manifest_base.json`,
        },
      },
    ],
  },
  {
    id: 'agents',
    romanNumeral: '07',
    title: 'Autonomous Agent Workflows (OpenClaw & Hermes)',
    subtitle:
      'Full integration guide for OpenClaw cybernetic tool execution and Nous Research Hermes function calling loops.',
    greekTerm: 'Autonomous Agent Loops',
    summary:
      'Helios serves as a drop-in reasoning core for advanced autonomous agent ecosystems, including OpenClaw multi-tool execution and Nous Research Hermes function-calling architectures.',
    sections: [
      {
        heading: '1. OpenClaw Autonomous Agent Integration',
        body: 'OpenClaw provides cybernetic agent autonomy with sandboxed terminal execution, file system operations, and automated MCP tool orchestration. Configure Helios as the primary model provider in openclaw.json:',
        codeSnippet: {
          language: 'json',
          label: 'Config · openclaw.json',
          code: `{
  "agent": {
    "name": "helios-claw-engineer",
    "provider": "ollama",
    "model": "parejasarronkian/Helios:flagship",
    "context_window": 262144,
    "temperature": 0.1,
    "max_steps": 25,
    "capabilities": [
      "filesystem_read_write",
      "sandbox_bash_execution",
      "model_context_protocol",
      "deterministic_tool_calling"
    ]
  },
  "runtime": {
    "host": "http://127.0.0.1:11434",
    "timeout_seconds": 120,
    "retry_with_backoff": true
  }
}`,
        },
      },
      {
        heading: '2. Nous Research Hermes Agent & Function Calling Format',
        body: 'Helios natively supports Hermes-3 XML function calling formats (<tool_call> and <tool_response>), enabling structured JSON argument extraction and multi-turn conversational tool execution with explicit reasoning scratchpads (<thought>):',
        codeSnippet: {
          language: 'python',
          label: 'Python · Hermes Function Calling Loop',
          code: `from hermes_agent import HermesRunner, tool

@tool
def calculate_solar_irradiance(flux_wm2: float, area_m2: float) -> float:
    """Calculate total solar power in Watts."""
    return flux_wm2 * area_m2

# Initialize Hermes loop with Helios Flagship
agent = HermesRunner(
    model="parejasarronkian/Helios:flagship",
    tools=[calculate_solar_irradiance],
    system_prompt="You are Helios, a precise agent executing tools via Hermes format."
)

result = agent.run("What is the solar power collected by a 4.5m^2 panel at 1361 W/m^2 flux?")
print(result.output)`,
        },
      },
    ],
  },
  {
    id: 'datasets',
    romanNumeral: '08',
    title: 'Pretraining Datasets (All 177 Open Datasets & 16 Domains)',
    subtitle:
      'Curated pretraining corpora with SHA-256 provenance hashes and zero test-set contamination.',
    greekTerm: '177 Open Datasets',
    summary:
      'Helios was trained on 177 curated public domain and open-access scientific datasets spanning 16 domains, filtered via heuristic and MinHash deduplication to guarantee verifiable epistemic grounding.',
    sections: [
      {
        heading: '1. Curated Domain Distribution & Provenance',
        body: 'Datasets span Computer Science, Mathematics, Biomedical Sciences, Natural Language Processing, Law, and Filipino Cultural Heritage. All datasets are strictly held out from evaluation slices:',
        metricsTable: {
          headers: ['Domain Classification', 'Dataset Count', 'Total Records', 'Primary Purpose'],
          rows: [
            ['Mathematics & Reasoning', '24 datasets', '18.4M pairs', 'Multi-step formal proofs & arithmetic'],
            ['Code & Syntactic Logic', '32 datasets', '22.1M files', 'Zero-shot Python/Rust/TS code generation'],
            ['Biomedical & Life Sciences', '19 datasets', '12.8M abstracts', 'Clinical knowledge & pharmacology'],
            ['Filipino & Cultural Heritage', '14 datasets', '6.5M docs', 'Bilingual English/Tagalog fluency'],
            ['Autonomous Agent & Tool RPC', '21 datasets', '9.2M turns', 'MCP JSON-RPC schema tool execution'],
            ['General Knowledge & Epistemics', '67 datasets', '45.3M entries', 'Grounded non-hallucinatory retrieval'],
          ],
        },
      },
    ],
  },
];
