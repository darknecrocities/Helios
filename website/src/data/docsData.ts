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
      'Personally conceived and architected by Arron Kian Parejas in the Philippines, the Panoptes Transformer scales cleanly across 6 tiers (NANO to FLAGSHIP) with native Multimodal Vision (ViT) on Core and Flagship.',
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
      '6-state atomic pretraining controller, LoRA/QLoRA adaptation, and GGUF/SafeTensors/ONNX export.',
    greekTerm: 'Training & Fine-Tuning',
    summary:
      'Train custom domain adapters using built-in LoRA / QLoRA pipelines targeting all 7 attention and SwiGLU projections, backed by SHA-256 atomic checkpointing and multi-format export.',
    sections: [
      {
        heading: '1. Supervised Fine-Tuning (SFT) with LoRA / QLoRA',
        body: 'Fine-tune any Helios tier on custom instruction or reasoning JSONL datasets. LoRA adapters inject low-rank matrices (A, B) across q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, and down_proj.',
        codeSnippet: {
          language: 'bash',
          label: 'Terminal · LoRA Supervised Fine-Tuning',
          code: `uv run python -m helios.train.sft \\
  --preset core-1b \\
  --dataset data/helios_expanded_reasoning_corpus.jsonl \\
  --lora-rank 32 \\
  --lora-alpha 64 \\
  --lr 2e-4 \\
  --epochs 3 \\
  --bf16 \\
  --output-dir checkpoints/hyperion-custom-lora`,
        },
      },
      {
        heading: '2. 6-State Atomic Pretraining & Resumption Controller',
        body: 'Multi-day pretraining runs are protected by POSIX SIGINT/SIGTERM signal trapping and atomic .pt.tmp → .pt rename transactions with SHA-256 sidecar verification, preserving Model Weights, AdamW Optimizer states, Cosine LR Scheduler, GradScaler, Global Step/Tokens, and Deterministic RNG states (Python, NumPy, PyTorch CPU/CUDA).',
        codeSnippet: {
          language: 'bash',
          label: 'Terminal · Atomic Pretraining & Export to GGUF / SafeTensors / ONNX',
          code: `# Resume pretraining from the latest verified atomic checkpoint
uv run python -m helios.train.pretrain --preset flagship-6b --resume auto

# Export trained checkpoint to GGUF (for Ollama), SafeTensors (for HF), and ONNX
uv run python scripts/export_all_formats.py \\
  --tier flagship-6b \\
  --quantization q4_k_m \\
  --formats safetensors gguf onnx \\
  --verify-sha256`,
        },
      },
    ],
  },
  {
    id: 'datasets',
    romanNumeral: '06',
    title: 'Pretraining Datasets (All 177 Datasets & 16 Domains)',
    subtitle:
      'Complete Single Source of Truth catalog of every corpus used to pretrain and ground Helios.',
    greekTerm: '177 Open Datasets',
    summary:
      'Helios was pretrained and grounded on 177 curated external datasets spanning 16 foundational domains (consolidated into 90.38M tokens / 134,728 grounded FTS5 records). Below is the complete breakdown by domain and dataset.',
    sections: [
      {
        heading: 'Summary of the 16 Knowledge Domains',
        body: 'All 177 datasets are organized across 16 balanced domains to ensure rigorous reasoning in mathematics, software engineering, world history, science, law, and bilingual Filipino/Tagalog.',
        metricsTable: {
          headers: ['Domain #', 'Knowledge Domain', 'Dataset Count', 'Core Focus'],
          rows: [
            ['01', 'Foundational Web & Encyclopedic', '10 Datasets', 'FineWeb-Edu, Wikipedia, DCLM, Cosmopedia v2, RedPajama-v2, RefinedWeb, C4, The Pile'],
            ['02', 'Technology, Systems & Cloud', '12 Datasets', 'SmolLM Corpus, SecQA, OpenHermes-2.5, IETF RFCs, Linux Kernel, Kubernetes, CNCF, UltraChat'],
            ['03', 'World History & Timelines', '12 Datasets', 'Encyclopaedia Britannica, Project Gutenberg, Fordham Sourcebooks, US National Archives, Yale Avalon'],
            ['04', 'Ancient Civilizations & Archaeology', '10 Datasets', 'Perseus Digital Library (Greece/Rome), Thesaurus Linguae Graecae, CDLI Cuneiform, Laguna Copperplate'],
            ['05', 'Complex Mathematics & Formal Proofs', '12 Datasets', 'NuminaMath-CoT, OpenMathInstruct-2, OpenWebMath, Proof-Pile-2, GSM8K, MATH, MetaMathQA, Lean 4'],
            ['06', 'Biology, Genetics & Life Sciences', '12 Datasets', 'MedMCQA, PubMedQA, BioASQ, UniProtKB, MedQA USMLE, NCBI RefSeq, ChEMBL, Protein Data Bank, Ensembl'],
            ['07', 'Nature, Ecology & Earth Sciences', '10 Datasets', 'AllenAI SciQ, AI2 ARC, OpenBookQA, GBIF Biodiversity, NASA Earth Data, USGS, IPCC AR6, NOAA, IUCN'],
            ['08', 'Places, World Geography & Landmarks', '10 Datasets', 'GeoNames (11M+ Places), UNESCO World Heritage, Philippine Geopolitical Atlas, OpenStreetMap, Natural Earth'],
            ['09', 'Coding, Algorithms & Competitive Programming', '15 Datasets', 'The Stack v2, StarCoderData, Magicoder, CodeAlpaca, Glaive Code v3, Hendrycks APPS, CodeContests, HumanEval+'],
            ['10', 'Software Engineering (SWE) & GitHub PRs', '12 Datasets', 'SWE-bench Verified/Lite, CommitPackFT, SWE-gym, GitHub Code Clean, CodeXGLUE, Commit Chronicle, BugsInPy'],
            ['11', 'Agentic Skills & Multi-Step Planning', '12 Datasets', 'ToolMind (160K), AgentInstruct, UltraInteract, OpenThoughts-114k, Bespoke-Stratos, Magpie-Reasoning-V2, GAIA'],
            ['12', 'Model Context Protocol (MCP) & APIs', '10 Datasets', 'Salesforce xLAM (60K), Glaive Function-Calling v2, Gorilla BFCL, MCP Official Spec, ToolACE, ToolBench'],
            ['13', 'Technical Documentation, RFCs & Manuals', '10 Datasets', 'ReadTheDocs, MDN Web Docs, W3C Specs, Python Official Docs & PEPs, Rust Reference, POSIX, OpenAPI 3.1, PostgreSQL'],
            ['14', 'Scientific Research & Peer Review', '10 Datasets', 'AllenAI PeS2o (40M+ Papers), SciRIFF, arXiv & PubMed, OpenAlex, Semantic Scholar S2ORC, PLOS, BioRxiv'],
            ['15', 'Law, Governance, Philosophy & Social Sciences', '10 Datasets', 'Pile of Law, CourtListener, Stanford Encyclopedia of Philosophy, UN ODS, EUR-Lex, Philippine Supreme Court E-Library'],
            ['16', 'Bilingual Filipino / Tagalog & Multilingual', '10 Datasets', 'CulturaX Tagalog, OSCAR-2301, Aya Collection, Capybara, Open-Platypus, HelpSteer2, NHCP Archives, KWF Diksiyonaryo'],
          ],
        },
      },
      {
        heading: 'Complete Registry of All 177 Pretraining Datasets',
        body: 'Every external dataset integrated into the Helios pretraining and grounding pipeline:',
        metricsTable: {
          headers: ['#', 'Domain', 'Dataset Name', 'Identifier / Source'],
          rows: [
            ['1–10', 'Foundational Web', 'FineWeb-Edu, FineWeb, DCLM-Baseline, Cosmopedia v2, Wikipedia, Wikidata, RedPajama-V2, RefinedWeb, C4, The Pile', 'HuggingFaceFW / EleutherAI / AllenAI'],
            ['11–22', 'Tech & Systems', 'SmolLM-Corpus, SecQA, OpenHermes-2.5, IETF RFCs, SlimOrca, Infinity-Instruct, Cybersecurity-v1, Linux Kernel, Kubernetes, CNCF, MMLU-CS, UltraChat-200k', 'HuggingFaceTB / Teknium / Linux / CNCF'],
            ['23–34', 'World History', 'Britannica Historical, VDR History, Project Gutenberg, BigScience Historical, MMLU History, Pleias History, Fordham IHSP, US National Archives, FRUS, Harvard Historical, Yale Avalon, IWM', 'Vintage-LLM / Fordham / Yale Law'],
            ['35–44', 'Ancient Civilizations', 'CoT Ancient Past, AncientLanguages, CATMuS Medieval, Perseus Digital Library, OpenPecha, CDLI Cuneiform, Thesaurus Linguae Graecae (TLG), Papyri.info, Digital Epigraphy, Laguna Copperplate PH', 'Perseus / TLG / Darknecrocities/Helios'],
            ['45–56', 'Mathematics & Proofs', 'NuminaMath-CoT (860K), NVIDIA OpenMathInstruct-2 (14M), OpenWebMath, Proof-Pile-2, MathCode-Pile, GSM8K, Hendrycks MATH, MetaMathQA, MathQA, Lean 4 Mathlib, Isabelle/HOL AFP, Coq Stdlib', 'AI-MO / NVIDIA / OpenAI / LeanProver'],
            ['57–68', 'Biology & Life Sciences', 'Biology Textbook, MedMCQA (194K), PubMedQA, BioASQ, UniProtKB, MedQA-USMLE, NCBI RefSeq, ChEMBL, Protein Data Bank (PDB), ClinicalTrials.gov, MMLU Medicine, Ensembl Genomes', 'PubMed / UniProt / ChEMBL / PDB'],
            ['69–78', 'Nature & Earth Sciences', 'AllenAI SciQ, AI2 ARC, OpenBookQA, ClimateBERT, GBIF Biodiversity, NASA EarthData, USGS Geology, IPCC AR6 Reports, NOAA Climatology, IUCN Red List', 'AllenAI / NASA / IPCC / NOAA'],
            ['79–88', 'World Geography', 'Worldbuilding Geography, GeoNames (11M+), UNESCO World Heritage, Philippine Provincial Atlas, OpenStreetMap Planet, Natural Earth, CIA World Factbook, UN Geospatial, GADM, WDPA', 'UNESCO / OSM / Darknecrocities/Helios'],
            ['89–103', 'Coding & Algorithms', 'The Stack v2 (67.5TB), StarCoderData, CodeAlpaca-20k, Python-18k-Alpaca, Magicoder-OSS-75K, Magicoder-Evol-110K, Evol-Code-80k, Glaive-Code-v3, CodeFeedback-156K, LeetCode-2K, APPS, CodeContests, HumanEval+, MBPP+, SmolLM-Python', 'BigCode / DeepMind / EvalPlus'],
            ['104–115', 'Software Engineering', 'SWE-bench Verified, SWE-bench Lite, SWE-bench Extra, CommitPackFT, SWE-gym, GitHub-Code-Clean, SWE-smith, CodeXGLUE, Commit Chronicle, GitBug, Defects4J, BugsInPy', 'Princeton-NLP / BigCode / JetBrains'],
            ['116–127', 'Agentic Planning', 'ToolMind (160K), Agentic-Tool-Call-12k, THUDM AgentInstruct, UltraInteract, OpenThoughts-114k, Bespoke-Stratos-17k, ServiceNow R1-Distill, Magpie-Reasoning-V2, GAIA, AgentBench, WebArena, OSWorld', 'OpenThoughts / BespokeLabs / GAIA'],
            ['128–137', 'MCP & Tool Calling', 'Salesforce xLAM-60k, Glaive Function-Calling v2, Gorilla BFCL, Model Context Protocol (MCP) Spec, ToolACE, Hermes Function-Calling v1, ToolBench (16K APIs), APIBench, RestBench, JSON-RPC 2.0 Spec', 'Salesforce / Gorilla / MCP'],
            ['138–147', 'Technical Docs & RFCs', 'The Stack v2 Docs, MDN Web Docs, W3C Specifications, Python Docs & PEP Archive, Rust Reference, POSIX IEEE 1003.1, OpenAPI 3.1 Spec, PostgreSQL Docs, Docker/OCI Runtime Specs, IEEE Standards', 'MDN / W3C / Python / Rust / Postgres'],
            ['148–157', 'Scientific Literature', 'AllenAI PeS2o (40M+ Papers), AllenAI SciRIFF, arXiv & PubMed Corpus, OpenAlex (250M+ Works), Semantic Scholar S2ORC, PLOS Open Access, DOAJ, BioRxiv & MedRxiv, Europe PMC, CERN Zenodo', 'AllenAI / arXiv / OpenAlex / CERN'],
            ['158–167', 'Law & Philosophy', 'Pile of Law, Free Law CourtListener, MMLU Law & Ethics, Stanford Encyclopedia of Philosophy (SEP), World Bank Indicators, UN Official Docs, EU EUR-Lex, Philippine Supreme Court E-Library, MMLU Economics, PhilPapers', 'Pile-of-Law / Stanford SEP / SC-PH'],
            ['168–177', 'Bilingual Filipino / Tagalog', 'CulturaX Tagalog, OSCAR-2301 Tagalog, Alpaca-Cleaned, Cohere Aya Collection, Capybara Multi-Turn, Open-Platypus, NVIDIA HelpSteer2, Glot500 Parallel, NHCP Philippine Historical Archives, KWF Diksiyonaryo Filipino', 'CulturaX / Aya / NHCP / KWF'],
          ],
        },
      },
    ],
  },
];
