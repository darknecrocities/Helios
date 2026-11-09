export interface TierQuantMatrix {
  id: string;
  tierBadge: string;
  greekName: string;
  tiedParams: string;
  fp32Size: string;
  fp16Bf16Size: string;
  int8Size: string;
  q4KmSize: string;
  loraRank16Trainable: string;
  loraAdapterSize: string;
  qloraVramMin: string;
  kvCacheInt8Max: string;
  visionEncoder: string;
}

export interface PrecisionModeSpec {
  id: string;
  badge: string;
  bitsPerWeight: string;
  compression: string;
  format: string;
  useCase: string;
  desc: string;
}

export const TIER_QUANT_MATRIX: TierQuantMatrix[] = [
  {
    id: 'nano-20m',
    tierBadge: 'NANO',
    greekName: 'Helios Aether',
    tiedParams: '17.84M',
    fp32Size: '71.4 MB',
    fp16Bf16Size: '35.7 MB',
    int8Size: '17.8 MB',
    q4KmSize: '8.93 MB',
    loraRank16Trainable: '0.41M (2.29%)',
    loraAdapterSize: '1.6 MB',
    qloraVramMin: '0.4 GB VRAM / RAM',
    kvCacheInt8Max: '17.5 MB (16K Ctx)',
    visionEncoder: 'Text + Speculative Draft (γ=4)',
  },
  {
    id: 'alpha-100m',
    tierBadge: 'ALPHA',
    greekName: 'Helios Phaethon',
    tiedParams: '100.09M',
    fp32Size: '400.4 MB',
    fp16Bf16Size: '200.2 MB',
    int8Size: '100.1 MB',
    q4KmSize: '50.11 MB',
    loraRank16Trainable: '1.77M (1.76%)',
    loraAdapterSize: '7.1 MB',
    qloraVramMin: '0.9 GB VRAM / RAM',
    kvCacheInt8Max: '192.0 MB (32K Ctx)',
    visionEncoder: 'Text + Wasm Edge Engine',
  },
  {
    id: 'beta-300m',
    tierBadge: 'BETA',
    greekName: 'Helios Eos',
    tiedParams: '303.35M',
    fp32Size: '1.21 GB',
    fp16Bf16Size: '606.7 MB',
    int8Size: '303.4 MB',
    q4KmSize: '151.88 MB',
    loraRank16Trainable: '4.72M (1.55%)',
    loraAdapterSize: '18.9 MB',
    qloraVramMin: '1.6 GB VRAM / RAM',
    kvCacheInt8Max: '768.0 MB (64K Ctx)',
    visionEncoder: 'Text + 64K Code/Civic Copilot',
  },
  {
    id: 'core-1b',
    tierBadge: 'CORE',
    greekName: 'Helios Hyperion',
    tiedParams: '1.13B',
    fp32Size: '4.52 GB',
    fp16Bf16Size: '2.26 GB',
    int8Size: '1.13 GB',
    q4KmSize: '565.21 MB',
    loraRank16Trainable: '13.63M (1.21%)',
    loraAdapterSize: '54.5 MB',
    qloraVramMin: '2.8 GB VRAM',
    kvCacheInt8Max: '3.00 GB (128K Ctx)',
    visionEncoder: 'ViT 14×14 (256 Visual Patches)',
  },
  {
    id: 'pro-3b',
    tierBadge: 'PRO',
    greekName: 'Helios Apollo',
    tiedParams: '3.32B',
    fp32Size: '13.28 GB',
    fp16Bf16Size: '6.64 GB',
    int8Size: '3.32 GB',
    q4KmSize: '1.62 GB',
    loraRank16Trainable: '27.26M (0.82%)',
    loraAdapterSize: '109.1 MB',
    qloraVramMin: '4.2 GB VRAM',
    kvCacheInt8Max: '8.00 GB (128K Ctx)',
    visionEncoder: 'Text + Scientific/Legal Synthesis',
  },
  {
    id: 'flagship-6b',
    tierBadge: 'FLAGSHIP',
    greekName: 'Helios Sol Invictus',
    tiedParams: '5.80B',
    fp32Size: '23.21 GB',
    fp16Bf16Size: '11.60 GB',
    int8Size: '5.80 GB',
    q4KmSize: '2.84 GB',
    loraRank16Trainable: '40.11M (0.69%)',
    loraAdapterSize: '160.4 MB',
    qloraVramMin: '5.8 GB VRAM (Free Colab T4)',
    kvCacheInt8Max: '8.59 GB (256K Ctx)',
    visionEncoder: 'ViT 14×14 + 2-Layer GELU Bridge',
  },
];

export const PRECISION_MODES: PrecisionModeSpec[] = [
  {
    id: 'fp32',
    badge: 'FP32 (32-Bit Float)',
    bitsPerWeight: '32.0 bits/weight',
    compression: '1.0× Baseline',
    format: 'PyTorch / SafeTensors Master',
    useCase: 'Master Optimizer States, Reference Gradient Verification & Lossless Archival',
    desc: 'IEEE 754 single-precision floating point (1 sign, 8 exponent, 23 mantissa bits). Used for master parameter tracking and exact numerical unit testing.',
  },
  {
    id: 'bf16',
    badge: 'BF16 / FP16 (16-Bit)',
    bitsPerWeight: '16.0 bits/weight',
    compression: '2.0× Savings',
    format: 'SafeTensors / Hugging Face',
    useCase: 'Native Pretraining, Full-Parameter SFT & High-Throughput GPU Serving',
    desc: 'Brain Floating Point (BF16) preserves the exact 8-bit dynamic exponent range of FP32, preventing gradient underflow/overflow across 32-layer Panoptes blocks.',
  },
  {
    id: 'int8',
    badge: 'INT8 / Q8_0 (8-Bit)',
    bitsPerWeight: '8.5 bits/weight',
    compression: '4.0× Savings',
    format: 'GGUF Q8_0 / Symmetric INT8 KV',
    useCase: 'Zero-Perplexity Loss Local Inference & 256K Symmetric INT8 KV-Cache Compression',
    desc: 'Block-wise symmetric linear quantization (s = max(|x|)/127). Used both for near-lossless weight storage and compressing the 256K Flagship KV-cache from 34.3GB to 8.59GB.',
  },
  {
    id: 'q4',
    badge: 'INT4 / Q4_K_M / AWQ',
    bitsPerWeight: '4.5 bits/weight',
    compression: '7.5× Savings',
    format: 'Ollama GGUF / vLLM AWQ / ONNX',
    useCase: 'Default Everyday Consumer Laptop & Edge Deployment (61MB Nano to 2.84GB Flagship)',
    desc: 'K-quants super-block quantization (Q4_K_M) and Activation-Aware Weight Quantization (AWQ) allocate higher bit-width to salient attention/SwiGLU channels.',
  },
  {
    id: 'lora',
    badge: 'LoRA (BF16 Base + Rank r)',
    bitsPerWeight: '16-bit Base + FP16 ΔW',
    compression: '98.8% Fewer Trainable Params',
    format: 'PEFT Adapter SafeTensors',
    useCase: 'Fast Domain Adaptation across all 7 Panoptes Projections (q, k, v, o, gate, up, down)',
    desc: 'Freezes base weights W_0 and injects low-rank trainable matrices B·A (r=16, α=32, scaling α/r = 2.0) into every attention and SwiGLU linear projection.',
  },
  {
    id: 'qlora',
    badge: 'QLoRA (4-Bit NF4 + Double Quant)',
    bitsPerWeight: '4.15 bits/weight + BF16 ΔW',
    compression: '4× Training VRAM Reduction',
    format: 'bitsandbytes NF4 + Paged AdamW',
    useCase: 'Fine-Tuning Flagship 5.80B on a Single Free Google Colab T4 (5.8 GB VRAM)',
    desc: 'Quantizes frozen base weights into information-theoretically optimal 4-bit NormalFloat (NF4) with double quantization of constants, computing gradients in BF16 through LoRA adapters.',
  },
];
