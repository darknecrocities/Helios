import { HELIOS_MODELS } from './modelsData';
import type { HeliosModelTier } from '../types/models';

export type HeliosMetricKey = 'params' | 'context' | 'ram' | 'speed' | 'capability';

export interface HeliosMetricDefinition {
  id: HeliosMetricKey;
  label: string;
  unit: string;
  subtitle: string;
  getValue: (m: HeliosModelTier) => number;
  getFormatted: (m: HeliosModelTier) => string;
  getNormalizedPercent: (m: HeliosModelTier) => number;
}

export const HELIOS_FAMILY_METRICS: HeliosMetricDefinition[] = [
  {
    id: 'capability',
    label: 'Reasoning & Coding Score',
    unit: '%',
    subtitle:
      'Composite accuracy across multi-step Chain-of-Thought math, Python synthesis, and epistemic verification.',
    getValue: (m) => Math.round((m.reasoningScore + m.codingScore) / 2),
    getFormatted: (m) => `${m.reasoningScore}% Rsn · ${m.codingScore}% Code`,
    getNormalizedPercent: (m) => Math.round((m.reasoningScore + m.codingScore) / 2),
  },
  {
    id: 'params',
    label: 'Parameter Scale',
    unit: 'Params',
    subtitle:
      'Log-scaled parameter progression across all 6 Panoptes architecture tiers (17.84M to 5.80B).',
    getValue: (m) => m.tiedParamsExact / 1_000_000,
    getFormatted: (m) => `${m.tiedParams} (${m.layers}L)`,
    getNormalizedPercent: (m) => {
      const logMin = Math.log10(15);
      const logMax = Math.log10(6000);
      const val = Math.log10(m.tiedParamsExact / 1_000_000);
      return Math.round(18 + ((val - logMin) / (logMax - logMin)) * 82);
    },
  },
  {
    id: 'context',
    label: 'Context Window',
    unit: 'Tokens',
    subtitle:
      'Maximum sequence length powered by Dynamic NTK High-Theta RoPE (θ = 100K → 10M).',
    getValue: (m) => Math.round(m.contextTokens / 1024),
    getFormatted: (m) => `${Math.round(m.contextTokens / 1024)}K (${m.contextWindow})`,
    getNormalizedPercent: (m) => {
      const ctxK = Math.round(m.contextTokens / 1024);
      const logMin = Math.log2(8);
      const logMax = Math.log2(256);
      return Math.round(20 + ((Math.log2(ctxK) - logMin) / (logMax - logMin)) * 80);
    },
  },
  {
    id: 'speed',
    label: 'Generation Speed',
    unit: 'tok/s',
    subtitle:
      'Sustained local token generation throughput on Apple Silicon / Laptop GPU (with Speculative Decoding on Flagship).',
    getValue: (m) => m.speedTokensPerSec,
    getFormatted: (m) => `${m.speedTokensPerSec} tok/s`,
    getNormalizedPercent: (m) => Math.round((m.speedTokensPerSec / 265) * 100),
  },
  {
    id: 'ram',
    label: 'Active RAM (Q4_K_M)',
    unit: 'MB / GB',
    subtitle: 'Quantized INT4 memory footprint required for zero-swap local inference.',
    getValue: (m) => m.activeRam2KMB,
    getFormatted: (m) => m.activeRam2K,
    getNormalizedPercent: (m) => {
      const logMin = Math.log10(50);
      const logMax = Math.log10(3800);
      return Math.round(16 + ((Math.log10(m.activeRam2KMB) - logMin) / (logMax - logMin)) * 84);
    },
  },
];

export const HELIOS_FAMILY_COMPARISON_LIST = HELIOS_MODELS;
