export type GreekDeityTierId =
  | 'nano-20m'
  | 'alpha-100m'
  | 'beta-300m'
  | 'core-1b'
  | 'pro-3b'
  | 'flagship-6b';

export type HeliosTierBadge = 'NANO' | 'ALPHA' | 'BETA' | 'CORE' | 'PRO' | 'FLAGSHIP';

export interface HeliosModelTier {
  id: GreekDeityTierId;
  tierBadge: HeliosTierBadge;
  codeName: string;
  greekName: string;
  greekScript: string;
  epithet: string;
  divineDomain: string;
  versionTag: string;
  tiedParams: string;
  tiedParamsExact: number;
  untiedParams: string;
  untiedParamsExact: number;
  embeddingParams: string;
  nonEmbeddingParams: string;
  contextWindow: string;
  contextTokens: number;
  ropeTheta: string;
  ropeThetaExact: number;
  dModel: number;
  layers: number;
  headsQ: number;
  headsKV: number;
  gqaRatio: string;
  dFFN: number;
  rawQ4Weights: string;
  kvCache2K: string;
  kvCacheMaxQ8: string;
  activeRam2K: string;
  activeRam2KMB: number;
  speedTokensPerSec: number;
  reasoningScore: number;
  codingScore: number;
  multimodalVision: boolean;
  mcpToolCalling: boolean;
  speculativeRole: string;
  supportedFormats: string[];
  targetHardware: string;
  specializedCapabilities: string[];
  mythologicalParallel: string;
  ollamaCommand: string;
  hfRepo: string;
  accentColor: string;
}

export interface DocChapter {
  id: string;
  romanNumeral: string;
  title: string;
  subtitle: string;
  greekTerm: string;
  summary: string;
  sections: {
    heading: string;
    body: string;
    formula?: string;
    codeSnippet?: {
      language: string;
      label: string;
      code: string;
    };
    metricsTable?: {
      headers: string[];
      rows: string[][];
    };
  }[];
}
