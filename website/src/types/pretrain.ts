export interface CurriculumTopic {
  name: string;
  count: number;
  percentage: number;
  samplePrompts: string[];
}

export interface PlatformProfile {
  id: string;
  title: string;
  subtitle: string;
  category: string;
  architecture: string;
  latency: string;
  memory: string;
  throughput: string;
  command: string;
  statusBadge: string;
}

export interface CurriculumStats {
  totalDatasets: number;
  totalRecords: number;
  totalTokens: number;
  tokensFormatted: string;
  cumulativeUniqueTrained: number;
  activeCursorOffset: number;
  activeEpoch: number;
  duplicationRatePct: number;
  deduplicationProtocol: string;
  externalStorage: string;
  checkpointIntegrity: string;
}

export interface PretrainMetricsData {
  version: string;
  lastUpdatedUtc: string;
  creator: {
    name: string;
    role: string;
    title: string;
    honors: string[];
  };
  curriculum: CurriculumStats;
  topics: CurriculumTopic[];
  platforms: PlatformProfile[];
}
