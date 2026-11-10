import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Copy, Check, ArrowUpRight } from 'lucide-react';
import { AnimatedMorphNumber } from '../components/CinematicMorphText';
import pretrainData from '../data/pretrain_metrics.json';
import type { PretrainMetricsData } from '../types/pretrain';

const data = pretrainData as PretrainMetricsData;

export const PretrainedCurriculumSection: React.FC = () => {
  const [activeTopicIndex, setActiveTopicIndex] = useState<number>(0);
  const [activePlatformIndex, setActivePlatformIndex] = useState<number>(0);
  const [isAutoCycling, setIsAutoCycling] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);

  // Auto-cycle through the 8 platforms every 6 seconds if not paused
  useEffect(() => {
    if (!isAutoCycling) return;
    const interval = setInterval(() => {
      setActivePlatformIndex((prev) => (prev + 1) % data.platforms.length);
    }, 6000);
    return () => clearInterval(interval);
  }, [isAutoCycling]);

  const activeTopic = data.topics[activeTopicIndex] ?? data.topics[0];
  const activePlatform = data.platforms[activePlatformIndex] ?? data.platforms[0];

  const handleCopyCommand = () => {
    navigator.clipboard.writeText(activePlatform.command);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section
      id="training"
      className="relative py-24 md:py-36 bg-[#FBF9F5] text-[#141312] border-t border-[#141312]/10 overflow-hidden"
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        {/* Asymmetrical Section Header & Editorial Lead */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-16 items-start pb-16 border-b border-[#141312]/10">
          <div className="lg:col-span-5 space-y-4">
            <div className="flex items-center gap-3">
              <span className="font-serif-editorial italic text-2xl text-[#FF5500] font-semibold">
                II.
              </span>
              <span className="font-mono-tabular text-[11px] uppercase tracking-[0.25em] text-[#141312]/50">
                Curriculum & Provenance
              </span>
            </div>
            <h2 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-[#141312] leading-[1.08]">
              {data.curriculum.totalDatasets} Curated Datasets.
              <br />
              <span className="font-serif-editorial italic font-normal text-[#FF5500]">
                Zero Redundancy.
              </span>
            </h2>
          </div>

          <div className="lg:col-span-7 lg:pt-8 space-y-4">
            <p className="font-serif-editorial text-xl sm:text-2xl text-[#141312]/85 leading-relaxed font-normal">
              Every token in the Helios foundation models was ingested from {data.curriculum.totalDatasets} open-access
              scientific, mathematical, and algorithmic corpora. Powered by a persistent
              non-repeating curriculum ledger on OneTouch storage, pretraining advances with 0.0%
              repetition across consecutive runs.
            </p>
            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs font-mono-tabular text-[#141312]/50 pt-2">
              <span>● EXTERNAL DISK: 0B INTERNAL SSD</span>
              <span>● LEDGER: SHA-256 DEDUP</span>
              <span>● STATUS: 100% HEALTHY</span>
            </div>
          </div>
        </div>

        {/* Asymmetric Hero Tabular Numerals (Cardless Pure Text) */}
        <div className="py-20 border-b border-[#141312]/10 space-y-16">
          {/* Row 1: Datasets (5 cols) & Total Tokens (7 cols) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-16 items-start">
            <div className="lg:col-span-5 space-y-2">
              <div className="text-7xl sm:text-8xl lg:text-9xl text-[#141312] leading-none">
                <AnimatedMorphNumber
                  value={data.curriculum.totalDatasets ?? 177}
                  duration={1.8}
                />
              </div>
              <div className="font-mono-tabular text-[11px] uppercase tracking-[0.2em] text-[#141312]/60 pt-2">
                Verified Open Datasets
              </div>
              <p className="text-xs sm:text-sm text-[#141312]/70 leading-relaxed font-sans max-w-sm">
                Curated across 16 foundational domains from Hugging Face collections, BigCode,
                AllenAI, MMLU, and Open Filipino corpora.
              </p>
            </div>

            <div className="lg:col-span-7 space-y-2 lg:pt-2">
              <div className="text-6xl sm:text-7xl lg:text-8xl text-[#141312] leading-none">
                <AnimatedMorphNumber
                  value={(data.curriculum.totalTokens ?? 506588190) / 1_000_000}
                  decimals={2}
                  suffix="M"
                  duration={2.0}
                />
              </div>
              <div className="font-mono-tabular text-[11px] uppercase tracking-[0.2em] text-[#141312]/60 pt-2">
                Total Pretraining Tokens
              </div>
              <p className="text-xs sm:text-sm text-[#141312]/70 leading-relaxed font-sans max-w-md">
                {data.curriculum.tokensFormatted ?? "506.59M"} cumulative tokens ({data.curriculum.baseTokensFormatted ?? "90.38M"} base corpus
                + {data.curriculum.tokensTrainedFormatted ?? "416.21M"} parameter tokens ingested in non-repeating curriculum training across all 6 tiers).
              </p>
            </div>
          </div>

          <div className="h-[1px] w-full bg-[#141312]/5" />

          {/* Row 2: Total Records (5 cols) & Total Steps (7 cols) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-16 items-start">
            <div className="lg:col-span-5 space-y-2">
              <div className="text-6xl sm:text-7xl lg:text-8xl text-[#141312] leading-none">
                <AnimatedMorphNumber
                  value={data.curriculum.totalRecords ?? 676093}
                  duration={2.2}
                />
              </div>
              <div className="font-mono-tabular text-[11px] uppercase tracking-[0.2em] text-[#141312]/60 pt-2">
                Total Pretraining Records
              </div>
              <p className="text-xs sm:text-sm text-[#141312]/70 leading-relaxed font-sans max-w-sm">
                {(data.curriculum.totalRecords ?? 676093).toLocaleString()} consolidated records on OneTouch storage
                ({(data.curriculum.cumulativeUniqueTrained ?? 555477).toLocaleString()} unique samples ingested with zero repetition; cursor #{((data.curriculum.activeCursorOffset ?? 54019)).toLocaleString()}).
              </p>
            </div>

            <div className="lg:col-span-7 space-y-2 lg:pt-2">
              <div className="text-6xl sm:text-7xl lg:text-8xl text-[#FF5500] leading-none">
                <AnimatedMorphNumber
                  value={data.curriculum.totalSteps ?? 46970}
                  duration={1.4}
                />
              </div>
              <div className="font-mono-tabular text-[11px] uppercase tracking-[0.2em] text-[#141312]/60 pt-2">
                Total Training Steps Executed
              </div>
              <p className="text-xs sm:text-sm text-[#141312]/70 leading-relaxed font-sans max-w-md">
                {(data.curriculum.totalSteps ?? 46970).toLocaleString()} forward-backward steps completed with up to 99.9% loss drop and top held-out
                perplexity of 3.28 on Nano (0.19 loss on Core, 0.035 on Alpha).
              </p>
            </div>
          </div>
        </div>

        {/* Asymmetrical Topic Spectrum & Inline Query Morphing (Pure Typography) */}
        <div className="py-20 border-b border-[#141312]/10 space-y-10">
          <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-4">
            <span className="font-mono-tabular text-[11px] uppercase tracking-[0.25em] text-[#141312]/50">
              Verified Topic Allocation (Non-Repeating Cursor)
            </span>
            <span className="font-mono-tabular text-xs text-[#141312]/60">
              Click topic to inspect real trained queries
            </span>
          </div>

          {/* Staggered Horizontal Typographic List */}
          <div className="flex flex-wrap items-baseline gap-x-8 gap-y-4">
            {data.topics.map((topic, idx) => {
              const isSelected = idx === activeTopicIndex;
              return (
                <button
                  key={topic.name}
                  onClick={() => setActiveTopicIndex(idx)}
                  className="group text-left transition-colors cursor-pointer py-1"
                >
                  <span
                    className={`font-display text-lg sm:text-xl md:text-2xl transition-all duration-300 ${
                      isSelected
                        ? 'text-[#FF5500] font-bold'
                        : 'text-[#141312]/70 hover:text-[#141312]'
                    }`}
                  >
                    {topic.name}
                  </span>
                  <span
                    className={`font-mono-tabular text-xs ml-2 tracking-wider ${
                      isSelected ? 'text-[#FF5500]' : 'text-[#141312]/40'
                    }`}
                  >
                    {topic.percentage.toFixed(1)}%
                  </span>
                  {isSelected && (
                    <motion.div
                      layoutId="topicUnderline"
                      className="h-[2px] bg-[#FF5500] mt-1 w-full"
                    />
                  )}
                </button>
              );
            })}
          </div>

          {/* Inline Editorial Sample Morphing Canvas (Cardless) */}
          <div className="pt-4 max-w-4xl">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTopic.name}
                initial={{ opacity: 0, y: 8, filter: 'blur(4px)' }}
                animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
                exit={{ opacity: 0, y: -8, filter: 'blur(4px)' }}
                transition={{ duration: 0.35, ease: 'easeOut' }}
                className="space-y-4"
              >
                <div className="flex items-baseline gap-4">
                  <span className="font-mono-tabular text-xs font-semibold uppercase tracking-wider text-[#FF5500]">
                    {activeTopic.name} Domain
                  </span>
                  <span className="font-mono-tabular text-xs text-[#141312]/50">
                    {activeTopic.count.toLocaleString()} unique records trained in this curriculum
                  </span>
                </div>

                <div className="border-l-2 border-[#141312]/15 pl-6 py-1 space-y-2">
                  <p className="font-mono-tabular text-[11px] uppercase tracking-wider text-[#141312]/40">
                    Sample Prompt Ingested in Current Non-Repeating Slice:
                  </p>
                  <p className="font-serif-editorial italic text-lg sm:text-xl text-[#141312]/90 leading-relaxed">
                    “{activeTopic.samplePrompts[0] ?? 'Verified curriculum record'}”
                  </p>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>

        {/* Cinematic Typography Platform Stage (Cardless Asymmetrical Composition) */}
        <div
          className="pt-20 space-y-12"
          onMouseEnter={() => setIsAutoCycling(false)}
          onMouseLeave={() => setIsAutoCycling(true)}
        >
          <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-4">
            <div>
              <span className="font-mono-tabular text-[11px] uppercase tracking-[0.25em] text-[#FF5500]">
                Sovereign Execution Platforms
              </span>
              <h3 className="font-display text-2xl sm:text-3xl font-bold tracking-tight text-[#141312] mt-1">
                Trained & Verified Across 8 Compute Subsystems
              </h3>
            </div>
            <div className="font-mono-tabular text-xs text-[#141312]/50 flex items-center gap-2">
              <span
                className={`inline-block w-2 h-2 rounded-full ${
                  isAutoCycling ? 'bg-[#FF5500] animate-pulse' : 'bg-[#141312]/30'
                }`}
              />
              <span>{isAutoCycling ? 'Cinematic Auto-Cycle' : 'Paused on Interaction'}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-16 items-start">
            {/* Minimalist Platform Index (4 cols) */}
            <div className="lg:col-span-4 space-y-1 border-t lg:border-t-0 lg:border-r border-[#141312]/10 pt-6 lg:pt-0 lg:pr-8">
              {data.platforms.map((platform, idx) => {
                const isActive = idx === activePlatformIndex;
                return (
                  <button
                    key={platform.id}
                    onClick={() => {
                      setActivePlatformIndex(idx);
                      setIsAutoCycling(false);
                    }}
                    className={`w-full text-left py-2.5 px-3 transition-all flex items-baseline justify-between cursor-pointer rounded-none border-b border-[#141312]/5 ${
                      isActive
                        ? 'text-[#FF5500] font-semibold bg-[#141312]/[0.03]'
                        : 'text-[#141312]/60 hover:text-[#141312]'
                    }`}
                  >
                    <span className="font-mono-tabular text-xs tracking-wider">
                      {String(idx + 1).padStart(2, '0')}. {platform.title}
                    </span>
                    <span className="font-mono-tabular text-[10px] uppercase tracking-widest text-[#141312]/40">
                      {platform.category.split(' ')[0]}
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Cinematic Morphing Typography Canvas (8 cols) */}
            <div className="lg:col-span-8 min-h-[340px] flex flex-col justify-between">
              <AnimatePresence mode="wait">
                <motion.div
                  key={activePlatform.id}
                  initial={{ opacity: 0, x: 14, filter: 'blur(6px)' }}
                  animate={{ opacity: 1, x: 0, filter: 'blur(0px)' }}
                  exit={{ opacity: 0, x: -14, filter: 'blur(6px)' }}
                  transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
                  className="space-y-6"
                >
                  <div className="space-y-1">
                    <span className="font-mono-tabular text-[10px] uppercase tracking-[0.25em] text-[#FF5500] font-semibold">
                      {activePlatform.statusBadge}
                    </span>
                    <h4 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-[#141312] leading-[1.05]">
                      {activePlatform.title}
                    </h4>
                    <p className="font-serif-editorial italic text-lg sm:text-xl text-[#141312]/75">
                      {activePlatform.subtitle}
                    </p>
                  </div>

                  <p className="text-sm sm:text-base text-[#141312]/80 leading-relaxed font-sans max-w-2xl">
                    {activePlatform.architecture}
                  </p>

                  {/* Asymmetric Telemetry Grid (Pure Typography) */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 pt-4 border-t border-[#141312]/10">
                    <div>
                      <span className="font-mono-tabular text-[10px] uppercase tracking-wider text-[#141312]/50 block">
                        Latency / Pacing
                      </span>
                      <span className="font-mono-tabular text-sm font-semibold text-[#141312] mt-0.5 block">
                        {activePlatform.latency}
                      </span>
                    </div>
                    <div>
                      <span className="font-mono-tabular text-[10px] uppercase tracking-wider text-[#141312]/50 block">
                        Memory Ceiling
                      </span>
                      <span className="font-mono-tabular text-sm font-semibold text-[#141312] mt-0.5 block">
                        {activePlatform.memory}
                      </span>
                    </div>
                    <div>
                      <span className="font-mono-tabular text-[10px] uppercase tracking-wider text-[#141312]/50 block">
                        Throughput Target
                      </span>
                      <span className="font-mono-tabular text-sm font-semibold text-[#141312] mt-0.5 block">
                        {activePlatform.throughput}
                      </span>
                    </div>
                  </div>

                  {/* Bare Monospace Terminal Invocation with Copy Action */}
                  <div className="pt-2 flex items-center justify-between gap-4 font-mono-tabular text-xs bg-[#141312]/[0.04] px-4 py-2.5 border-l-2 border-[#FF5500]">
                    <span className="truncate text-[#141312]/90 select-all">
                      $ {activePlatform.command}
                    </span>
                    <button
                      onClick={handleCopyCommand}
                      className="text-[#141312]/60 hover:text-[#FF5500] transition-colors p-1 cursor-pointer flex-shrink-0"
                      title="Copy command"
                      aria-label="Copy platform command"
                    >
                      {copied ? (
                        <Check className="w-3.5 h-3.5 text-[#FF5500]" aria-hidden="true" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" aria-hidden="true" />
                      )}
                    </button>
                  </div>
                </motion.div>
              </AnimatePresence>

              {/* Provenance Link */}
              <div className="pt-8 flex items-center gap-6 text-xs font-mono-tabular text-[#141312]/50">
                <a
                  href="https://github.com/darknecrocities/Helios/blob/pretrained/docs/TRAINED_TOPICS_HISTORY.md"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-[#FF5500] transition-colors inline-flex items-center gap-1"
                >
                  Inspect Topic Ledger
                  <ArrowUpRight className="w-3 h-3" aria-hidden="true" />
                </a>
                <span>·</span>
                <a
                  href="https://github.com/darknecrocities/Helios/blob/pretrained/sot/datasets_sot.json"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-[#FF5500] transition-colors inline-flex items-center gap-1"
                >
                  SOT Catalog ({data.curriculum.totalDatasets} Datasets)
                  <ArrowUpRight className="w-3 h-3" aria-hidden="true" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
