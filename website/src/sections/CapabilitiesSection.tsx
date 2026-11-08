import React from 'react';
import { Cpu, Layers, Database, ShieldCheck, Globe2, Wrench } from 'lucide-react';
import { MorphingCard } from '../components/MorphingCard';
import { KineticHeading } from '../components/KineticText';
import { ScrollZoomSection } from '../components/ScrollZoomSection';

const CAPABILITIES = [
  {
    number: '01',
    icon: Layers,
    title: '256K Ultra-Long Context',
    metric: '262,144 Tokens',
    description:
      'Powered by Dynamic High-Theta Rotary Position Embeddings (θ = 10,000,000), Helios reads entire codebases, books, and legal archives in a single prompt without losing earlier details.',
  },
  {
    number: '02',
    icon: Cpu,
    title: '8× Lighter Memory (GQA)',
    metric: '3.54 GB Active RAM',
    description:
      'Grouped-Query Attention shares Key/Value projections across attention heads—allowing the 5.80B Flagship model to run smoothly on a standard 16GB laptop and the 17.8M Aether on a Raspberry Pi.',
  },
  {
    number: '03',
    icon: Database,
    title: '4-Tier Hierarchical Memory',
    metric: 'SQLite WAL + BM25 + FAISS',
    description:
      'Unites a sliding Working buffer, persistent SQLite WAL Episodic recall, hybrid Bilingual BM25 + FAISS Semantic vector search, and Procedural tool execution.',
  },
  {
    number: '04',
    icon: ShieldCheck,
    title: 'Honest Verification',
    metric: '99.6% Grounded Accuracy',
    description:
      'Checks whether a question’s subject entities exist in verified context before answering—refusing to guess or fabricate numbers when asked about unknown topics.',
  },
  {
    number: '05',
    icon: Wrench,
    title: 'Native MCP & Math Reasoning',
    metric: 'JSON-RPC Tool Calling',
    description:
      'Executes structured Model Context Protocol (MCP) tools, writes production software, and solves multi-step Olympiad and GSM8K mathematics with step-by-step verification.',
  },
  {
    number: '06',
    icon: Globe2,
    title: 'Bilingual English & Tagalog',
    metric: '177 Open Datasets',
    description:
      'Pretrained across 16 foundational domains including Philippine history, constitutional law, all 17 administrative regions, and natural English/Tagalog fluency.',
  },
];

export const CapabilitiesSection: React.FC = () => {
  return (
    <ScrollZoomSection
      id="capabilities"
      className="py-28 md:py-36 bg-[#F5F0E6] text-[#141312] border-t border-[#141312]/10"
    >
      <div className="max-w-[1320px] mx-auto px-4 sm:px-8 md:px-12">
        <div className="flex flex-col md:flex-row md:items-end justify-between pb-10 border-b border-[#141312]/10 gap-6">
          <div>
            <KineticHeading
              text="CORE CAPABILITIES"
              highlightWords={['CAPABILITIES']}
              className="font-display text-3xl sm:text-5xl font-bold tracking-tight text-[#141312]"
            />
          </div>
          <p className="text-sm sm:text-base text-[#141312]/70 max-w-md leading-relaxed">
            Built from first principles to combine long-context comprehension, persistent recall,
            and low memory consumption on consumer hardware.
          </p>
        </div>

        <div className="mt-10 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {CAPABILITIES.map((cap) => {
            const Icon = cap.icon;
            return (
              <MorphingCard key={cap.number} className="p-8 bg-[#FAF7F2] flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="font-mono-tabular text-xs text-[#FF5500] font-semibold">
                      {cap.number}
                    </span>
                    <Icon className="w-5 h-5 text-[#141312]" />
                  </div>
                  <h3 className="font-display text-xl font-bold text-[#141312] mt-5">
                    {cap.title}
                  </h3>
                  <p className="mt-2.5 text-sm text-[#141312]/70 leading-relaxed">
                    {cap.description}
                  </p>
                </div>

                <div className="mt-6 pt-4 border-t border-[#141312]/10 flex items-center justify-between">
                  <span className="text-[11px] uppercase tracking-wider text-[#141312]/50">
                    Verified Spec
                  </span>
                  <span className="font-mono-tabular text-xs font-semibold text-[#FF5500]">
                    {cap.metric}
                  </span>
                </div>
              </MorphingCard>
            );
          })}
        </div>
      </div>
    </ScrollZoomSection>
  );
};
