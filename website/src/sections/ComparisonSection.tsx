import React from 'react';
import { Layers } from 'lucide-react';
import { HELIOS_FAMILY_COMPARISON_LIST } from '../data/comparisonData';
import { BespokeSeal } from '../components/BespokeSeals';
import { AnimatedMultiLineChart } from '../components/AnimatedMultiLineChart';

export const ComparisonSection: React.FC = () => {
  return (
    <section id="comparison" className="relative py-20 md:py-28 bg-[#FBF9F5] text-[#141312] border-t border-[#141312]/8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 gap-4">
          <div>
            <h2 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-[#141312]">
              HELIOS <span className="text-[#FF5500]">FAMILY</span> BENCHMARKS
            </h2>
          </div>
          <p className="text-xs sm:text-sm text-[#141312]/70 max-w-md leading-relaxed">
            Multi-model empirical evaluation curves across reasoning (MMLU), math (GSM8K), science (SciQ), medicine (MedMCQA), code (HumanEval), and MCP tool orchestration.
          </p>
        </div>

        {/* Multi-Line Animated Benchmark Chart directly modeled after reference */}
        <div className="mb-10">
          <AnimatedMultiLineChart showBaselinesToggle={true} />
        </div>

        {/* Detailed Architecture & Hardware Matrix Table */}
        <div className="rounded-2xl bg-white border border-[#141312]/10 overflow-hidden shadow-xs">
          <div className="px-5 py-4 bg-[#FBF9F5] border-b border-[#141312]/10 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-[#FF5500]" aria-hidden="true" />
              <h3 className="font-display text-sm sm:text-base font-bold text-[#141312]">
                Detailed Architecture & Hardware Matrix
              </h3>
            </div>
            <span className="text-[10px] font-mono-tabular text-[#141312]/50">
              Apache 2.0 Inspectable Weights
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs font-mono-tabular">
              <thead>
                <tr className="border-b border-[#141312]/10 text-[9px] uppercase tracking-wider text-[#141312]/60 bg-[#F5F0E6]/50">
                  <th className="py-3 px-4">Tier & Codename</th>
                  <th className="py-3 px-3">Parameters</th>
                  <th className="py-3 px-3">Context</th>
                  <th className="py-3 px-3">Attention</th>
                  <th className="py-3 px-3">RAM (Q4)</th>
                  <th className="py-3 px-3">Throughput</th>
                  <th className="py-3 px-4">Target Runtime Hardware</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#141312]/5">
                {HELIOS_FAMILY_COMPARISON_LIST.map((model) => (
                  <tr key={model.id} className="hover:bg-[#FBF9F5] transition-colors">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <BespokeSeal tierId={model.id} size={16} className="text-[#FF5500]" />
                        <div>
                          <span className="font-display font-bold text-[#141312] block">
                            {model.codeName}
                          </span>
                          <span className="text-[10px] text-[#FF5500]">
                            {model.tierBadge}
                          </span>
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-3 font-semibold text-[#141312]">{model.tiedParams}</td>
                    <td className="py-3 px-3 text-[#141312]/80">{model.contextWindow}</td>
                    <td className="py-3 px-3 text-[#141312]/80">{model.gqaRatio}</td>
                    <td className="py-3 px-3 font-semibold text-[#FF5500]">{model.activeRam2K}</td>
                    <td className="py-3 px-3 text-[#141312]/80">{model.speedTokensPerSec} tok/s</td>
                    <td className="py-3 px-4 text-[#141312]/70 text-[11px] font-sans">
                      {model.targetHardware}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
};
