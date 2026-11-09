import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  BENCHMARK_DIMENSIONS,
  HELIOS_FAMILY_BENCHMARK,
  BASELINE_MODELS_BENCHMARK,
  MODEL_SCALE_AXIS,
  CAPABILITY_SCALING_SERIES,
  type BenchmarkModelScore,
} from '../data/benchmarkData';
import { relaxLabels, generateCubicBezierPath } from './chartUtils';

interface AnimatedMultiLineChartProps {
  className?: string;
  showBaselinesToggle?: boolean;
}

export const AnimatedMultiLineChart: React.FC<AnimatedMultiLineChartProps> = ({
  className = '',
  showBaselinesToggle = true,
}) => {
  const [viewMode, setViewMode] = useState<'scaling' | 'models'>('scaling');
  const [showBaselines, setShowBaselines] = useState<boolean>(false);
  const [hoveredSeriesId, setHoveredSeriesId] = useState<string | null>(null);
  const [activePoint, setActivePoint] = useState<{
    title: string;
    dimLabel: string;
    score: number;
    x: number;
    y: number;
    color: string;
  } | null>(null);

  const width = 860, height = 440, padL = 56, padR = 190, padT = 40, padB = 52;
  const chartW = width - padL - padR, chartH = height - padT - padB;
  const yTicks = [0, 20, 40, 60, 80, 100];

  const getY = (score: number) => padT + chartH - (score / 100) * chartH;

  // VIEW 1: Scaling Law Trajectories across 6 Tiers (Matches reference sector chart)
  const scalingAxisLength = MODEL_SCALE_AXIS.length;
  const getScalingX = (idx: number) => padL + (idx / (scalingAxisLength - 1)) * chartW;

  const generateScalingPath = (scores: Record<string, number>) => {
    const pts = MODEL_SCALE_AXIS.map((tier, i) => ({
      x: getScalingX(i),
      y: getY(scores[tier.key] ?? 0),
    }));
    return generateCubicBezierPath(pts);
  };

  // VIEW 2: Task Dimension Breakdown across Model Tiers
  const taskAxisLength = BENCHMARK_DIMENSIONS.length;
  const getTaskX = (idx: number) => padL + (idx / (taskAxisLength - 1)) * chartW;

  const generateTaskPath = (scores: Record<string, number>) => {
    const pts = BENCHMARK_DIMENSIONS.map((dim, i) => ({
      x: getTaskX(i),
      y: getY(scores[dim.key] ?? 0),
    }));
    return generateCubicBezierPath(pts);
  };

  const displayModels: BenchmarkModelScore[] = useMemo(() => {
    return showBaselines
      ? [...HELIOS_FAMILY_BENCHMARK, ...BASELINE_MODELS_BENCHMARK]
      : HELIOS_FAMILY_BENCHMARK;
  }, [showBaselines]);

  // Label collision relaxation for View 1 (Scaling)
  const scalingLabelPositions = useMemo(() => {
    const items = CAPABILITY_SCALING_SERIES.map((s) => ({
      id: s.id,
      targetY: getY(s.scores.flagship ?? 0),
      label: s.endLabel,
      color: s.color,
    }));
    return relaxLabels(items, 22, padT, padT + chartH);
  }, [padT, chartH]);

  // Label collision relaxation for View 2 (Models)
  const modelLabelPositions = useMemo(() => {
    const items = displayModels.map((m) => {
      const lastKey = BENCHMARK_DIMENSIONS[BENCHMARK_DIMENSIONS.length - 1].key;
      return {
        id: m.id,
        targetY: getY(m.scores[lastKey] ?? 0),
        label: m.endLabel,
        color: m.color,
      };
    });
    return relaxLabels(items, 22, padT, padT + chartH);
  }, [displayModels, padT, chartH]);

  return (
    <div className={`w-full rounded-2xl bg-[#FAF8F5] border border-[#141312]/10 p-5 sm:p-7 shadow-sm ${className}`}>
      {/* Chart Header matching reference image */}
      <div className="flex flex-col lg:flex-row lg:items-start justify-between pb-4 border-b border-[#141312]/8 gap-4">
        <div>
          <h3 className="font-display text-xl sm:text-2xl font-bold tracking-tight text-[#141312]">
            {viewMode === 'scaling'
              ? 'Empirical Capability Scaling Across Helios Tiers'
              : 'Standardized Benchmark Breakdown by Model'}
          </h3>
          <p className="text-xs text-[#141312]/60 mt-1 font-mono-tabular">
            Source: Helios Neural Evaluation Harness | 0% Contamination Audited | SHA-256 ba9724f5...
          </p>
        </div>

        {/* View Mode Switcher + Baselines Toggle */}
        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <div className="inline-flex rounded-full bg-[#141312]/5 p-0.5 border border-[#141312]/10 text-xs font-mono-tabular">
            <button
              onClick={() => setViewMode('scaling')}
              className={`px-3 py-1 rounded-full transition-all cursor-pointer ${
                viewMode === 'scaling' ? 'bg-[#FF5500] text-white font-bold shadow-xs' : 'text-[#141312]/70 hover:text-[#141312]'
              }`}
            >
              Scaling Trajectory
            </button>
            <button
              onClick={() => setViewMode('models')}
              className={`px-3 py-1 rounded-full transition-all cursor-pointer ${
                viewMode === 'models' ? 'bg-[#FF5500] text-white font-bold shadow-xs' : 'text-[#141312]/70 hover:text-[#141312]'
              }`}
            >
              Task Breakdown
            </button>
          </div>

          {viewMode === 'models' && showBaselinesToggle && (
            <button
              onClick={() => setShowBaselines((prev) => !prev)}
              className={`px-3 py-1 rounded-full text-xs font-mono-tabular border transition-all cursor-pointer ${
                showBaselines
                  ? 'bg-[#141312] text-white border-[#141312]'
                  : 'bg-white text-[#141312]/80 border-[#141312]/20 hover:border-[#FF5500]'
              }`}
            >
              {showBaselines ? '✓ Showing Baselines' : '+ Baselines'}
            </button>
          )}
        </div>
      </div>

      {/* Filter / Highlight Pills */}
      <div className="flex flex-wrap items-center gap-2 pt-3 pb-2">
        <span className="text-[11px] font-mono-tabular uppercase tracking-wider text-[#141312]/50 mr-1">Series:</span>
        {viewMode === 'scaling'
          ? CAPABILITY_SCALING_SERIES.map((s) => {
              const isSel = hoveredSeriesId === s.id;
              return (
                <button
                  key={s.id}
                  onClick={() => setHoveredSeriesId((prev) => (prev === s.id ? null : s.id))}
                  onMouseEnter={() => setHoveredSeriesId(s.id)}
                  onMouseLeave={() => setHoveredSeriesId(null)}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono-tabular transition-all cursor-pointer ${
                    isSel
                      ? 'ring-2 ring-offset-1 text-white shadow-xs'
                      : 'bg-white/80 border border-[#141312]/10 text-[#141312]/80 hover:border-[#141312]/30'
                  }`}
                  style={{ backgroundColor: isSel ? s.color : undefined, color: isSel ? '#FFFFFF' : undefined }}
                >
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: s.color }} />
                  <span className="font-semibold">{s.name.split(' (')[0]}</span>
                </button>
              );
            })
          : displayModels.map((m) => {
              const isSel = hoveredSeriesId === m.id;
              return (
                <button
                  key={m.id}
                  onClick={() => setHoveredSeriesId((prev) => (prev === m.id ? null : m.id))}
                  onMouseEnter={() => setHoveredSeriesId(m.id)}
                  onMouseLeave={() => setHoveredSeriesId(null)}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono-tabular transition-all cursor-pointer ${
                    isSel
                      ? 'ring-2 ring-offset-1 text-white shadow-xs'
                      : 'bg-white/80 border border-[#141312]/10 text-[#141312]/80 hover:border-[#141312]/30'
                  }`}
                  style={{ backgroundColor: isSel ? m.color : undefined, color: isSel ? '#FFFFFF' : undefined }}
                >
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: m.color }} />
                  <span className="font-semibold">{m.name}</span>
                </button>
              );
            })}
      </div>

      {/* SVG Stage */}
      <div className="relative w-full aspect-[2.1/1] min-h-[300px] mt-2 select-none">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
          {/* Subtle horizontal grid lines */}
          {yTicks.map((tick) => {
            const y = getY(tick);
            return (
              <g key={`ytick-${tick}`}>
                <line
                  x1={padL}
                  y1={y}
                  x2={padL + chartW}
                  y2={y}
                  stroke="#141312"
                  strokeOpacity={tick === 0 ? 0.25 : 0.07}
                  strokeWidth={tick === 0 ? 1.5 : 1}
                  strokeDasharray={tick === 0 ? undefined : '3 3'}
                />
                <text x={padL - 10} y={y + 4} textAnchor="end" className="font-mono-tabular text-[11px] fill-[#141312]/60 font-medium">
                  {tick}%
                </text>
              </g>
            );
          })}

          {/* Left Y-Axis Spine */}
          <line x1={padL} y1={padT - 10} x2={padL} y2={padT + chartH} stroke="#141312" strokeWidth={1.5} strokeOpacity={0.4} />

          {/* Bottom X-Axis: Model Tiers or Task Dimensions */}
          {viewMode === 'scaling'
            ? MODEL_SCALE_AXIS.map((tier, idx) => {
                const x = getScalingX(idx);
                return (
                  <g key={`x-tier-${tier.key}`}>
                    <line x1={x} y1={padT + chartH} x2={x} y2={padT + chartH + 5} stroke="#141312" strokeWidth={1.2} strokeOpacity={0.3} />
                    <text x={x} y={padT + chartH + 20} textAnchor="middle" className="font-mono-tabular text-[11px] fill-[#141312]/80 font-bold">
                      {tier.name}
                    </text>
                    <text x={x} y={padT + chartH + 32} textAnchor="middle" className="font-mono-tabular text-[9px] fill-[#141312]/50">
                      {tier.params}
                    </text>
                  </g>
                );
              })
            : BENCHMARK_DIMENSIONS.map((dim, idx) => {
                const x = getTaskX(idx);
                return (
                  <g key={`x-dim-${dim.key}`}>
                    <line x1={x} y1={padT + chartH} x2={x} y2={padT + chartH + 5} stroke="#141312" strokeWidth={1.2} strokeOpacity={0.3} />
                    <text x={x} y={padT + chartH + 20} textAnchor="middle" className="font-mono-tabular text-[11px] fill-[#141312]/80 font-bold">
                      {dim.key}
                    </text>
                    <text x={x} y={padT + chartH + 32} textAnchor="middle" className="font-mono-tabular text-[9px] fill-[#141312]/50">
                      {dim.label.replace(/.*\((.*)\)/, '$1')}
                    </text>
                  </g>
                );
              })}

          {/* VIEW 1: Capability Scaling Trajectories (Reference chart aesthetic) */}
          {viewMode === 'scaling' &&
            CAPABILITY_SCALING_SERIES.map((s, sIdx) => {
              const isDim = hoveredSeriesId !== null && hoveredSeriesId !== s.id;
              const isHov = hoveredSeriesId === s.id;
              const strokeWidth = isHov ? 3.5 : isDim ? 1.5 : 2.6;
              const strokeOpacity = isDim ? 0.20 : 1;
              const pathD = generateScalingPath(s.scores);
              const lastX = getScalingX(scalingAxisLength - 1);
              const actualY = getY(s.scores.flagship ?? 0);
              const relaxedY = scalingLabelPositions.get(s.id) ?? actualY;

              return (
                <g
                  key={`scaling-line-${s.id}`}
                  className="transition-all duration-200 cursor-pointer"
                  onMouseEnter={() => setHoveredSeriesId(s.id)}
                  onMouseLeave={() => setHoveredSeriesId(null)}
                >
                  {isHov && <path d={pathD} fill="none" stroke={s.color} strokeWidth={8} strokeOpacity={0.25} strokeLinecap="round" />}
                  <motion.path
                    d={pathD}
                    fill="none"
                    stroke={s.color}
                    strokeWidth={strokeWidth}
                    strokeOpacity={strokeOpacity}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1], delay: sIdx * 0.08 }}
                  />
                  {MODEL_SCALE_AXIS.map((tier, tIdx) => {
                    const x = getScalingX(tIdx);
                    const score = s.scores[tier.key] ?? 0;
                    const y = getY(score);
                    return (
                      <circle
                        key={`pt-${s.id}-${tier.key}`}
                        cx={x}
                        cy={y}
                        r={isHov ? 4.5 : 3}
                        fill="#FFFFFF"
                        stroke={s.color}
                        strokeWidth={isHov ? 2.5 : 1.8}
                        opacity={strokeOpacity}
                        className="transition-all duration-150"
                        onMouseEnter={(e) => {
                          e.stopPropagation();
                          setHoveredSeriesId(s.id);
                          setActivePoint({ title: s.name, dimLabel: `${tier.name} (${tier.params})`, score, x, y, color: s.color });
                        }}
                        onMouseLeave={() => setActivePoint(null)}
                      />
                    );
                  })}
                  {/* End-of-line Label with collision relaxation */}
                  <motion.g initial={{ opacity: 0, x: -6 }} animate={{ opacity: strokeOpacity, x: 0 }} transition={{ duration: 0.4, delay: 0.5 + sIdx * 0.08 }}>
                    <line x1={lastX + 4} y1={actualY} x2={lastX + 14} y2={relaxedY} stroke={s.color} strokeWidth={1.2} strokeOpacity={strokeOpacity * 0.7} />
                    <text
                      x={lastX + 18}
                      y={relaxedY + 3.5}
                      fill={s.color}
                      className={`font-mono-tabular text-[11px] font-bold ${isHov ? 'underline' : ''}`}
                    >
                      {s.scores.flagship?.toFixed(1)}%{' '}
                      <tspan className="font-sans font-medium text-[10px]" fill={s.color} opacity={0.88}>
                        {s.name.split(' (')[0]}
                      </tspan>
                    </text>
                  </motion.g>
                </g>
              );
            })}

          {/* VIEW 2: Model Tier Breakdown */}
          {viewMode === 'models' &&
            displayModels.map((m, mIdx) => {
              const isDim = hoveredSeriesId !== null && hoveredSeriesId !== m.id;
              const isHov = hoveredSeriesId === m.id;
              const strokeWidth = isHov ? 3.5 : isDim ? 1.5 : 2.6;
              const strokeOpacity = isDim ? 0.22 : 1;
              const pathD = generateTaskPath(m.scores);
              const lastKey = BENCHMARK_DIMENSIONS[taskAxisLength - 1].key;
              const lastX = getTaskX(taskAxisLength - 1);
              const actualY = getY(m.scores[lastKey] ?? 0);
              const relaxedY = modelLabelPositions.get(m.id) ?? actualY;

              return (
                <g
                  key={`model-line-${m.id}`}
                  className="transition-all duration-200 cursor-pointer"
                  onMouseEnter={() => setHoveredSeriesId(m.id)}
                  onMouseLeave={() => setHoveredSeriesId(null)}
                >
                  {isHov && <path d={pathD} fill="none" stroke={m.color} strokeWidth={8} strokeOpacity={0.25} strokeLinecap="round" />}
                  <motion.path
                    d={pathD}
                    fill="none"
                    stroke={m.color}
                    strokeWidth={strokeWidth}
                    strokeOpacity={strokeOpacity}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeDasharray={m.lineStyle === 'dashed' ? '6 4' : m.lineStyle === 'dotted' ? '2 3' : undefined}
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1], delay: mIdx * 0.08 }}
                  />
                  {BENCHMARK_DIMENSIONS.map((dim, dIdx) => {
                    const x = getTaskX(dIdx);
                    const score = m.scores[dim.key] ?? 0;
                    const y = getY(score);
                    return (
                      <circle
                        key={`pt-${m.id}-${dim.key}`}
                        cx={x}
                        cy={y}
                        r={isHov ? 4.5 : 3}
                        fill="#FFFFFF"
                        stroke={m.color}
                        strokeWidth={isHov ? 2.5 : 1.8}
                        opacity={strokeOpacity}
                        className="transition-all duration-150"
                        onMouseEnter={(e) => {
                          e.stopPropagation();
                          setHoveredSeriesId(m.id);
                          setActivePoint({ title: m.name, dimLabel: dim.label, score, x, y, color: m.color });
                        }}
                        onMouseLeave={() => setActivePoint(null)}
                      />
                    );
                  })}
                  {/* End-of-line Label with collision relaxation */}
                  <motion.g initial={{ opacity: 0, x: -6 }} animate={{ opacity: strokeOpacity, x: 0 }} transition={{ duration: 0.4, delay: 0.5 + mIdx * 0.08 }}>
                    <line x1={lastX + 4} y1={actualY} x2={lastX + 14} y2={relaxedY} stroke={m.color} strokeWidth={1.2} strokeOpacity={strokeOpacity * 0.7} />
                    <text
                      x={lastX + 18}
                      y={relaxedY + 3.5}
                      fill={m.color}
                      className={`font-mono-tabular text-[11px] font-bold ${isHov ? 'underline' : ''}`}
                    >
                      {(m.scores[lastKey] ?? 0).toFixed(1)}%{' '}
                      <tspan className="font-sans font-medium text-[10px]" fill={m.color} opacity={0.88}>
                        {m.name.replace('Helios ', '')}
                      </tspan>
                    </text>
                  </motion.g>
                </g>
              );
            })}

          {/* Interactive Floating Hover Tooltip */}
          {activePoint && (
            <g transform={`translate(${activePoint.x}, ${activePoint.y - 14})`}>
              <rect
                x={-60}
                y={-34}
                width={120}
                height={28}
                rx={6}
                fill="#141312"
                stroke={activePoint.color}
                strokeWidth={1.5}
                filter="drop-shadow(0 4px 10px rgba(0,0,0,0.25))"
              />
              <text x={0} y={-20} textAnchor="middle" fill="#FFFFFF" className="font-mono-tabular text-[10px] font-bold">
                {activePoint.score}% · {activePoint.dimLabel.split(' ')[0]}
              </text>
              <text x={0} y={-10} textAnchor="middle" fill={activePoint.color} className="font-display text-[9px] font-semibold truncate">
                {activePoint.title}
              </text>
            </g>
          )}
        </svg>
      </div>

      {/* Footer Notes & Contamination Clean Badge */}
      <div className="mt-3 pt-3 border-t border-[#141312]/8 flex flex-col sm:flex-row items-start sm:items-center justify-between text-[11px] font-mono-tabular text-[#141312]/60 gap-2">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0" />
          <span>Contamination Clean (0 / 23 slices audited)</span>
        </div>
        <span>Deterministic Seed 42 · Greedy Decoding (T=0.0)</span>
      </div>
    </div>
  );
};
