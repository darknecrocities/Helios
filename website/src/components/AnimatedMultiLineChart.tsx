import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  BENCHMARK_DIMENSIONS,
  HELIOS_FAMILY_BENCHMARK,
  BASELINE_MODELS_BENCHMARK,
  type BenchmarkModelScore,
} from '../data/benchmarkData';

interface AnimatedMultiLineChartProps {
  className?: string;
  showBaselinesToggle?: boolean;
}

export const AnimatedMultiLineChart: React.FC<AnimatedMultiLineChartProps> = ({
  className = '',
  showBaselinesToggle = true,
}) => {
  const [showBaselines, setShowBaselines] = useState<boolean>(false);
  const [hoveredModelId, setHoveredModelId] = useState<string | null>(null);
  const [activePoint, setActivePoint] = useState<{
    modelName: string;
    dimLabel: string;
    score: number;
    x: number;
    y: number;
    color: string;
  } | null>(null);

  const displayModels: BenchmarkModelScore[] = showBaselines
    ? [...HELIOS_FAMILY_BENCHMARK, ...BASELINE_MODELS_BENCHMARK]
    : HELIOS_FAMILY_BENCHMARK;

  const width = 840, height = 440, padL = 56, padR = 175, padT = 40, padB = 48;
  const chartW = width - padL - padR, chartH = height - padT - padB;
  const yTicks = [0, 20, 40, 60, 80, 100];

  const getX = (idx: number) => padL + (idx / (BENCHMARK_DIMENSIONS.length - 1)) * chartW;
  const getY = (score: number) => padT + chartH - (score / 100) * chartH;

  const generateSmoothPath = (scores: Record<string, number>) => {
    const pts = BENCHMARK_DIMENSIONS.map((dim, i) => ({ x: getX(i), y: getY(scores[dim.key] ?? 0) }));
    if (pts.length <= 1) return pts.length === 1 ? `M ${pts[0].x} ${pts[0].y}` : '';
    let d = `M ${pts[0].x} ${pts[0].y}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = i > 0 ? pts[i - 1] : pts[i];
      const p1 = pts[i];
      const p2 = pts[i + 1];
      const p3 = i < pts.length - 2 ? pts[i + 2] : p2;
      const cp1x = p1.x + (p2.x - p0.x) / 6, cp1y = p1.y + (p2.y - p0.y) / 6;
      const cp2x = p2.x - (p3.x - p1.x) / 6, cp2y = p2.y - (p3.y - p1.y) / 6;
      d += ` C ${cp1x.toFixed(1)} ${cp1y.toFixed(1)}, ${cp2x.toFixed(1)} ${cp2y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
    }
    return d;
  };

  return (
    <div className={`w-full rounded-2xl bg-[#FAF8F5] border border-[#141312]/10 p-5 sm:p-7 shadow-sm ${className}`}>
      {/* Chart Header matching reference image */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between pb-4 border-b border-[#141312]/8 gap-4">
        <div>
          <h3 className="font-display text-xl sm:text-2xl font-bold tracking-tight text-[#141312]">
            Standardized Benchmark Trajectory by Model Tier
          </h3>
          <p className="text-xs text-[#141312]/60 mt-1 font-mono-tabular">
            Source: Helios Neural Evaluation Harness | 0% Contamination Audited | SHA-256 ba9724f5...
          </p>
        </div>
        {showBaselinesToggle && (
          <button
            onClick={() => setShowBaselines((prev) => !prev)}
            className={`px-3 py-1.5 rounded-full text-xs font-mono-tabular border transition-all cursor-pointer ${
              showBaselines ? 'bg-[#141312] text-white border-[#141312]' : 'bg-white text-[#141312]/80 border-[#141312]/20 hover:border-[#FF5500]'
            }`}
          >
            {showBaselines ? '✓ Showing Baselines (Qwen / Llama / Gemma)' : '+ Compare External Baselines'}
          </button>
        )}
      </div>

      {/* Filter Pills */}
      <div className="flex flex-wrap items-center gap-2 pt-3 pb-2">
        <span className="text-[11px] font-mono-tabular uppercase tracking-wider text-[#141312]/50 mr-1">Highlight:</span>
        {displayModels.map((m) => {
          const isSel = hoveredModelId === m.id;
          return (
            <button
              key={m.id}
              onClick={() => setHoveredModelId((prev) => (prev === m.id ? null : m.id))}
              onMouseEnter={() => setHoveredModelId(m.id)}
              onMouseLeave={() => setHoveredModelId(null)}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono-tabular transition-all cursor-pointer ${
                isSel ? 'ring-2 ring-offset-1 text-white' : 'bg-white/80 border border-[#141312]/10 text-[#141312]/80 hover:border-[#141312]/30'
              }`}
              style={{ backgroundColor: isSel ? m.color : undefined, color: isSel ? '#FFFFFF' : undefined }}
            >
              <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: m.color }} />
              <span className="font-semibold">{m.name}</span>
            </button>
          );
        })}
      </div>

      {/* Interactive Responsive SVG Stage */}
      <div className="relative w-full aspect-[2.1/1] min-h-[290px] mt-2 select-none">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
          {/* Subtle horizontal grid lines */}
          {yTicks.map((tick) => {
            const y = getY(tick);
            return (
              <g key={`ytick-${tick}`}>
                <line x1={padL} y1={y} x2={padL + chartW} y2={y} stroke="#141312" strokeOpacity={tick === 0 ? 0.25 : 0.07} strokeWidth={tick === 0 ? 1.5 : 1} strokeDasharray={tick === 0 ? undefined : '3 3'} />
                <text x={padL - 10} y={y + 4} textAnchor="end" className="font-mono-tabular text-[11px] fill-[#141312]/60 font-medium">{tick}%</text>
              </g>
            );
          })}
          {/* Left Y-Axis Spine */}
          <line x1={padL} y1={padT - 10} x2={padL} y2={padT + chartH} stroke="#141312" strokeWidth={1.5} strokeOpacity={0.4} />

          {/* Bottom X-Axis Dimension Labels */}
          {BENCHMARK_DIMENSIONS.map((dim, idx) => {
            const x = getX(idx);
            return (
              <g key={`xdim-${dim.key}`}>
                <line x1={x} y1={padT + chartH} x2={x} y2={padT + chartH + 5} stroke="#141312" strokeWidth={1.2} strokeOpacity={0.3} />
                <text x={x} y={padT + chartH + 20} textAnchor="middle" className="font-mono-tabular text-[11px] fill-[#141312]/80 font-medium">{dim.key}</text>
                <text x={x} y={padT + chartH + 32} textAnchor="middle" className="font-mono-tabular text-[9px] fill-[#141312]/50">{dim.label.replace(/.*\((.*)\)/, '$1')}</text>
              </g>
            );
          })}

          {/* Benchmark Lines for Each Model */}
          {displayModels.map((m, mIdx) => {
            const isDim = hoveredModelId !== null && hoveredModelId !== m.id;
            const isHov = hoveredModelId === m.id;
            const strokeWidth = isHov ? 3.5 : isDim ? 1.5 : 2.6;
            const strokeOpacity = isDim ? 0.22 : 1;
            const pathD = generateSmoothPath(m.scores);
            const lastIdx = BENCHMARK_DIMENSIONS.length - 1;
            const lastX = getX(lastIdx);
            const lastScore = m.scores[BENCHMARK_DIMENSIONS[lastIdx].key] ?? 0;
            const lastY = getY(lastScore);

            return (
              <g key={`model-line-${m.id}`} className="transition-all duration-200 cursor-pointer" onMouseEnter={() => setHoveredModelId(m.id)} onMouseLeave={() => setHoveredModelId(null)}>
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
                  const x = getX(dIdx), score = m.scores[dim.key] ?? 0, y = getY(score);
                  return (
                    <circle
                      key={`pt-${m.id}-${dim.key}`}
                      cx={x} cy={y} r={isHov ? 4.5 : 3}
                      fill="#FFFFFF" stroke={m.color} strokeWidth={isHov ? 2.5 : 1.8} opacity={strokeOpacity}
                      className="transition-all duration-150"
                      onMouseEnter={(e) => {
                        e.stopPropagation();
                        setHoveredModelId(m.id);
                        setActivePoint({ modelName: m.name, dimLabel: dim.label, score, x, y, color: m.color });
                      }}
                      onMouseLeave={() => setActivePoint(null)}
                    />
                  );
                })}
                {/* End-of-line Label directly inspired by user reference image */}
                <motion.g initial={{ opacity: 0, x: -6 }} animate={{ opacity: strokeOpacity, x: 0 }} transition={{ duration: 0.4, delay: 0.6 + mIdx * 0.08 }}>
                  <line x1={lastX + 4} y1={lastY} x2={lastX + 12} y2={lastY} stroke={m.color} strokeWidth={1.2} strokeOpacity={strokeOpacity} />
                  <text x={lastX + 16} y={lastY + 3.5} fill={m.color} className={`font-mono-tabular text-[11px] font-bold ${isHov ? 'underline' : ''}`}>
                    {lastScore.toFixed(1)}% <tspan className="font-sans font-medium text-[10px]" fill={m.color} opacity={0.88}>{m.name.replace('Helios ', '')}</tspan>
                  </text>
                </motion.g>
              </g>
            );
          })}

          {/* Interactive Floating Hover Tooltip */}
          {activePoint && (
            <g transform={`translate(${activePoint.x}, ${activePoint.y - 12})`}>
              <rect x={-50} y={-34} width={100} height={28} rx={6} fill="#141312" stroke={activePoint.color} strokeWidth={1.5} filter="drop-shadow(0 4px 10px rgba(0,0,0,0.25))" />
              <text x={0} y={-20} textAnchor="middle" fill="#FFFFFF" className="font-mono-tabular text-[10px] font-bold">{activePoint.score}% · {activePoint.dimLabel.split(' ')[0]}</text>
              <text x={0} y={-10} textAnchor="middle" fill={activePoint.color} className="font-display text-[9px] font-semibold">{activePoint.modelName}</text>
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
