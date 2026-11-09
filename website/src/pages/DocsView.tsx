import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft,
  BookOpen,
  Check,
  Copy,
  Cpu,
  Database,
  Layers,
  Terminal,
  Wrench,
  Search,
  BarChart2,
  Bot,
  ExternalLink,
} from 'lucide-react';
import { HELIOS_DOCS_CHAPTERS } from '../data/docsData';
import {
  OllamaLogo,
  HuggingFaceLogo,
  GoogleColabLogo,
  JupyterLogo,
} from '../components/PlatformLogos';
import { AnimatedMultiLineChart } from '../components/AnimatedMultiLineChart';
import { ModelFrameworkDiagram } from '../components/ModelFrameworkDiagram';

interface DocsViewProps {
  onBackToShowcase: () => void;
}

export const DocsView: React.FC<DocsViewProps> = ({ onBackToShowcase }) => {
  const [activeChapterId, setActiveChapterId] = useState<string>(HELIOS_DOCS_CHAPTERS[0].id);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const activeChapter =
    HELIOS_DOCS_CHAPTERS.find((c) => c.id === activeChapterId) || HELIOS_DOCS_CHAPTERS[0];

  const handleCopy = (code: string, key: string) => {
    navigator.clipboard.writeText(code);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const getChapterIcon = (id: string) => {
    switch (id) {
      case 'journal':
        return <BookOpen className="w-4 h-4" />;
      case 'quickstart':
        return <Terminal className="w-4 h-4" />;
      case 'mcp-memory':
        return <Layers className="w-4 h-4" />;
      case 'techniques':
        return <Cpu className="w-4 h-4" />;
      case 'finetuning':
        return <Wrench className="w-4 h-4" />;
      case 'benchmarks':
        return <BarChart2 className="w-4 h-4" />;
      case 'agents':
        return <Bot className="w-4 h-4" />;
      case 'datasets':
      default:
        return <Database className="w-4 h-4" />;
    }
  };

  const filteredChapters = HELIOS_DOCS_CHAPTERS.filter(
    (c) =>
      c.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      c.summary.toLowerCase().includes(searchQuery.toLowerCase()) ||
      c.sections.some(
        (s) =>
          s.heading.toLowerCase().includes(searchQuery.toLowerCase()) ||
          s.body.toLowerCase().includes(searchQuery.toLowerCase())
      )
  );

  return (
    <div className="min-h-screen pt-28 pb-24 bg-[#FBF9F5] text-[#141312]">
      <div className="max-w-[1360px] mx-auto px-4 sm:px-8 md:px-12">
        {/* Clean Minimalist Header */}
        <div className="pb-8 border-b border-[#141312]/10 flex flex-col lg:flex-row lg:items-end justify-between gap-6">
          <div>
            <button
              onClick={onBackToShowcase}
              className="inline-flex items-center gap-2 text-xs uppercase tracking-widest text-[#FF5500] hover:underline mb-3 cursor-pointer"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back to Showcase</span>
            </button>
            <h1 className="font-display text-3xl sm:text-5xl font-bold tracking-tight text-[#141312]">
              HELIOS DOCUMENTATION & JOURNAL
            </h1>
            <p className="mt-2 text-sm sm:text-base text-[#141312]/70 max-w-2xl">
              Authored by <strong className="text-[#141312]">Arron Kian Parejas</strong> — complete
              Research → Problem → Solution engineering journal, QLoRA 4-bit fine-tuning, Colab recipes, OpenClaw & Hermes agents, and all 177 datasets.
            </p>
          </div>

          {/* Quick Launch Platforms */}
          <div className="flex flex-wrap items-center gap-2.5">
            <a
              href="https://ollama.com/parejasarronkian"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-white border border-[#141312]/15 text-xs font-mono-tabular hover:border-[#FF5500] transition-colors shadow-xs"
            >
              <OllamaLogo size={15} />
              <span>Ollama</span>
            </a>
            <a
              href="https://huggingface.co/Darknecrocities/Helios"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-white border border-[#141312]/15 text-xs font-mono-tabular hover:border-[#FF5500] transition-colors shadow-xs"
            >
              <HuggingFaceLogo size={15} />
              <span>Hugging Face</span>
            </a>
            <button
              onClick={() => setActiveChapterId('techniques')}
              className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-white border border-[#141312]/15 text-xs font-mono-tabular hover:border-[#FF5500] transition-colors shadow-xs cursor-pointer"
            >
              <Cpu className="w-3.5 h-3.5 text-[#FF5500]" />
              <span>Framework Diagram</span>
            </button>
            <button
              onClick={() => setActiveChapterId('finetuning')}
              className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-white border border-[#141312]/15 text-xs font-mono-tabular hover:border-[#FF5500] transition-colors shadow-xs cursor-pointer"
            >
              <GoogleColabLogo size={15} />
              <span>Colab QLoRA</span>
            </button>
            <button
              onClick={() => setActiveChapterId('benchmarks')}
              className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-[#141312] text-white text-xs font-mono-tabular hover:bg-[#FF5500] transition-colors shadow-xs cursor-pointer"
            >
              <BarChart2 className="w-3.5 h-3.5 text-[#FF5500]" />
              <span>Benchmarks</span>
            </button>
          </div>
        </div>

        {/* Workspace Grid */}
        <div className="mt-10 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Sidebar */}
          <aside className="lg:col-span-4 lg:sticky lg:top-24 space-y-4">
            <div className="p-5 rounded-3xl bg-white border border-[#141312]/10 shadow-xs">
              <div className="relative">
                <Search className="w-4 h-4 text-[#141312]/40 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search docs, QLoRA, OpenClaw..."
                  className="w-full pl-10 pr-4 py-2.5 rounded-full bg-[#F5F0E6]/60 border border-[#141312]/10 text-xs focus:outline-none focus:border-[#FF5500]"
                />
              </div>

              <div className="mt-4 space-y-1.5">
                {filteredChapters.map((chapter) => {
                  const isSelected = chapter.id === activeChapter.id;
                  return (
                    <button
                      key={chapter.id}
                      onClick={() => setActiveChapterId(chapter.id)}
                      className={`w-full flex items-start gap-3 p-3 rounded-2xl text-left transition-all cursor-pointer ${
                        isSelected
                          ? 'bg-[#141312] text-[#FAF7F2]'
                          : 'hover:bg-[#141312]/5 text-[#141312]'
                      }`}
                    >
                      <div
                        className={`p-2 rounded-xl shrink-0 mt-0.5 ${
                          isSelected
                            ? 'bg-[#FF5500] text-white'
                            : 'bg-[#141312]/8 text-[#141312]/70'
                        }`}
                      >
                        {getChapterIcon(chapter.id)}
                      </div>
                      <div className="min-w-0">
                        <span className="font-mono-tabular text-[10px] uppercase tracking-widest text-[#FF5500] block truncate">
                          {chapter.romanNumeral} · {chapter.greekTerm}
                        </span>
                        <h3 className="font-display text-sm font-bold mt-0.5 leading-snug truncate">
                          {chapter.title}
                        </h3>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </aside>

          {/* Main Reader */}
          <main className="lg:col-span-8">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeChapter.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                className="rounded-3xl bg-white border border-[#141312]/10 p-6 sm:p-10 space-y-8 shadow-xs"
              >
                {/* Chapter Banner */}
                <div className="pb-6 border-b border-[#141312]/10">
                  <span className="font-mono-tabular text-xs uppercase tracking-widest text-[#FF5500]">
                    {activeChapter.romanNumeral} / {activeChapter.greekTerm}
                  </span>
                  <h2 className="font-display text-2xl sm:text-4xl font-bold text-[#141312] mt-2">
                    {activeChapter.title}
                  </h2>
                  <p className="mt-2 text-sm sm:text-base text-[#141312]/75 leading-relaxed">
                    {activeChapter.summary}
                  </p>
                </div>

                {/* Chapter Specific Injections */}
                {(activeChapter.id === 'techniques' || activeChapter.id === 'finetuning') && (
                  <div className="pb-4">
                    <ModelFrameworkDiagram />
                  </div>
                )}

                {activeChapter.id === 'benchmarks' && (
                  <div className="pb-4 space-y-4">
                    <AnimatedMultiLineChart showBaselinesToggle={true} />
                    <div className="p-4 rounded-2xl bg-[#FBF9F5] border border-[#141312]/10 flex flex-wrap items-center justify-between gap-3">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-xl bg-orange-500/10 text-[#FF5500]">
                          <GoogleColabLogo size={24} />
                        </div>
                        <div>
                          <h4 className="font-display font-bold text-xs text-[#141312]">
                            HELIOS Eval · GPU Colab Benchmark Harness
                          </h4>
                          <p className="text-[11px] text-[#141312]/60 font-mono-tabular">
                            Run automated MMLU, GSM8K, SciQ, HumanEval & BFCL on Colab GPUs.
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <a
                          href="https://colab.research.google.com"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#141312] text-white text-xs font-mono-tabular hover:bg-[#FF5500] transition-colors"
                        >
                          <GoogleColabLogo size={14} />
                          <span>Run on Colab</span>
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                    </div>
                  </div>
                )}

                {activeChapter.id === 'finetuning' && (
                  <div className="p-4 rounded-2xl bg-[#FBF9F5] border border-[#141312]/10 flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-xl bg-amber-500/10 text-amber-600">
                        <GoogleColabLogo size={24} />
                      </div>
                      <div>
                        <h4 className="font-display font-bold text-xs text-[#141312]">
                          Interactive Google Colab & Jupyter Runbooks
                        </h4>
                        <p className="text-[11px] text-[#141312]/60 font-mono-tabular">
                          Run 4-bit QLoRA fine-tuning on a free T4 GPU in one click.
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <a
                        href="https://colab.research.google.com"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#141312] text-white text-xs font-mono-tabular hover:bg-[#FF5500] transition-colors"
                      >
                        <GoogleColabLogo size={14} />
                        <span>Open Colab</span>
                        <ExternalLink className="w-3 h-3" />
                      </a>
                      <button
                        onClick={() => {
                          const el = document.querySelector('pre');
                          if (el) el.scrollIntoView({ behavior: 'smooth' });
                        }}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white border border-[#141312]/20 text-[#141312] text-xs font-mono-tabular hover:border-[#FF5500] transition-colors cursor-pointer"
                      >
                        <JupyterLogo size={14} />
                        <span>Jupyter Cell</span>
                      </button>
                    </div>
                  </div>
                )}

                {/* Chapter Sections */}
                <div className="space-y-8">
                  {activeChapter.sections.map((sec, idx) => {
                    const snippetKey = `${activeChapter.id}-${idx}`;
                    return (
                      <div key={idx} className="space-y-4">
                        <h3 className="font-display text-lg sm:text-xl font-bold text-[#141312]">{sec.heading}</h3>
                        <div className="text-sm sm:text-base text-[#141312]/80 leading-relaxed whitespace-pre-line">{sec.body}</div>
                        {sec.formula && (
                          <div className="p-4 rounded-2xl bg-[#F5F0E6] border border-[#141312]/10 font-mono-tabular text-xs sm:text-sm text-[#141312] overflow-x-auto">
                            <code>{sec.formula}</code>
                          </div>
                        )}
                        {sec.codeSnippet && (
                          <div className="rounded-2xl overflow-hidden bg-[#141312] text-[#FAF7F2]">
                            <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/10 text-xs font-mono-tabular">
                              <span className="text-[#FF5500]">{sec.codeSnippet.label}</span>
                              <button
                                onClick={() => handleCopy(sec.codeSnippet!.code, snippetKey)}
                                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/10 hover:bg-[#FF5500] text-white text-[11px] transition-colors cursor-pointer"
                              >
                                {copiedKey === snippetKey ? (<><Check className="w-3 h-3" /><span>Copied</span></>) : (<><Copy className="w-3 h-3" /><span>Copy</span></>)}
                              </button>
                            </div>
                            <pre className="p-5 text-xs font-mono-tabular leading-relaxed overflow-x-auto text-[#FAF7F2]/90">
                              <code>{sec.codeSnippet.code}</code>
                            </pre>
                          </div>
                        )}
                        {sec.metricsTable && (
                          <div className="overflow-x-auto rounded-2xl border border-[#141312]/10">
                            <table className="w-full text-left border-collapse font-mono-tabular text-xs">
                              <thead>
                                <tr className="bg-[#141312] text-[#FAF7F2] uppercase text-[10px] tracking-wider">
                                  {sec.metricsTable.headers.map((h, i) => (<th key={i} className="py-3.5 px-4">{h}</th>))}
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-[#141312]/10 bg-[#F5F0E6]/60">
                                {sec.metricsTable.rows.map((row, rIdx) => (
                                  <tr key={rIdx} className="hover:bg-[#FF5500]/5">
                                    {row.map((cell, cIdx) => (
                                      <td key={cIdx} className={`py-3 px-4 ${cIdx === 0 ? 'font-bold text-[#141312]' : 'text-[#141312]/80'}`}>
                                        {cell}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </motion.div>
            </AnimatePresence>
          </main>
        </div>
      </div>
    </div>
  );
};
