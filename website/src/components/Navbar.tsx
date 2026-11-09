import React, { useState, useEffect } from 'react';
import { motion, useScroll, useTransform, useSpring, AnimatePresence } from 'framer-motion';
import { Menu, X } from 'lucide-react';
import { OllamaLogo, HuggingFaceLogo } from './PlatformLogos';

interface NavbarProps {
  activeView: 'landing' | 'docs';
  setActiveView: (view: 'landing' | 'docs') => void;
  activeSection: string;
}

const NAV_LINKS = [
  { id: 'hero', label: 'Helios' },
  { id: 'about', label: 'About' },
  { id: 'family', label: 'Models' },
  { id: 'capabilities', label: 'Capabilities' },
  { id: 'comparison', label: 'Comparison' },
  { id: 'platforms', label: 'Deploy' },
];

export const Navbar: React.FC<NavbarProps> = ({
  activeView,
  setActiveView,
  activeSection,
}) => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const { scrollY } = useScroll();
  const smoothScrollY = useSpring(scrollY, { damping: 30, stiffness: 260, mass: 0.7 });

  const navMaxWidth = useTransform(smoothScrollY, [0, 140], [1320, 1080]);
  const navTopOffset = useTransform(smoothScrollY, [0, 140], [0, 14]);
  const navBorderRadius = useTransform(smoothScrollY, [0, 140], [0, 9999]);
  const navPaddingY = useTransform(smoothScrollY, [0, 140], [18, 10]);
  const navPaddingX = useTransform(smoothScrollY, [0, 140], [28, 20]);

  useEffect(() => {
    const unsubscribe = scrollY.on('change', (latest) => {
      setIsScrolled(latest > 30);
    });
    return () => unsubscribe();
  }, [scrollY]);

  const handleSectionNavigate = (sectionId: string) => {
    setMobileMenuOpen(false);
    if (activeView !== 'landing') {
      setActiveView('landing');
      setTimeout(() => {
        const el = document.getElementById(sectionId);
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } else {
      const el = document.getElementById(sectionId);
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 flex justify-center pointer-events-none px-4">
      <motion.nav
        style={{
          maxWidth: navMaxWidth,
          marginTop: navTopOffset,
          borderRadius: navBorderRadius,
          paddingTop: navPaddingY,
          paddingBottom: navPaddingY,
          paddingLeft: navPaddingX,
          paddingRight: navPaddingX,
        }}
        className={`pointer-events-auto w-full transition-colors duration-300 flex items-center justify-between gap-4 ${
          isScrolled
            ? 'bg-[#141312]/92 text-[#FAF7F2] backdrop-blur-xl border border-white/10 shadow-[0_16px_40px_rgba(20,19,18,0.22)]'
            : activeView === 'docs'
            ? 'bg-[#141312]/95 text-[#FAF7F2] backdrop-blur-xl border border-white/10 shadow-lg'
            : 'bg-transparent text-[#FAF7F2]'
        }`}
      >
        {/* Brand Identity */}
        <button
          onClick={() => handleSectionNavigate('hero')}
          className="flex items-center gap-3 focus:outline-none cursor-pointer shrink-0"
        >
          <div className="w-8 h-8 rounded-full overflow-hidden border border-[#FF5500]/50 bg-[#141312] shrink-0">
            <img
              src="/assets/helios_logo.png"
              alt="Helios Logo"
              className="w-full h-full object-cover"
            />
          </div>
          <span className="font-display font-bold tracking-[0.22em] text-sm sm:text-base">
            HELIOS
          </span>
        </button>

        {/* Center Minimalist Links */}
        <div className="hidden lg:flex items-center gap-1 shrink-0">
          {NAV_LINKS.map((item) => {
            const isActive = activeView === 'landing' && activeSection === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleSectionNavigate(item.id)}
                className={`relative px-3.5 py-1.5 rounded-full text-xs font-mono-tabular tracking-wider uppercase transition-colors cursor-pointer whitespace-nowrap ${
                  isActive
                    ? 'text-white font-bold'
                    : 'text-[#FAF7F2]/70 hover:text-white'
                }`}
              >
                {isActive && (
                  <motion.span
                    layoutId="navMorphActivePill"
                    transition={{ type: 'spring', stiffness: 320, damping: 28 }}
                    className="absolute inset-0 rounded-full bg-[#FF5500] -z-10"
                  />
                )}
                {item.label}
              </button>
            );
          })}
        </div>

        {/* Right: Platform Repos & Docs Switcher */}
        <div className="hidden sm:flex items-center gap-2 shrink-0">
          <a
            href="https://ollama.com/parejasarronkian"
            target="_blank"
            rel="noopener noreferrer"
            title="Ollama"
            className="p-2 rounded-full text-white/70 hover:text-white hover:bg-white/10 transition-colors"
          >
            <OllamaLogo size={16} />
          </a>
          <a
            href="https://huggingface.co/Darknecrocities/Helios"
            target="_blank"
            rel="noopener noreferrer"
            title="Hugging Face"
            className="p-2 rounded-full text-white/70 hover:text-white hover:bg-white/10 transition-colors"
          >
            <HuggingFaceLogo size={16} />
          </a>

          <button
            onClick={() => {
              setActiveView(activeView === 'docs' ? 'landing' : 'docs');
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            className="px-4 py-1.5 rounded-full text-xs font-mono-tabular font-bold tracking-wider uppercase bg-[#FF5500] text-white hover:bg-[#E64A00] transition-all cursor-pointer active:scale-95"
          >
            {activeView === 'docs' ? 'Showcase' : 'Docs'}
          </button>
        </div>

        {/* Mobile Menu Trigger */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label="Toggle navigation menu"
          className="lg:hidden p-2 cursor-pointer shrink-0 text-white min-w-[44px] min-h-[44px] flex items-center justify-center"
        >
          {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </motion.nav>

      {/* Mobile Drawer Sheet */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -16 }}
            className="pointer-events-auto fixed top-20 left-4 right-4 z-50 rounded-2xl bg-[#141312] text-[#FAF7F2] border border-white/10 p-5 shadow-2xl lg:hidden backdrop-blur-2xl"
          >
            <div className="flex flex-col gap-2">
              {NAV_LINKS.map((item) => (
                <button
                  key={item.id}
                  onClick={() => handleSectionNavigate(item.id)}
                  className="px-4 py-3 rounded-xl hover:bg-white/10 text-left text-sm font-mono-tabular tracking-wider uppercase min-h-[44px]"
                >
                  {item.label}
                </button>
              ))}

              <div className="pt-2 border-t border-white/10 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <a
                    href="https://ollama.com/parejasarronkian"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 rounded-lg bg-white/5 text-white"
                  >
                    <OllamaLogo size={18} />
                  </a>
                  <a
                    href="https://huggingface.co/Darknecrocities/Helios"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 rounded-lg bg-white/5 text-white"
                  >
                    <HuggingFaceLogo size={18} />
                  </a>
                </div>

                <button
                  onClick={() => {
                    setMobileMenuOpen(false);
                    setActiveView(activeView === 'docs' ? 'landing' : 'docs');
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                  }}
                  className="px-4 py-2 rounded-xl bg-[#FF5500] text-white text-xs font-mono-tabular font-bold uppercase tracking-wider min-h-[44px]"
                >
                  {activeView === 'docs' ? 'Showcase' : 'Docs'}
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
};
