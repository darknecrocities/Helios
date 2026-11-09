import React, { useState, useEffect } from 'react';
import { motion, useScroll, useSpring, AnimatePresence } from 'framer-motion';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { BrandBelt } from './components/BrandBelt';
import { HeroSection } from './sections/HeroSection';
import { AboutSection } from './sections/AboutSection';
import { FamilySection } from './sections/FamilySection';
import { CapabilitiesSection } from './sections/CapabilitiesSection';
import { ComparisonSection } from './sections/ComparisonSection';
import { PlatformsSection } from './sections/PlatformsSection';
import {
  EditorialReasoningSection,
  EditorialHonestySection,
} from './sections/OpusEditorialSections';
import { PretrainedCurriculumSection } from './sections/PretrainedCurriculumSection';
import { DocsView } from './pages/DocsView';

export const App: React.FC = () => {
  const [activeView, setActiveView] = useState<'landing' | 'docs'>('landing');
  const [activeSection, setActiveSection] = useState<string>('hero');

  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, {
    stiffness: 200,
    damping: 30,
    restDelta: 0.001,
  });

  useEffect(() => {
    if (activeView !== 'landing') return;

    const sectionIds = ['hero', 'about', 'family', 'training', 'capabilities', 'comparison', 'platforms'];
    const handleScroll = () => {
      const scrollPosition = window.scrollY + 260;
      for (const id of sectionIds) {
        const el = document.getElementById(id);
        if (el) {
          const top = el.offsetTop;
          const height = el.offsetHeight;
          if (scrollPosition >= top && scrollPosition < top + height) {
            setActiveSection(id);
            break;
          }
        }
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [activeView]);

  const handleOpenDocs = () => {
    setActiveView('docs');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleNavigateSection = (sectionId: string) => {
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
    <div className="min-h-screen flex flex-col bg-[#FBF9F5] text-[#141312] relative overflow-x-clip">
      <motion.div
        style={{ scaleX }}
        className="fixed top-0 left-0 right-0 h-[2px] bg-[#FF5500] origin-left z-[60]"
      />

      <Navbar
        activeView={activeView}
        setActiveView={setActiveView}
        activeSection={activeSection}
      />

      <AnimatePresence mode="wait">
        {activeView === 'landing' ? (
          <motion.main
            key="landing-view"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex-1"
          >
            <HeroSection onOpenDocs={handleOpenDocs} />
            <AboutSection />
            <BrandBelt />
            <EditorialReasoningSection />
            <FamilySection />
            <PretrainedCurriculumSection />
            <CapabilitiesSection />
            <ComparisonSection />
            <EditorialHonestySection />
            <PlatformsSection onOpenDocs={handleOpenDocs} />
          </motion.main>
        ) : (
          <motion.div
            key="docs-view"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className="flex-1"
          >
            <DocsView onBackToShowcase={() => setActiveView('landing')} />
          </motion.div>
        )}
      </AnimatePresence>

      <Footer onOpenDocs={handleOpenDocs} onNavigateSection={handleNavigateSection} />
    </div>
  );
};

export default App;
