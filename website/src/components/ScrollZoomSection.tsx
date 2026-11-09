import React, { useRef } from 'react';
import { motion, useScroll, useTransform, useSpring } from 'framer-motion';

interface ScrollZoomSectionProps {
  id?: string;
  children: React.ReactNode;
  className?: string;
}

export const ScrollZoomSection: React.FC<ScrollZoomSectionProps> = ({
  id,
  children,
  className = '',
}) => {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start end', 'end start'],
  });

  const smoothProgress = useSpring(scrollYProgress, {
    damping: 30,
    stiffness: 220,
    mass: 0.7,
  });

  // Subtle zoom scaling without aggressive opacity dropoffs
  const scale = useTransform(smoothProgress, [0, 0.2, 0.8, 1], [0.96, 1, 1, 1.03]);
  const opacity = useTransform(smoothProgress, [0, 0.15, 0.85, 1], [0.85, 1, 1, 0.85]);

  return (
    <section id={id} ref={ref} className={`relative overflow-hidden ${className}`}>
      <motion.div
        style={{ scale, opacity }}
        className="w-full h-full will-change-transform origin-center"
      >
        {children}
      </motion.div>
    </section>
  );
};
