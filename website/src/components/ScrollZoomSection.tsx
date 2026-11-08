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

  // Zoom in from 0.92 -> 1.0 in center -> 1.07 zoom as it transitions to the next section
  const scale = useTransform(smoothProgress, [0, 0.25, 0.75, 1], [0.93, 1, 1, 1.06]);
  const opacity = useTransform(smoothProgress, [0, 0.14, 0.86, 1], [0.35, 1, 1, 0.25]);
  const y = useTransform(smoothProgress, [0, 0.25, 0.75, 1], [40, 0, 0, -40]);

  return (
    <section id={id} ref={ref} className={`relative overflow-hidden ${className}`}>
      <motion.div
        style={{ scale, opacity, y }}
        className="w-full h-full will-change-transform origin-center"
      >
        {children}
      </motion.div>
    </section>
  );
};
