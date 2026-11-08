import React, { useRef } from 'react';
import {
  motion,
  useInView,
  useScroll,
  useVelocity,
  useTransform,
  useSpring,
} from 'framer-motion';

interface KineticHeadingProps {
  text: string;
  className?: string;
  as?: 'h1' | 'h2' | 'h3' | 'div';
  delay?: number;
  highlightWords?: string[];
}

export const KineticHeading: React.FC<KineticHeadingProps> = ({
  text,
  className = '',
  as = 'h2',
  delay = 0,
  highlightWords = [],
}) => {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: false, margin: '-8% 0px -8% 0px' });

  // Scroll velocity skew morphing
  const { scrollY } = useScroll();
  const scrollVelocity = useVelocity(scrollY);
  const rawSkew = useTransform(scrollVelocity, [-2000, 0, 2000], [-2.2, 0, 2.2]);
  const smoothSkew = useSpring(rawSkew, { damping: 30, stiffness: 320, mass: 0.6 });

  const words = text.split(' ');
  const Tag = motion[as] as React.ElementType;

  return (
    <Tag
      ref={ref}
      style={{ skewY: smoothSkew }}
      className={`inline-flex flex-wrap gap-x-[0.28em] gap-y-1 ${className}`}
    >
      {words.map((word, idx) => {
        const cleanWord = word.replace(/[^a-zA-Z0-9Α-Ωα-ω]/g, '');
        const isHighlighted = highlightWords.includes(cleanWord) || highlightWords.includes(word);

        return (
          <span key={idx} className="overflow-hidden inline-block py-0.5">
            <motion.span
              initial={{ y: '105%', rotateZ: 3, opacity: 0, filter: 'blur(6px)' }}
              animate={
                isInView
                  ? { y: '0%', rotateZ: 0, opacity: 1, filter: 'blur(0px)' }
                  : { y: '105%', rotateZ: 3, opacity: 0, filter: 'blur(4px)' }
              }
              transition={{
                type: 'spring',
                damping: 26,
                stiffness: 260,
                mass: 0.8,
                delay: delay + idx * 0.045,
              }}
              className={`inline-block ${isHighlighted ? 'text-[#FF5500]' : ''}`}
            >
              {word}
            </motion.span>
          </span>
        );
      })}
    </Tag>
  );
};

interface ScrollMorphImageProps {
  src: string;
  alt: string;
  className?: string;
  caption?: string;
  greekTag?: string;
}

export const ScrollMorphImage: React.FC<ScrollMorphImageProps> = ({
  src,
  alt,
  className = '',
  caption,
  greekTag,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start end', 'end start'],
  });

  // Smooth spring progress
  const smoothProgress = useSpring(scrollYProgress, {
    damping: 28,
    stiffness: 220,
    mass: 0.7,
  });

  // Image internal scale & vertical parallax
  const imgScale = useTransform(smoothProgress, [0, 0.5, 1], [1.18, 1.0, 1.12]);
  const imgY = useTransform(smoothProgress, [0, 1], ['-8%', '8%']);
  const borderRadius = useTransform(smoothProgress, [0, 0.45, 0.55, 1], ['36px', '14px', '14px', '36px']);

  return (
    <motion.div
      ref={containerRef}
      style={{ borderRadius }}
      className={`relative overflow-hidden border border-[#141312]/15 bg-[#11100F] shadow-2xl group ${className}`}
    >
      <motion.img
        src={src}
        alt={alt}
        style={{ scale: imgScale, y: imgY }}
        className="w-full h-full object-cover object-center will-change-transform transition-all duration-500"
      />
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[#0D0C0B]/80 via-transparent to-transparent" />
      {(caption || greekTag) && (
        <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between text-xs text-[#FBF9F5]/90 font-mono-tabular border-t border-white/15 pt-2.5">
          <span>{caption}</span>
          {greekTag && <span className="text-[#FF5500] uppercase tracking-widest">{greekTag}</span>}
        </div>
      )}
    </motion.div>
  );
};
