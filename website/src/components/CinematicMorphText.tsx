import React, { useEffect, useRef, useState } from 'react';
import { motion, useInView, animate } from 'framer-motion';

interface AnimatedMorphNumberProps {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  duration?: number;
  className?: string;
}

export const AnimatedMorphNumber: React.FC<AnimatedMorphNumberProps> = ({
  value,
  decimals = 0,
  prefix = '',
  suffix = '',
  duration = 1.6,
  className = '',
}) => {
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true, margin: '-10% 0px -10% 0px' });
  const [displayValue, setDisplayValue] = useState<string>(
    decimals > 0 ? (0).toFixed(decimals) : '0'
  );

  useEffect(() => {
    if (!isInView) return;

    const controls = animate(0, value, {
      duration,
      ease: [0.16, 1, 0.3, 1], // Custom expo-out spring-like curve
      onUpdate: (latest) => {
        if (decimals > 0) {
          setDisplayValue(latest.toFixed(decimals));
        } else {
          setDisplayValue(Math.round(latest).toLocaleString('en-US'));
        }
      },
    });

    return () => controls.stop();
  }, [isInView, value, decimals, duration]);

  return (
    <motion.span
      ref={ref}
      initial={{ opacity: 0, filter: 'blur(8px)', y: 12 }}
      animate={isInView ? { opacity: 1, filter: 'blur(0px)', y: 0 } : {}}
      transition={{ duration: 0.8, ease: 'easeOut' }}
      className={`font-mono-tabular font-light tracking-tight inline-block ${className}`}
    >
      {prefix}
      {displayValue}
      {suffix}
    </motion.span>
  );
};

interface CinematicWordMorphProps {
  text: string;
  className?: string;
  as?: 'h1' | 'h2' | 'h3' | 'p' | 'span';
}

export const CinematicWordMorph: React.FC<CinematicWordMorphProps> = ({
  text,
  className = '',
  as = 'span',
}) => {
  const Component = motion[as] as React.ElementType;

  return (
    <Component
      key={text}
      initial={{ opacity: 0, y: 10, filter: 'blur(6px)' }}
      animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      exit={{ opacity: 0, y: -10, filter: 'blur(6px)' }}
      transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
      className={className}
    >
      {text}
    </Component>
  );
};
