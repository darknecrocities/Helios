import React, { useRef, useState } from 'react';
import {
  motion,
  useMotionValue,
  useSpring,
  useTransform,
} from 'framer-motion';

interface MorphingCardProps {
  children: React.ReactNode;
  className?: string;
  dark?: boolean;
  intensity?: number;
  onClick?: () => void;
}

export const MorphingCard: React.FC<MorphingCardProps> = ({
  children,
  className = '',
  dark = false,
  intensity = 7,
  onClick,
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [isHovered, setIsHovered] = useState(false);

  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);
  const glowX = useMotionValue(200);
  const glowY = useMotionValue(200);

  const springConfig = { damping: 28, stiffness: 260, mass: 0.8 };
  const rotateX = useSpring(useTransform(mouseY, [-0.5, 0.5], [intensity, -intensity]), springConfig);
  const rotateY = useSpring(useTransform(mouseX, [-0.5, 0.5], [-intensity, intensity]), springConfig);
  const scale = useSpring(isHovered ? 1.012 : 1, springConfig);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const localX = e.clientX - rect.left;
    const localY = e.clientY - rect.top;

    mouseX.set(localX / rect.width - 0.5);
    mouseY.set(localY / rect.height - 0.5);
    glowX.set(localX);
    glowY.set(localY);
  };

  const handleMouseEnter = () => setIsHovered(true);
  const handleMouseLeave = () => {
    setIsHovered(false);
    mouseX.set(0);
    mouseY.set(0);
  };

  const specularBackground = useTransform(
    [glowX, glowY],
    ([x, y]) =>
      dark
        ? `radial-gradient(520px circle at ${x}px ${y}px, rgba(255, 85, 0, 0.14), transparent 75%)`
        : `radial-gradient(520px circle at ${x}px ${y}px, rgba(255, 85, 0, 0.08), transparent 75%)`
  );

  return (
    <motion.div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={onClick}
      style={{
        rotateX,
        rotateY,
        scale,
        transformStyle: 'preserve-3d',
        perspective: 1100,
      }}
      className={`relative transition-colors duration-300 rounded-3xl overflow-hidden ${
        dark
          ? 'bg-[#161514] border border-white/10 text-[#FAF7F2]'
          : 'bg-[#FAF7F2] border border-[#141312]/10 text-[#141312]'
      } ${className}`}
    >
      <motion.div
        className="pointer-events-none absolute inset-0 z-0 transition-opacity duration-300"
        style={{
          background: specularBackground,
          opacity: isHovered ? 1 : 0,
        }}
      />
      <div className="relative z-10 h-full">{children}</div>
    </motion.div>
  );
};
