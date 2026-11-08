import React from 'react';
import { GreekDeityTierId } from '../types/models';

interface SealProps {
  tierId: GreekDeityTierId;
  className?: string;
  size?: number;
}

export const BespokeSeal: React.FC<SealProps> = ({ tierId, className = '', size = 56 }) => {
  switch (tierId) {
    case 'nano-20m':
      // Helios Aether: Primordial Diamond & Radiating Light Shards
      return (
        <svg
          width={size}
          height={size}
          viewBox="0 0 64 64"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className={className}
          aria-hidden="true"
        >
          <circle cx="32" cy="32" r="28" stroke="currentColor" strokeWidth="1" strokeDasharray="2 3" opacity="0.4" />
          <polygon
            points="32,8 54,32 32,56 10,32"
            stroke="#FF5500"
            strokeWidth="1.75"
            fill="rgba(255,85,0,0.08)"
          />
          <polygon points="32,16 46,32 32,48 18,32" stroke="currentColor" strokeWidth="1.25" />
          <line x1="32" y1="2" x2="32" y2="12" stroke="#FF5500" strokeWidth="1.75" />
          <line x1="32" y1="52" x2="32" y2="62" stroke="#FF5500" strokeWidth="1.75" />
          <line x1="2" y1="32" x2="12" y2="32" stroke="#FF5500" strokeWidth="1.75" />
          <line x1="52" y1="32" x2="62" y2="32" stroke="#FF5500" strokeWidth="1.75" />
          <circle cx="32" cy="32" r="4" fill="#FF5500" />
        </svg>
      );

    case 'alpha-100m':
      // Helios Phaethon: 4-Spoke Solar Chariot Wheel & Kinetic Ring
      return (
        <svg
          width={size}
          height={size}
          viewBox="0 0 64 64"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className={className}
          aria-hidden="true"
        >
          <circle cx="32" cy="32" r="27" stroke="currentColor" strokeWidth="1.5" />
          <circle cx="32" cy="32" r="21" stroke="#FF5500" strokeWidth="1.5" fill="rgba(255,85,0,0.06)" />
          <line x1="32" y1="5" x2="32" y2="59" stroke="currentColor" strokeWidth="1.75" />
          <line x1="5" y1="32" x2="59" y2="32" stroke="currentColor" strokeWidth="1.75" />
          <line x1="13" y1="13" x2="51" y2="51" stroke="#FF5500" strokeWidth="1.25" />
          <line x1="51" y1="13" x2="13" y2="51" stroke="#FF5500" strokeWidth="1.25" />
          <circle cx="32" cy="32" r="7" fill="#141312" stroke="#FF5500" strokeWidth="2" />
          <circle cx="32" cy="32" r="2.5" fill="#FF5500" />
        </svg>
      );

    case 'beta-300m':
      // Helios Eos: Winged Solar Corona Rising Over Architectural Horizon
      return (
        <svg
          width={size}
          height={size}
          viewBox="0 0 64 64"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className={className}
          aria-hidden="true"
        >
          <path d="M6 42H58" stroke="currentColor" strokeWidth="2" />
          <path d="M12 48H52" stroke="#FF5500" strokeWidth="1.5" />
          <path d="M20 54H44" stroke="currentColor" strokeWidth="1" opacity="0.6" />
          <path
            d="M16 42C16 33.1634 23.1634 26 32 26C40.8366 26 48 33.1634 48 42"
            fill="rgba(255,85,0,0.14)"
            stroke="#FF5500"
            strokeWidth="2"
          />
          <path d="M32 10V22" stroke="#FF5500" strokeWidth="2" />
          <path d="M16 16L23 25" stroke="currentColor" strokeWidth="1.5" />
          <path d="M48 16L41 25" stroke="currentColor" strokeWidth="1.5" />
          <path d="M8 28L18 33" stroke="#FF5500" strokeWidth="1.5" />
          <path d="M56 28L46 33" stroke="#FF5500" strokeWidth="1.5" />
          <circle cx="32" cy="36" r="4" fill="#FF5500" />
        </svg>
      );

    case 'core-1b':
      // Helios Hyperion: All-Seeing Geometric Eye Inscribed in Dodecagon
      return (
        <svg
          width={size}
          height={size}
          viewBox="0 0 64 64"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className={className}
          aria-hidden="true"
        >
          <polygon
            points="32,4 46,8 56,18 60,32 56,46 46,56 32,60 18,56 8,46 4,32 8,18 18,8"
            stroke="currentColor"
            strokeWidth="1.25"
            fill="rgba(255,85,0,0.05)"
          />
          <path
            d="M10 32C16 20 24 16 32 16C40 16 48 20 54 32C48 44 40 48 32 48C24 48 16 44 10 32Z"
            stroke="#FF5500"
            strokeWidth="1.75"
          />
          <circle cx="32" cy="32" r="9" stroke="currentColor" strokeWidth="1.5" />
          <circle cx="32" cy="32" r="4.5" fill="#FF5500" />
        </svg>
      );

    case 'pro-3b':
      // Helios Apollo: Classical Harmonic Lyre & Solar Laurel Crown
      return (
        <svg
          width={size}
          height={size}
          viewBox="0 0 64 64"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className={className}
          aria-hidden="true"
        >
          <circle cx="32" cy="32" r="28" stroke="currentColor" strokeWidth="1" opacity="0.4" />
          <path
            d="M18 16C14 26 16 40 24 48H40C48 40 50 26 46 16"
            stroke="#FF5500"
            strokeWidth="2"
            strokeLinecap="round"
          />
          <path d="M22 48L20 54H44L42 48" stroke="currentColor" strokeWidth="1.75" />
          <line x1="28" y1="18" x2="28" y2="48" stroke="currentColor" strokeWidth="1.25" />
          <line x1="32" y1="14" x2="32" y2="48" stroke="#FF5500" strokeWidth="1.5" />
          <line x1="36" y1="18" x2="36" y2="48" stroke="currentColor" strokeWidth="1.25" />
          <line x1="20" y1="22" x2="44" y2="22" stroke="currentColor" strokeWidth="1.75" />
          <circle cx="32" cy="11" r="3" fill="#FF5500" />
        </svg>
      );

    case 'flagship-6b':
    default:
      // Helios Sol Invictus: 16-Ray Macedonian-Hellenic Sun Star
      return (
        <svg
          width={size}
          height={size}
          viewBox="0 0 64 64"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className={className}
          aria-hidden="true"
        >
          <circle cx="32" cy="32" r="29" stroke="currentColor" strokeWidth="1.25" />
          <circle cx="32" cy="32" r="25" stroke="#FF5500" strokeWidth="1" strokeDasharray="3 2" />
          {/* Primary 8 Solar Rays */}
          <path
            d="M32 4L34.5 22L32 26L29.5 22L32 4Z
               M32 60L34.5 42L32 38L29.5 42L32 60Z
               M4 32L22 29.5L26 32L22 34.5L4 32Z
               M60 32L42 29.5L38 32L42 34.5L60 32Z
               M12.2 12.2L26 23.5L27.5 27.5L23.5 26L12.2 12.2Z
               M51.8 51.8L38 40.5L36.5 36.5L40.5 38L51.8 51.8Z
               M51.8 12.2L40.5 26L36.5 27.5L38 23.5L51.8 12.2Z
               M12.2 51.8L23.5 38L27.5 36.5L26 40.5L12.2 51.8Z"
            fill="#FF5500"
          />
          <circle cx="32" cy="32" r="8" fill="#141312" stroke="#FF5500" strokeWidth="2" />
          <circle cx="32" cy="32" r="3.5" fill="#FF5500" />
        </svg>
      );
  }
};
