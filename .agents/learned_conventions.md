# Learned Conventions & Architectural Invariants

## Project: Helios Landing Page Redesign
- **Core Aesthetic**: Minimalist, high contrast, warm parchment (`#F5F0E6`, `#FAF7F2`) and deep obsidian (`#141312`, `#0D0C0B`) with solar amber accent (`#FF5500`, `#E85D2A`).
- **Inspirations**:
  - `getdevlets.com`: Developer-centric clarity, razor-sharp hairline borders, monospace telemetry, modular tactile widgets, zero marketing fluff.
  - `getkeeby.com`: Horizontal scrolling product gallery showcase, smooth tactile physical momentum, custom notch/capsule details, responsive touch fallback.
- **Strict Anti-Slop Policy**:
  - No decorative shields.io badges or synthetic marketing copy.
  - No purple neon glows or gradient soups.
  - No raw emojis in interactive controls; all icons are semantic Lucide SVGs with `aria-hidden="true"`.
  - Zero fake mocks; display genuine architecture parameters (17.84M to 5.80B params, 256K NTK RoPE context, GQA 8:1 ratio, 4-tier Panoptes memory).
- **Responsive Viewport Standards**:
  - Mobile 375px, Tablet 768px, Desktop 1280px, Widescreen 1920px.
  - 44x44px minimum touch targets.
  - Horizontal scrolling must smoothly translate vertical scroll on desktop and offer effortless touch swipe on mobile.
- **Code Hygiene**:
  - Physical ceiling of 500 lines per file (target 150-250 lines).
  - Maximum 40 physical lines per function.
