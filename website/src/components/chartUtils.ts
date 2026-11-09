/**
 * Geometry and label relaxation utilities for HELIOS SVG charts.
 */

export interface LabelRelaxItem {
  id: string;
  targetY: number;
  label: string;
  color: string;
}

/**
 * Relax overlapping Y coordinates to guarantee minimum vertical separation between labels.
 */
export function relaxLabels(
  items: LabelRelaxItem[],
  minGap = 22,
  minY = 30,
  maxY = 380
): Map<string, number> {
  const sorted = [...items].sort((a, b) => a.targetY - b.targetY);
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].targetY - sorted[i - 1].targetY < minGap) {
      sorted[i].targetY = sorted[i - 1].targetY + minGap;
    }
  }
  // Clamp within bounds from the bottom up
  for (let i = sorted.length - 1; i >= 0; i--) {
    if (sorted[i].targetY > maxY) {
      sorted[i].targetY = maxY;
      if (i > 0 && sorted[i].targetY - sorted[i - 1].targetY < minGap) {
        sorted[i - 1].targetY = sorted[i].targetY - minGap;
      }
    }
  }
  const map = new Map<string, number>();
  sorted.forEach((item) => map.set(item.id, Math.max(minY, item.targetY)));
  return map;
}

/**
 * Generate a smooth cubic bezier SVG path between points.
 */
export function generateCubicBezierPath(points: { x: number; y: number }[]): string {
  if (points.length <= 1) return points.length === 1 ? `M ${points[0].x} ${points[0].y}` : '';
  let d = `M ${points[0].x} ${points[0].y}`;
  for (let i = 0; i < points.length - 1; i++) {
    const p1 = points[i];
    const p2 = points[i + 1];
    const cp1x = p1.x + (p2.x - p1.x) * 0.45;
    const cp2x = p2.x - (p2.x - p1.x) * 0.45;
    d += ` C ${cp1x.toFixed(1)} ${p1.y.toFixed(1)}, ${cp2x.toFixed(1)} ${p2.y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
  }
  return d;
}
