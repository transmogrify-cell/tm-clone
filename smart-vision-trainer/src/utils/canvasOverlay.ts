/**
 * Canvas overlay for detections (canvas, not DOM: one draw call per frame, no
 * layout work). The canvas has the SOURCE resolution (e.g. 640x480) and is
 * stretched over the video with CSS, so boxes can be drawn in source pixels.
 *
 * Mirroring: the webcam video is mirrored with CSS (selfie view), the canvas is
 * not, so box x-coordinates are flipped here and text stays readable.
 */
import { FACE_COLOR, HAND_COLOR } from '../config';
import { HAND_CONNECTIONS } from '../ml/handDetector';
import type { Box, DetectionFrame, HandDetection } from '../types';

export interface OverlayBox {
  box: Box;
  color: string;
  label?: string;
  /** Dashed outline, for "localized but not confidently classified". */
  dashed?: boolean;
}

export interface DrawOptions {
  mirror?: boolean;
}

/** Fixed colors for the classes people test most, a stable hashed hue for the rest. */
const COCO_COLORS: Record<string, string> = {
  person: '#ef4444',
  car: '#3b82f6',
  bottle: '#22c55e',
  bicycle: '#f97316',
  motorcycle: '#ec4899',
  bus: '#0ea5e9',
  truck: '#6366f1',
  cup: '#14b8a6',
  'cell phone': '#06b6d4',
  chair: '#a16207',
  book: '#84cc16',
  laptop: '#64748b',
  dog: '#d97706',
  cat: '#f59e0b',
};

export function cocoColor(label: string): string {
  if (COCO_COLORS[label]) return COCO_COLORS[label];
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) % 360;
  return `hsl(${h} 75% 50%)`;
}

/** Sizes the canvas to the source (only when it changed, which would reallocate) and clears it. */
export function prepareCanvas(canvas: HTMLCanvasElement, width: number, height: number): CanvasRenderingContext2D {
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, width, height);
  return ctx;
}

export function clearOverlay(canvas: HTMLCanvasElement | null): void {
  canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
}

/** Line width / font size grow with the source resolution so they look the same on screen. */
const unit = (ctx: CanvasRenderingContext2D) => Math.max(1, ctx.canvas.width / 640);

function flipX(ctx: CanvasRenderingContext2D, box: Box, mirror?: boolean): Box {
  return mirror ? { ...box, x: ctx.canvas.width - box.x - box.width } : box;
}

/** Perceived-brightness check so label text is readable on any box color. */
function textOn(color: string): string {
  const probe = document.createElement('canvas').getContext('2d')!;
  probe.fillStyle = color;
  const hex = probe.fillStyle; // normalized to #rrggbb
  if (!hex.startsWith('#') || hex.length !== 7) return '#fff';
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return 0.299 * r + 0.587 * g + 0.114 * b > 160 ? '#0b0b0b' : '#fff';
}

const textColorCache = new Map<string, string>();

export function drawBoxes(ctx: CanvasRenderingContext2D, boxes: OverlayBox[], { mirror }: DrawOptions = {}): void {
  const u = unit(ctx);
  ctx.font = `600 ${Math.round(13 * u)}px Inter, ui-sans-serif, system-ui, sans-serif`;
  ctx.textBaseline = 'middle';
  for (const { box, color, label, dashed } of boxes) {
    const b = flipX(ctx, box, mirror);
    ctx.lineWidth = 2.5 * u;
    ctx.strokeStyle = color;
    ctx.setLineDash(dashed ? [6 * u, 4 * u] : []);
    ctx.strokeRect(b.x, b.y, b.width, b.height);
    ctx.setLineDash([]);
    if (!label) continue;

    const padX = 5 * u;
    const h = 20 * u;
    const w = ctx.measureText(label).width + padX * 2;
    // Above the box, or just inside its top edge when the box touches the top of the frame.
    const y = b.y - h >= 0 ? b.y - h : b.y;
    const x = Math.min(Math.max(0, b.x - ctx.lineWidth / 2), ctx.canvas.width - w);
    ctx.fillStyle = color;
    ctx.fillRect(x, y, w, h);
    if (!textColorCache.has(color)) textColorCache.set(color, textOn(color));
    ctx.fillStyle = textColorCache.get(color)!;
    ctx.fillText(label, x + padX, y + h / 2 + 0.5);
  }
}

/** 21-keypoint hand skeleton: bones as lines, joints as dots. */
export function drawKeypoints(
  ctx: CanvasRenderingContext2D,
  hands: HandDetection[],
  { mirror, color = HAND_COLOR }: DrawOptions & { color?: string } = {},
): void {
  const u = unit(ctx);
  const W = ctx.canvas.width;
  const px = (x: number) => (mirror ? W - x : x);
  for (const hand of hands) {
    const k = hand.keypoints;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * u;
    ctx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.moveTo(px(k[a].x), k[a].y);
      ctx.lineTo(px(k[b].x), k[b].y);
    }
    ctx.stroke();
    for (const p of k) {
      ctx.beginPath();
      ctx.arc(px(p.x), p.y, 3 * u, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.fill();
      ctx.lineWidth = 1.5 * u;
      ctx.stroke();
    }
  }
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** Mode A view: COCO boxes (per-class color), faces (yellow), hands (purple box + skeleton). */
export function drawDetections(canvas: HTMLCanvasElement, frame: DetectionFrame, opts: DrawOptions = {}): void {
  const ctx = prepareCanvas(canvas, frame.width, frame.height);
  drawBoxes(
    ctx,
    [
      ...frame.coco.map((d) => ({ box: d.box, color: cocoColor(d.label), label: `${d.label} ${pct(d.score)}` })),
      ...frame.faces.map((f) => ({ box: f.box, color: FACE_COLOR, label: 'face' })),
      ...frame.hands.map((h) => ({ box: h.box, color: HAND_COLOR, label: `${h.handedness.toLowerCase()} hand ${pct(h.score)}` })),
    ],
    opts,
  );
  drawKeypoints(ctx, frame.hands, opts);
}
