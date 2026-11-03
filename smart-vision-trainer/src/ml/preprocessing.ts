/**
 * Mode B STEP 1: preprocessing, chosen by class type.
 *
 *   Person  -> BlazeFace -> exactly one face -> square crop, 20% margin -> 224x224
 *   Object  -> whole image -> letterbox to 224x224 (aspect kept, black padding)
 *   Unknown -> the letterboxed frame, plus the face crop when there is exactly
 *              one face. Inference is dual-path (face crop AND full frame), so
 *              the rejection class needs examples of what both paths see.
 *
 * Both views end as 224x224 RGB in [0, 255]; normalization to [-1, 1] happens
 * in the MobileNet graph (see mobilenet.ts), identically for both.
 */
import { IMAGE_SIZE } from './mobilenet';
import { cropFace, detectFaces, FACE_FILL, significantFaces, sourceSize, type Detectable } from './faceDetector';
import type { Box, ClassKind, FaceDetection, SampleView } from '../types';

/** Letterbox padding: black, as in the spec (and the export metadata). */
export const LETTERBOX_FILL = 0;

/** Fill value used when augmenting a sample of this view (matches its padding). */
export const fillFor = (view: SampleView): number => (view === 'face' ? FACE_FILL : LETTERBOX_FILL);

/**
 * Fits the source rect (default: whole image) into a 224x224 canvas, keeping
 * the aspect ratio and padding the short side with black.
 */
export function letterbox(
  source: Detectable,
  region?: Box,
  canvas: HTMLCanvasElement = document.createElement('canvas'),
): HTMLCanvasElement {
  const { width: sw, height: sh } = region ?? sourceSize(source);
  const sx = region?.x ?? 0;
  const sy = region?.y ?? 0;
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = `rgb(${LETTERBOX_FILL},${LETTERBOX_FILL},${LETTERBOX_FILL})`;
  ctx.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  if (!sw || !sh) return canvas;
  const scale = IMAGE_SIZE / Math.max(sw, sh);
  const dw = sw * scale;
  const dh = sh * scale;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, sx, sy, sw, sh, (IMAGE_SIZE - dw) / 2, (IMAGE_SIZE - dh) / 2, dw, dh);
  return canvas;
}

/** A detector box grown by `margin` on every side and clipped to the image (for hybrid crops). */
export function expandBox(box: Box, margin: number, width: number, height: number): Box {
  const x = Math.max(0, box.x - box.width * margin);
  const y = Math.max(0, box.y - box.height * margin);
  const x2 = Math.min(width, box.x + box.width * (1 + margin));
  const y2 = Math.min(height, box.y + box.height * (1 + margin));
  return { x, y, width: Math.max(1, x2 - x), height: Math.max(1, y2 - y) };
}

export interface PreparedSample {
  view: SampleView;
  canvas: HTMLCanvasElement;
}

export type PreprocessResult =
  | { status: 'ok'; samples: PreparedSample[] }
  | { status: 'none' }
  | { status: 'multiple'; count: number };

/**
 * Turns one raw image/frame into the training sample(s) for a class of this
 * kind, given the faces BlazeFace found in it (ignored for Object classes).
 */
export function prepareSamples(source: Detectable, kind: ClassKind, faces: FaceDetection[]): PreprocessResult {
  if (kind === 'object') return { status: 'ok', samples: [{ view: 'letterbox', canvas: letterbox(source) }] };
  const found = significantFaces(faces);
  const oneFace = found.length === 1 ? { view: 'face' as const, canvas: cropFace(source, found[0].box) } : null;
  if (kind === 'person') {
    if (oneFace) return { status: 'ok', samples: [oneFace] };
    return found.length === 0 ? { status: 'none' } : { status: 'multiple', count: found.length };
  }
  // Unknown: the full frame always; the face crop too when it is unambiguous.
  return { status: 'ok', samples: [{ view: 'letterbox', canvas: letterbox(source) }, ...(oneFace ? [oneFace] : [])] };
}

/** Same as prepareSamples, running the face detector first when the class needs it. */
export async function preprocessForClass(source: Detectable, kind: ClassKind): Promise<PreprocessResult> {
  return prepareSamples(source, kind, kind === 'object' ? [] : await detectFaces(source));
}
