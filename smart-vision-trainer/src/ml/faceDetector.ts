/**
 * Mode A face boxes + Mode B STEP 1 (Person classes): face detection and crop.
 *
 * A full photo is mostly background, clothes and lighting; MobileNet features
 * of the whole frame describe those as much as the person. Cropping to the
 * face first makes every Person sample "the same kind of picture", so the head
 * learns identity rather than scenery.
 *
 * Detector: MediaPipe BlazeFace (short range) via @tensorflow-models/face-detection,
 * tfjs runtime, model served from public/models/blazeface.
 */
import * as faceDetection from '@tensorflow-models/face-detection';
import { IMAGE_SIZE, initBackend, modelLoadError, MODELS_BASE } from './mobilenet';
import type { Box, FaceDetection } from '../types';

export const BLAZEFACE_URL = `${MODELS_BASE}/blazeface/model.json`;

/** Crop = face box grown by 20% of its size on every side, then squared. */
export const FACE_MARGIN = 0.2;

/**
 * Background faces much smaller than the main one (a poster, someone far
 * behind) don't count towards "multiple faces" when collecting samples.
 */
export const SECONDARY_FACE_MIN_RATIO = 0.45;

/** Out-of-frame parts of a face crop are filled with mid-gray (also used by augmentation). */
export const FACE_FILL = 128;

export type Detectable = HTMLImageElement | HTMLVideoElement | HTMLCanvasElement;

let detectorPromise: Promise<faceDetection.FaceDetector> | null = null;

export function loadFaceDetector(): Promise<faceDetection.FaceDetector> {
  detectorPromise ??= (async () => {
    await initBackend();
    let detector: faceDetection.FaceDetector;
    try {
      detector = await faceDetection.createDetector(faceDetection.SupportedModels.MediaPipeFaceDetector, {
        runtime: 'tfjs',
        modelType: 'short',
        maxFaces: 5, // need more than 1 to notice multi-face photos
        detectorModelUrl: BLAZEFACE_URL,
      });
    } catch (err) {
      throw modelLoadError('the face detector (BlazeFace)', BLAZEFACE_URL, err);
    }
    // Warm-up compiles shaders so the first real frame isn't slow.
    const blank = document.createElement('canvas');
    blank.width = blank.height = 128;
    await detector.estimateFaces(blank);
    return detector;
  })().catch((err) => {
    detectorPromise = null;
    throw err;
  });
  return detectorPromise;
}

export function sourceSize(source: Detectable): { width: number; height: number } {
  if (source instanceof HTMLVideoElement) return { width: source.videoWidth, height: source.videoHeight };
  if (source instanceof HTMLImageElement) return { width: source.naturalWidth, height: source.naturalHeight };
  return { width: source.width, height: source.height };
}

/** All faces, largest first. */
export async function detectFaces(source: Detectable): Promise<FaceDetection[]> {
  const detector = await loadFaceDetector();
  const faces = await detector.estimateFaces(source, { flipHorizontal: false });
  return faces
    .map(({ box }) => ({ box: { x: box.xMin, y: box.yMin, width: box.width, height: box.height }, score: 1 }))
    .filter((f) => f.box.width > 0 && f.box.height > 0)
    .sort((a, b) => b.box.width * b.box.height - a.box.width * a.box.height);
}

/** Faces that matter: the largest, plus any at least SECONDARY_FACE_MIN_RATIO of its width. */
export function significantFaces(faces: FaceDetection[]): FaceDetection[] {
  if (faces.length === 0) return faces;
  const main = faces[0].box.width;
  return faces.filter((f) => f.box.width >= main * SECONDARY_FACE_MIN_RATIO);
}

/** The square crop region (in source pixels) for a face box: 20% margin per side. */
export function cropRegion(box: Box) {
  const side = Math.max(box.width, box.height) * (1 + 2 * FACE_MARGIN);
  return { sx: box.x + box.width / 2 - side / 2, sy: box.y + box.height / 2 - side / 2, side };
}

/** Draws the face crop into a 224x224 canvas (gray where the margin leaves the frame). */
export function cropFace(source: Detectable, box: Box, canvas = document.createElement('canvas')): HTMLCanvasElement {
  const { sx, sy, side } = cropRegion(box);
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = `rgb(${FACE_FILL},${FACE_FILL},${FACE_FILL})`;
  ctx.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  // The source rect may extend past the image; drawImage clips it and keeps the scale.
  ctx.drawImage(source, sx, sy, side, side, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  return canvas;
}

export type FaceCropResult =
  | { status: 'ok'; crop: HTMLCanvasElement; box: Box; faceCount: number }
  | { status: 'none' }
  | { status: 'multiple'; count: number };

/**
 * Detects faces and crops one.
 *  - 'strict' (training data): exactly one significant face, otherwise rejected.
 *  - 'largest' (inference): crop the largest face, report how many there were.
 * Pass `faces` to reuse a detection that already ran on the same pixels.
 */
export async function detectAndCropFace(
  source: Detectable,
  mode: 'strict' | 'largest' = 'strict',
  faces?: FaceDetection[],
): Promise<FaceCropResult> {
  const { width, height } = sourceSize(source);
  if (!width || !height) return { status: 'none' };
  const found = significantFaces(faces ?? (await detectFaces(source)));
  if (found.length === 0) return { status: 'none' };
  if (found.length > 1 && mode === 'strict') return { status: 'multiple', count: found.length };
  return { status: 'ok', crop: cropFace(source, found[0].box), box: found[0].box, faceCount: found.length };
}
