import { SAMPLE_MIME, SAMPLE_QUALITY } from '../config';

export async function loadImage(src: string): Promise<HTMLImageElement> {
  const img = new Image();
  img.decoding = 'async';
  img.src = src;
  await img.decode();
  return img;
}

/** Uploaded photos larger than this (long side) are downscaled once on decode; detectors resize anyway. */
export const MAX_DECODE_SIDE = 1280;

/**
 * Decodes an uploaded file into a canvas (respecting EXIF orientation where
 * the browser does). Every pipeline (COCO-SSD, BlazeFace, Hands, MobileNet)
 * accepts a canvas, so the rest of the app never sees ImageBitmaps.
 */
export async function fileToCanvas(file: Blob, maxSide = MAX_DECODE_SIDE): Promise<HTMLCanvasElement> {
  let source: ImageBitmap | HTMLImageElement;
  try {
    source = await createImageBitmap(file);
  } catch {
    const url = URL.createObjectURL(file);
    try {
      source = await loadImage(url);
    } finally {
      URL.revokeObjectURL(url);
    }
  }
  const w = source instanceof HTMLImageElement ? source.naturalWidth : source.width;
  const h = source instanceof HTMLImageElement ? source.naturalHeight : source.height;
  if (!w || !h) throw new Error('Image has no pixels.');
  const scale = Math.min(1, maxSide / Math.max(w, h));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  if ('close' in source) source.close();
  return canvas;
}

/** Samples are stored as 224x224 JPEG data URLs (thumbnail and model input in one). */
export function canvasToSample(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL(SAMPLE_MIME, SAMPLE_QUALITY);
}

/**
 * Copies the current video frame to a canvas so detection, cropping and
 * classification all work on the exact same pixels. Not mirrored: samples keep
 * the true orientation, like uploaded photos (the flip augmentation covers both).
 */
export function snapshotVideo(video: HTMLVideoElement, canvas = document.createElement('canvas')): HTMLCanvasElement {
  if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
  if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
  canvas.getContext('2d')!.drawImage(video, 0, 0);
  return canvas;
}

export const isVideoReady = (v: HTMLVideoElement | null): v is HTMLVideoElement => !!v && v.readyState >= 2 && v.videoWidth > 0;
