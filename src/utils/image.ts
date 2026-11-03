import * as tf from '@tensorflow/tfjs';
import { IMAGE_SIZE } from '../ml/mobilenet';
import { SAMPLE_MIME, SAMPLE_QUALITY } from '../config';

/** Anything we can draw onto a canvas and feed to MobileNet. */
export type ImageLike = HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageBitmap;

export function sourceSize(source: ImageLike): { width: number; height: number } {
  if (source instanceof HTMLVideoElement) return { width: source.videoWidth, height: source.videoHeight };
  if (source instanceof HTMLImageElement) return { width: source.naturalWidth, height: source.naturalHeight };
  return { width: source.width, height: source.height };
}

/**
 * Center-crops `source` to a square and scales it to `size` x `size`.
 * MobileNet expects a square input; cropping (instead of squashing) keeps the
 * object's proportions, which is what Teachable Machine does too.
 *
 * `mirror` flips horizontally, used for webcam frames so that what is stored
 * and classified matches the mirrored preview the user sees.
 */
export function cropToSquare(
  source: ImageLike,
  size = IMAGE_SIZE,
  opts: { mirror?: boolean; canvas?: HTMLCanvasElement } = {},
): HTMLCanvasElement {
  const { width: sw, height: sh } = sourceSize(source);
  if (!sw || !sh) throw new Error('The image has no pixels yet.');
  const side = Math.min(sw, sh);
  const sx = (sw - side) / 2;
  const sy = (sh - side) / 2;

  const canvas = opts.canvas ?? document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D is not available in this browser.');
  ctx.save();
  if (opts.mirror) {
    ctx.translate(size, 0);
    ctx.scale(-1, 1);
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, sx, sy, side, side, 0, 0, size, size);
  ctx.restore();
  return canvas;
}

/**
 * MobileNet V2 preprocessing: map uint8 pixels [0, 255] to floats in [-1, 1]
 * (x / 127.5 - 1). Identical to `tf.keras.applications.mobilenet_v2.preprocess_input`.
 */
export function normalize(pixels: tf.Tensor): tf.Tensor {
  return tf.tidy(() => tf.cast(pixels, 'float32').div(127.5).sub(1));
}

// One reusable canvas for tensor conversion. tf.browser.fromPixels reads it
// synchronously, so sharing it between calls is safe.
let scratch: HTMLCanvasElement | null = null;

/**
 * Image -> normalized [224, 224, 3] float tensor, ready for MobileNet.
 * Non-square or differently sized inputs are center-cropped and resized first.
 * The caller owns (and must dispose) the returned tensor.
 */
export function tensorFromImage(source: ImageLike, opts: { mirror?: boolean } = {}): tf.Tensor3D {
  return tf.tidy(() => {
    const { width, height } = sourceSize(source);
    const exact = width === IMAGE_SIZE && height === IMAGE_SIZE && !opts.mirror;
    const pixelSource = exact
      ? source
      : cropToSquare(source, IMAGE_SIZE, { mirror: opts.mirror, canvas: (scratch ??= document.createElement('canvas')) });
    const pixels = tf.browser.fromPixels(pixelSource, 3);
    return normalize(pixels) as tf.Tensor3D;
  });
}

export async function loadImage(src: string): Promise<HTMLImageElement> {
  const img = new Image();
  img.decoding = 'async';
  img.src = src;
  await img.decode();
  return img;
}

/** Stored samples are 224x224 JPEG data URLs: small, and already model-sized. */
function canvasToSample(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL(SAMPLE_MIME, SAMPLE_QUALITY);
}

/** Decodes an uploaded file and returns it as a cropped 224x224 sample. */
export async function fileToSample(file: File): Promise<string> {
  let image: ImageBitmap | HTMLImageElement;
  try {
    image = await createImageBitmap(file);
  } catch {
    // Some formats (e.g. SVG) are not supported by createImageBitmap.
    const url = URL.createObjectURL(file);
    try {
      image = await loadImage(url);
    } finally {
      URL.revokeObjectURL(url);
    }
  }
  const canvas = cropToSquare(image);
  if ('close' in image) image.close();
  return canvasToSample(canvas);
}

let captureCanvas: HTMLCanvasElement | null = null;

/** Grabs the current webcam frame (mirrored, like the preview) as a sample. */
export function videoFrameToSample(video: HTMLVideoElement): string {
  const canvas = cropToSquare(video, IMAGE_SIZE, {
    mirror: true,
    canvas: (captureCanvas ??= document.createElement('canvas')),
  });
  return canvasToSample(canvas);
}
