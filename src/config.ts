// App-wide limits and defaults. ML constants that describe the base model live
// in src/ml/mobilenet.ts; these are product/UX knobs.

export const MAX_CLASSES = 10;
export const MIN_CLASSES = 2;

/** A class needs at least this many samples before "Train" unlocks. */
export const MIN_SAMPLES_PER_CLASS = 5;

/** Hard cap per class so a long webcam hold can't exhaust memory. */
export const MAX_SAMPLES_PER_CLASS = 500;

/** Webcam "Hold to Record" capture rate. */
export const CAPTURE_FPS = 10;

/** Thumbnail / stored-sample encoding. Samples are stored already cropped to 224x224. */
export const SAMPLE_MIME = 'image/jpeg';
export const SAMPLE_QUALITY = 0.92;

/** Images are pushed through MobileNet this many at a time. */
export const FEATURE_BATCH_SIZE = 16;

/** One accent per class slot (cards and preview bars share it). */
export const CLASS_COLORS = [
  '#f97316', '#ec4899', '#3b82f6', '#10b981', '#a855f7',
  '#eab308', '#06b6d4', '#ef4444', '#84cc16', '#6366f1',
];

export const DEFAULT_TRAINING = {
  epochs: 20,
  batchSize: 16,
  learningRate: 0.001,
  validationSplit: 0.15,
};
