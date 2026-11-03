// Product limits and defaults. Model constants live in src/ml/*.

/** Custom classes (plus the optional "Unknown" class), one categorical color each. */
export const MAX_CLASSES = 8;

/** Minimum samples before Train unlocks, and what counts as "Good". */
export const MIN_SAMPLES = 20;
export const GOOD_SAMPLES = 40;
export const MIN_SAMPLES_UNKNOWN = 10;
export const GOOD_SAMPLES_UNKNOWN = 20;

/** Hard cap per class so a long webcam hold can't exhaust memory. */
export const MAX_SAMPLES_PER_CLASS = 500;

/** Webcam "Hold to capture" rate (person classes also run the face detector per frame). */
export const CAPTURE_FPS = 10;

/** Mode A: all enabled detectors run at most this often (rAF + timestamp throttle). */
export const DETECT_FPS = 10;

/** Name of the special rejection class. */
export const UNKNOWN_LABEL = 'Unknown';

/** Shown when the top confidence is below the threshold. */
export const NOT_SURE_LABEL = 'Unknown';

export const DEFAULT_THRESHOLD = 0.6;
export const THRESHOLD_MIN = 0.3;
export const THRESHOLD_MAX = 0.9;

/** Stored samples: 224x224 JPEG data URLs (thumbnail and model input in one). */
export const SAMPLE_MIME = 'image/jpeg';
export const SAMPLE_QUALITY = 0.92;

/**
 * Samples pushed through MobileNet per step. Each becomes 6 augmented
 * variants, so one step is a [24, 224, 224, 3] batch.
 */
export const SAMPLES_PER_BATCH = 4;

/** A per-class recall (or precision) below this gets a "needs more data" warning. */
export const WEAK_CLASS_THRESHOLD = 0.7;

export const DEFAULT_TRAINING = {
  epochs: 40,
  batchSize: 16,
  learningRate: 0.0005,
  validationSplit: 0.2,
  patience: 5,
};

/** COCO-SSD: boxes below this score are dropped. */
export const COCO_MIN_SCORE = 0.5;
export const COCO_MAX_BOXES = 20;

/** Hybrid "Detect + Classify": at most this many boxes are re-classified per frame. */
export const HYBRID_MAX_BOXES = 6;

/**
 * Categorical identity colors for custom classes (CVD-safe in this order).
 * "Unknown" is always neutral gray.
 */
export const CLASS_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
export const UNKNOWN_COLOR = '#8a8984';

/** Mode A overlay colors. */
export const FACE_COLOR = '#facc15';
export const HAND_COLOR = '#a855f7';
