import type * as tf from '@tensorflow/tfjs';

// ───────────────────────────── Mode A: pretrained detection ─────────────────────────────

/** Axis-aligned box in source-image pixels. */
export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface CocoDetection {
  box: Box;
  /** One of the 80 COCO class names, e.g. "person", "cell phone". */
  label: string;
  score: number;
}

export interface FaceDetection {
  box: Box;
  score: number;
}

export interface Keypoint {
  x: number;
  y: number;
  name?: string;
}

export interface HandDetection {
  box: Box;
  /** 21 MediaPipe hand landmarks in source pixels (wrist first). */
  keypoints: Keypoint[];
  handedness: 'Left' | 'Right';
  score: number;
}

export type DetectorId = 'coco' | 'face' | 'hands';

/** Everything the enabled detectors found in one frame. */
export interface DetectionFrame {
  width: number;
  height: number;
  coco: CocoDetection[];
  faces: FaceDetection[];
  hands: HandDetection[];
  /** Wall time of the parallel detector run. */
  ms: number;
}

// ───────────────────────────── Mode B: custom classes ─────────────────────────────

/** Person classes are trained on face crops, Object classes on the whole (letterboxed) image. */
export type ClassType = 'person' | 'object';

/** 'unknown' marks the optional rejection class (it holds both views, see preprocessing.ts). */
export type ClassKind = ClassType | 'unknown';

/** How a sample was preprocessed: a 20%-margin face crop, or the full frame letterboxed. */
export type SampleView = 'face' | 'letterbox';

/** One training image: 224x224, already preprocessed, stored as a JPEG data URL. */
export interface Sample {
  id: string;
  dataUrl: string;
  view: SampleView;
  source: 'webcam' | 'upload';
}

export interface CustomClass {
  id: string;
  name: string;
  kind: ClassKind;
  samples: Sample[];
}

export interface TrainingParams {
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationSplit: number;
  /** Early stopping: epochs without a val_loss improvement before stopping. */
  patience: number;
}

export interface EpochLog {
  epoch: number;
  loss: number;
  acc: number;
  valLoss?: number;
  valAcc?: number;
}

export type TrainingPhase = 'idle' | 'extracting' | 'training' | 'evaluating' | 'done' | 'error';

export interface TrainingProgress {
  phase: TrainingPhase;
  /** Feature extraction counts augmented images (samples x 6). */
  extracted: number;
  extractTotal: number;
  samples: number;
  epoch: number;
  epochFraction: number;
  totalEpochs: number;
  history: EpochLog[];
  /** Epoch (1-based) whose weights were kept (lowest val_loss). */
  bestEpoch?: number;
  stoppedEarly?: boolean;
  error?: string;
}

export interface ClassMetrics {
  label: string;
  kind: ClassKind;
  /** Of the validation samples predicted as this class, the fraction that really were. NaN if none predicted. */
  precision: number;
  /** Of this class's validation samples, the fraction recognised correctly. NaN if none held out. */
  recall: number;
  f1: number;
  /** Validation samples of this class. */
  support: number;
  /** All samples of this class (train + validation). */
  samples: number;
}

/** Post-training evaluation on the held-out validation samples (original, un-augmented). */
export interface Evaluation {
  labels: string[];
  overallAccuracy: number;
  perClass: ClassMetrics[];
  /** confusion[actual][predicted] = count. */
  confusion: number[][];
  /** Softmax output per validation sample, for threshold analysis. */
  valProbs: number[][];
  valTrue: number[];
}

/** The trained head plus everything needed to interpret its outputs. MobileNet is not part of it. */
export interface TrainedModel {
  head: tf.LayersModel;
  /** Class ids in output order (so renames after training still map). */
  classIds: string[];
  labels: string[];
  kinds: ClassKind[];
  featureDim: number;
  trainedAt: number;
  sampleCount: number;
  evaluation: Evaluation | null;
  bestEpoch?: number;
  epochsRun?: number;
}

/** Which inference paths a model supports: face crops if it has Person classes, full frames if it has Object classes. */
export interface InferencePaths {
  face: boolean;
  full: boolean;
}

/** metadata.json in the TF.js export. */
export interface ModelMetadata {
  labels: string[];
  /** Per label: "person" (face crop), "object" (letterboxed frame) or "unknown" (rejection class). */
  classTypes: ClassKind[];
  imageSize: number;
  normalization: '[-1,1]';
  baseModel: 'mobilenet_v2';
  confidenceThreshold: number;
  dualPathInference: boolean;
  /** Paths the head was trained for (a path with no classes of its type is skipped at inference). */
  paths: InferencePaths;
  unknownLabel: string | null;
  faceDetector: 'mediapipe_blazeface_short';
  /** Face crop = face box grown by this fraction on every side, squared, resized to imageSize. */
  faceMargin: number;
  /** Letterbox padding value (0 = black) for the full-image path. */
  letterboxFill: number;
  featureDim: number;
  mobilenet: {
    version: 2;
    alpha: 1;
    /** Pixel range fed to the TF-Hub graph; its first op rescales to MobileNet's native [-1, 1]. */
    inputRange: [number, number];
    embeddingLayer: string;
    source: string;
  };
  createdAt: string;
  generator: string;
  version: number;
}

export type InferencePath = 'face' | 'full';

export interface Prediction {
  /** A class label, the Unknown class, or "Unknown" when below the threshold. */
  label: string;
  confidence: number;
  topIndex: number;
  isUnknown: boolean;
  reason: 'match' | 'unknown-class' | 'below-threshold';
  /** label -> probability, in output order. */
  allProbs: Record<string, number>;
}

/** Raw head output for one path. */
export interface PathResult {
  path: InferencePath;
  probs: number[];
  topIndex: number;
  confidence: number;
}

/**
 * STEP 6 output: both paths' raw results and the gated final answer from the
 * more confident one. `prediction` is null only when no path could run (e.g.
 * a person-only model and no face in view).
 */
export interface DualPathPrediction {
  prediction: Prediction | null;
  chosen: InferencePath | null;
  face?: PathResult;
  full?: PathResult;
  /** Face used for the face path (largest), in source pixels. */
  faceBox?: Box;
  faceCount: number;
}

/** One box of the hybrid "Detect + Classify" view: Mode A localizes, Mode B names it. */
export interface HybridBox {
  box: Box;
  detector: 'coco' | 'face';
  /** COCO label (or "face") from Mode A. */
  baseLabel: string;
  baseScore: number;
  prediction: Prediction | null;
}
