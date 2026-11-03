import type * as tf from '@tensorflow/tfjs';

/** One training image, already center-cropped and resized to 224x224. */
export interface Sample {
  id: string;
  /** JPEG data URL, used both as the thumbnail and as the CNN input. */
  dataUrl: string;
  source: 'webcam' | 'upload';
}

/** A user-defined class ("Cat", "Dog", ...) and its samples. */
export interface ImageClass {
  id: string;
  name: string;
  samples: Sample[];
}

export interface TrainingParams {
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationSplit: number;
}

/** Metrics reported at the end of each epoch. */
export interface EpochLog {
  epoch: number;
  loss: number;
  acc: number;
  valLoss?: number;
  valAcc?: number;
}

export type TrainingPhase = 'idle' | 'extracting' | 'training' | 'done' | 'error';

export interface TrainingProgress {
  phase: TrainingPhase;
  /** Feature extraction: images done / total. */
  extracted: number;
  extractTotal: number;
  /** Training: 0-based epoch currently running, and fraction of it done. */
  epoch: number;
  epochFraction: number;
  totalEpochs: number;
  history: EpochLog[];
  error?: string;
}

/**
 * Output of a training run: the small classification head plus the label
 * snapshot it was trained on. The frozen MobileNet base is NOT part of it.
 */
export interface TrainedModel {
  head: tf.LayersModel;
  /** Class ids in output-neuron order, so renames after training still map. */
  classIds: string[];
  /** Class names at training time (fallback if a class is later deleted). */
  labels: string[];
  featureDim: number;
  trainedAt: number;
  sampleCount: number;
  finalMetrics?: EpochLog;
}

/** Contents of metadata.json in every export. */
export interface ModelMetadata {
  labels: string[];
  imageSize: number;
  normalization: '[-1,1]';
  baseModel: 'mobilenet_v2';
  /** Name of the MobileNet layer the features are tapped from. */
  featureLayer: string;
  /** How the featureLayer's 7x7x1280 map is reduced to a vector. */
  pooling: 'global_average';
  alpha: number;
  featureDim: number;
  /** Where to load the base from, relative to the app (informational). */
  baseModelUrl: string;
  createdAt: string;
  generator: string;
  version: number;
}

export interface Prediction {
  className: string;
  probability: number;
}
