/**
 * Mode B STEP 4: training the classification head (transfer learning).
 *
 *   224x224 sample ──► 6 augmented variants ──► frozen MobileNet ──► 1280-d features (cached)
 *                                                                        │
 *     Dense 256 relu (L2 1e-3) ─ BatchNorm ─ Dropout .4 ─ Dense 128 relu (L2 1e-3) ─ Dropout .3 ─ Dense N softmax
 *
 *   adam(0.0005) · categoricalCrossentropy · 40 epochs · batch 16 · 20% validation
 *   EarlyStopping(val_loss, patience 5, restoreBestWeights)
 *
 * MobileNet runs once per variant, never per epoch: features are cached by
 * sample id, so retraining after adding samples only processes the new ones.
 *
 * Validation is split by SAMPLE before augmentation, stratified per class. If
 * variants were split instead, a flipped copy of a validation image would sit
 * in the training set and the accuracy report would be inflated. Validation
 * uses the original (un-augmented) samples only, which is what the model sees
 * in real use.
 */
import * as tf from '@tensorflow/tfjs';
import { extractFeatures, type FeatureExtractor } from './mobilenet';
import { generateAugmentedVariants, seedFromId, VARIANTS_PER_IMAGE } from './augmentation';
import { fillFor } from './preprocessing';
import { loadImage } from '../utils/image';
import { SAMPLES_PER_BATCH } from '../config';
import type { CustomClass, EpochLog, TrainingParams } from '../types';

/**
 * Gives the browser a chance to paint. tf.nextFrame() (requestAnimationFrame)
 * never fires in a background tab, so when hidden we yield with a
 * MessageChannel task instead; when visible we never wait more than 50 ms.
 */
export function yieldToUI(): Promise<void> {
  if (typeof document !== 'undefined' && document.hidden) {
    return new Promise((resolve) => {
      const channel = new MessageChannel();
      channel.port1.onmessage = () => resolve();
      channel.port2.postMessage(null);
    });
  }
  return Promise.race([tf.nextFrame(), new Promise<void>((resolve) => setTimeout(resolve, 50))]);
}

// ───────────────────────────── Feature cache ─────────────────────────────

/** sampleId -> features of its 6 variants, flattened [6 * featureDim]. */
const featureCache = new Map<string, Float32Array>();

export function forgetFeatures(sampleIds: string[]): void {
  for (const id of sampleIds) featureCache.delete(id);
}

export function clearFeatureCache(): void {
  featureCache.clear();
}

interface SampleFeatures {
  /** [VARIANTS_PER_IMAGE * featureDim]; variant 0 is the original sample. */
  variants: Float32Array;
  label: number;
}

/**
 * Augment + extract features for every sample, SAMPLES_PER_BATCH samples
 * (= 24 images) per MobileNet call, inside tf.tidy so activations are freed.
 * Progress is reported in augmented images ("45 / 240").
 */
export async function getFeatures(
  classes: CustomClass[],
  extractor: FeatureExtractor,
  onProgress?: (done: number, total: number) => void,
): Promise<SampleFeatures[]> {
  const items = classes.flatMap((c, label) => c.samples.map((sample) => ({ sample, label })));
  const total = items.length * VARIANTS_PER_IMAGE;
  const pending = items.filter(({ sample }) => !featureCache.has(sample.id));
  let done = (items.length - pending.length) * VARIANTS_PER_IMAGE;
  onProgress?.(done, total);

  const dim = extractor.featureDim;
  for (let start = 0; start < pending.length; start += SAMPLES_PER_BATCH) {
    const batch = pending.slice(start, start + SAMPLES_PER_BATCH);
    const images = await Promise.all(batch.map(({ sample }) => loadImage(sample.dataUrl)));

    const features = tf.tidy(() => {
      const variants = images.map((img, i) =>
        generateAugmentedVariants(tf.browser.fromPixels(img, 3), seedFromId(batch[i].sample.id), fillFor(batch[i].sample.view)),
      );
      return extractFeatures(extractor, tf.concat(variants, 0) as tf.Tensor4D);
    });
    const flat = (await features.data()) as Float32Array;
    features.dispose();

    const per = VARIANTS_PER_IMAGE * dim;
    batch.forEach(({ sample }, i) => featureCache.set(sample.id, flat.slice(i * per, (i + 1) * per)));

    done += batch.length * VARIANTS_PER_IMAGE;
    onProgress?.(done, total);
    await yieldToUI();
  }

  return items.map(({ sample, label }) => ({ variants: featureCache.get(sample.id)!, label }));
}

// ───────────────────────────── Head ─────────────────────────────

/** Small, regularized head: L2 on the dense kernels, batch norm, two dropouts. */
export function buildHead(featureDim: number, numClasses: number, learningRate: number): tf.Sequential {
  const head = tf.sequential({ name: 'custom_head' });
  head.add(
    tf.layers.dense({
      units: 256,
      activation: 'relu',
      inputShape: [featureDim],
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
      name: 'head_dense_1',
    }),
  );
  head.add(tf.layers.batchNormalization({ name: 'head_bn' }));
  head.add(tf.layers.dropout({ rate: 0.4, name: 'head_dropout_1' }));
  head.add(
    tf.layers.dense({
      units: 128,
      activation: 'relu',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
      name: 'head_dense_2',
    }),
  );
  head.add(tf.layers.dropout({ rate: 0.3, name: 'head_dropout_2' }));
  head.add(tf.layers.dense({ units: numClasses, activation: 'softmax', name: 'head_output' }));
  head.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return head;
}

function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Stratified split by sample: each class keeps ~valFraction of its samples for
 * validation (at least 1 when it has 2+), so every class is evaluated.
 */
export function stratifiedSplit(labels: number[], numClasses: number, valFraction: number) {
  const train: number[] = [];
  const val: number[] = [];
  for (let c = 0; c < numClasses; c++) {
    const idx = shuffle(labels.flatMap((label, i) => (label === c ? [i] : [])));
    const nVal =
      valFraction > 0 && idx.length >= 2 ? Math.min(idx.length - 1, Math.max(1, Math.round(idx.length * valFraction))) : 0;
    val.push(...idx.slice(0, nVal));
    train.push(...idx.slice(nVal));
  }
  return { train, val };
}

/**
 * Balanced class weights: a class with half as many training images counts
 * twice as much per image, so a smaller "Unknown" class isn't drowned out.
 */
function classWeights(labels: number[], numClasses: number): Record<number, number> {
  const counts = new Array(numClasses).fill(0);
  labels.forEach((l) => counts[l]++);
  const weights: Record<number, number> = {};
  counts.forEach((n, c) => (weights[c] = n > 0 ? labels.length / (numClasses * n) : 1));
  return weights;
}

export interface TrainCallbacks {
  onExtractProgress?: (done: number, total: number) => void;
  onTrainStart?: () => void;
  onBatchEnd?: (epoch: number, fraction: number) => void;
  onEpochEnd?: (log: EpochLog) => void;
  /** Return true to stop after the current batch (the best weights are still restored). */
  shouldStop?: () => boolean;
}

export interface TrainResult {
  head: tf.Sequential;
  history: EpochLog[];
  bestEpoch: number;
  epochsRun: number;
  stoppedEarly: boolean;
  /** Original-sample features of the validation set (for the accuracy report). */
  valFeatures: Float32Array[];
  valLabels: number[];
  sampleCount: number;
}

export async function trainHead(
  classes: CustomClass[],
  params: TrainingParams,
  extractor: FeatureExtractor,
  callbacks: TrainCallbacks = {},
): Promise<TrainResult> {
  const numClasses = classes.length;
  const dim = extractor.featureDim;
  const samples = await getFeatures(classes, extractor, callbacks.onExtractProgress);
  const { train, val } = stratifiedSplit(
    samples.map((p) => p.label),
    numClasses,
    params.validationSplit,
  );

  // Training set: all 6 variants of each training sample.
  const xTrain = new Float32Array(train.length * VARIANTS_PER_IMAGE * dim);
  const yTrain: number[] = [];
  train.forEach((p, row) => {
    xTrain.set(samples[p].variants, row * VARIANTS_PER_IMAGE * dim);
    for (let v = 0; v < VARIANTS_PER_IMAGE; v++) yTrain.push(samples[p].label);
  });
  // Validation set: only the original (variant 0) of each held-out sample.
  const valFeatures = val.map((p) => samples[p].variants.slice(0, dim));
  const valLabels = val.map((p) => samples[p].label);

  const oneHot = (labels: number[]) =>
    tf.tidy(() => tf.cast(tf.oneHot(tf.tensor1d(labels, 'int32'), numClasses), 'float32') as tf.Tensor2D);
  const xVal = new Float32Array(val.length * dim);
  valFeatures.forEach((f, i) => xVal.set(f, i * dim));
  const tensors = {
    xs: tf.tensor2d(xTrain, [yTrain.length, dim]),
    ys: oneHot(yTrain),
    xv: val.length ? tf.tensor2d(xVal, [val.length, dim]) : null,
    yv: val.length ? oneHot(valLabels) : null,
  };

  const head = buildHead(dim, numClasses, params.learningRate);
  const history: EpochLog[] = [];
  const batchesPerEpoch = Math.ceil(yTrain.length / params.batchSize);
  let currentEpoch = 0;
  let lastYield = performance.now();

  // EarlyStopping(val_loss, patience, restoreBestWeights). tf.callbacks.earlyStopping
  // can stop, but tfjs doesn't implement restoreBestWeights, so we keep our own copy.
  let bestLoss = Infinity;
  let bestEpoch = 0;
  let bestWeights: tf.Tensor[] | null = null;
  let sinceBest = 0;
  let stoppedEarly = false;

  callbacks.onTrainStart?.();
  try {
    await head.fit(tensors.xs, tensors.ys, {
      epochs: params.epochs,
      batchSize: params.batchSize,
      shuffle: true,
      validationData: tensors.xv && tensors.yv ? [tensors.xv, tensors.yv] : undefined,
      classWeight: classWeights(yTrain, numClasses),
      yieldEvery: 'never', // tfjs' auto-yield is rAF-based; we yield ourselves
      callbacks: {
        onEpochBegin: async (epoch) => {
          currentEpoch = epoch;
        },
        onBatchEnd: async (batch) => {
          callbacks.onBatchEnd?.(currentEpoch, (batch + 1) / batchesPerEpoch);
          if (callbacks.shouldStop?.()) head.stopTraining = true;
          if (performance.now() - lastYield > 50) {
            await yieldToUI();
            lastYield = performance.now();
          }
        },
        onEpochEnd: async (epoch, logs = {}) => {
          const log: EpochLog = {
            epoch: epoch + 1,
            loss: logs.loss,
            acc: logs.acc ?? logs.accuracy,
            valLoss: logs.val_loss,
            valAcc: logs.val_acc ?? logs.val_accuracy,
          };
          history.push(log);

          const monitored = log.valLoss ?? log.loss;
          if (monitored < bestLoss - 1e-4) {
            bestLoss = monitored;
            bestEpoch = log.epoch;
            sinceBest = 0;
            bestWeights?.forEach((w) => w.dispose());
            bestWeights = head.getWeights().map((w) => w.clone());
          } else if (++sinceBest >= params.patience) {
            stoppedEarly = true;
            head.stopTraining = true;
          }
          if (callbacks.shouldStop?.()) head.stopTraining = true;

          callbacks.onEpochEnd?.(log);
          await yieldToUI();
          lastYield = performance.now();
        },
      },
    });
    // restoreBestWeights: keep the weights from the epoch with the lowest validation loss.
    if (bestWeights) head.setWeights(bestWeights);
  } catch (err) {
    head.dispose();
    throw err;
  } finally {
    (bestWeights as tf.Tensor[] | null)?.forEach((w) => w.dispose());
    tf.dispose([tensors.xs, tensors.ys, ...(tensors.xv ? [tensors.xv] : []), ...(tensors.yv ? [tensors.yv] : [])]);
  }

  return {
    head,
    history,
    bestEpoch,
    epochsRun: history.length,
    stoppedEarly,
    valFeatures,
    valLabels,
    sampleCount: samples.length,
  };
}

/** Softmax outputs for a set of feature vectors (used by the evaluator). */
export async function predictFeatures(head: tf.LayersModel, features: Float32Array[], dim: number): Promise<number[][]> {
  if (features.length === 0) return [];
  const flat = new Float32Array(features.length * dim);
  features.forEach((f, i) => flat.set(f, i * dim));
  const probs = tf.tidy(() => head.predict(tf.tensor2d(flat, [features.length, dim])) as tf.Tensor2D);
  try {
    return (await probs.array()) as number[][];
  } finally {
    probs.dispose();
  }
}
