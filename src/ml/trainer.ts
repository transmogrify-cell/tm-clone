/**
 * Training pipeline (spec §3, steps B and C):
 *
 *   samples ──► frozen MobileNet ──► 1280-d features (cached) ──► small dense head ──► softmax
 *
 * MobileNet runs once per image, never per epoch: features are computed up
 * front and cached by sample id, so retraining after adding a few samples
 * only processes the new ones. Only the ~165k-parameter head is trained.
 */
import * as tf from '@tensorflow/tfjs';
import { extractFeatures, type FeatureExtractor } from './mobilenet';
import { loadImage, tensorFromImage } from '../utils/image';
import { FEATURE_BATCH_SIZE } from '../config';
import type { EpochLog, ImageClass, TrainedModel, TrainingParams } from '../types';

/**
 * Gives the browser a chance to paint between batches/epochs.
 * tf.nextFrame() is requestAnimationFrame, which never fires in a background
 * tab (training would freeze if the user switched tabs) and is throttled in
 * occluded windows. So: hidden -> a MessageChannel task (not throttled like
 * setTimeout); visible -> the next frame, but never wait more than 50 ms.
 */
export function yieldToUI(): Promise<void> {
  const task = () =>
    new Promise<void>((resolve) => {
      const channel = new MessageChannel();
      channel.port1.onmessage = () => resolve();
      channel.port2.postMessage(null);
    });
  if (typeof document !== 'undefined' && document.hidden) return task();
  return Promise.race([tf.nextFrame(), new Promise<void>((resolve) => setTimeout(resolve, 50))]);
}

// ───────────────────────────── Feature cache ─────────────────────────────

/** sampleId -> feature vector. Samples are immutable, so the id is a safe key. */
const featureCache = new Map<string, Float32Array>();

export function forgetFeatures(sampleIds: string[]): void {
  for (const id of sampleIds) featureCache.delete(id);
}

export function clearFeatureCache(): void {
  featureCache.clear();
}

export interface LabeledFeatures {
  /** One feature vector per sample, in class order. */
  features: Float32Array[];
  /** Class index for each entry of `features`. */
  labels: number[];
}

/**
 * Step B: every sample -> frozen MobileNet -> feature vector.
 *
 * Images go through the network FEATURE_BATCH_SIZE at a time inside tf.tidy()
 * so intermediate activations are freed; the only tensor that escapes the
 * tidy is the [batch, featureDim] output, which is downloaded and disposed.
 */
export async function getFeatures(
  classes: ImageClass[],
  extractor: FeatureExtractor,
  onProgress?: (done: number, total: number) => void,
): Promise<LabeledFeatures> {
  const items = classes.flatMap((c, classIndex) => c.samples.map((sample) => ({ sample, classIndex })));
  const total = items.length;
  const pending = items.filter(({ sample }) => !featureCache.has(sample.id));
  let done = total - pending.length;
  onProgress?.(done, total);

  for (let start = 0; start < pending.length; start += FEATURE_BATCH_SIZE) {
    const batchItems = pending.slice(start, start + FEATURE_BATCH_SIZE);
    // Decode outside tidy (tidy cannot span an await).
    const images = await Promise.all(batchItems.map(({ sample }) => loadImage(sample.dataUrl)));

    const featureTensor = tf.tidy(() => {
      const batch = tf.stack(images.map((img) => tensorFromImage(img))) as tf.Tensor4D;
      return extractFeatures(extractor, batch);
    });
    const flat = (await featureTensor.data()) as Float32Array;
    featureTensor.dispose();

    const dim = extractor.featureDim;
    batchItems.forEach(({ sample }, i) => {
      featureCache.set(sample.id, flat.slice(i * dim, (i + 1) * dim));
    });

    done += batchItems.length;
    onProgress?.(done, total);
    await yieldToUI(); // let React paint the progress bar
  }

  return {
    features: items.map(({ sample }) => featureCache.get(sample.id)!),
    labels: items.map(({ classIndex }) => classIndex),
  };
}

// ───────────────────────────── Model head ─────────────────────────────

/**
 * Step C: the only part that learns. Dense(128, relu) -> Dropout(0.2) ->
 * Dense(numClasses, softmax). Layer names are fixed so exports are predictable.
 */
export function buildHead(featureDim: number, numClasses: number, learningRate: number): tf.Sequential {
  const head = tf.sequential({ name: 'classifier_head' });
  head.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [featureDim], name: 'head_dense' }));
  head.add(tf.layers.dropout({ rate: 0.2, name: 'head_dropout' }));
  head.add(tf.layers.dense({ units: numClasses, activation: 'softmax', name: 'head_output' }));
  head.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return head;
}

/** Fisher-Yates shuffle (in place). */
function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Stratified train/validation split.
 *
 * Why not just `validationSplit` in fit()? tfjs takes the validation set from
 * the *tail* of the arrays before shuffling. Our samples are ordered by class,
 * so the validation set would be (almost) entirely the last class and that
 * class would barely be trained on. Splitting per class keeps the 15% ratio
 * while guaranteeing every class appears in both sets.
 */
export function stratifiedSplit(labels: number[], numClasses: number, valFraction: number) {
  const train: number[] = [];
  const val: number[] = [];
  for (let c = 0; c < numClasses; c++) {
    const idx = shuffle(labels.flatMap((label, i) => (label === c ? [i] : [])));
    const nVal = valFraction > 0 && idx.length >= 2 ? Math.min(idx.length - 1, Math.max(1, Math.round(idx.length * valFraction))) : 0;
    val.push(...idx.slice(0, nVal));
    train.push(...idx.slice(nVal));
  }
  return { train: shuffle(train), val: shuffle(val) };
}

function toTensors(data: LabeledFeatures, indices: number[], numClasses: number, featureDim: number) {
  const x = new Float32Array(indices.length * featureDim);
  indices.forEach((src, row) => x.set(data.features[src], row * featureDim));
  return tf.tidy(() => ({
    xs: tf.tensor2d(x, [indices.length, featureDim]),
    ys: tf.cast(tf.oneHot(tf.tensor1d(indices.map((i) => data.labels[i]), 'int32'), numClasses), 'float32') as tf.Tensor2D,
  }));
}

export interface TrainCallbacks {
  onExtractProgress?: (done: number, total: number) => void;
  onTrainStart?: () => void;
  onBatchEnd?: (epoch: number, fraction: number) => void;
  onEpochEnd?: (log: EpochLog) => void;
}

/**
 * Full run: extract (cached) features, build a fresh head, fit it.
 * Returns the trained head; the caller owns it and must dispose the previous one.
 */
export async function trainClassifier(
  classes: ImageClass[],
  params: TrainingParams,
  extractor: FeatureExtractor,
  callbacks: TrainCallbacks = {},
): Promise<TrainedModel> {
  const numClasses = classes.length;
  const data = await getFeatures(classes, extractor, callbacks.onExtractProgress);
  const { train, val } = stratifiedSplit(data.labels, numClasses, params.validationSplit);

  const head = buildHead(extractor.featureDim, numClasses, params.learningRate);
  const trainSet = toTensors(data, train, numClasses, extractor.featureDim);
  const valSet = val.length > 0 ? toTensors(data, val, numClasses, extractor.featureDim) : null;
  const batchesPerEpoch = Math.ceil(train.length / params.batchSize);
  const history: EpochLog[] = [];
  let currentEpoch = 0;
  let lastYield = performance.now();

  callbacks.onTrainStart?.();
  try {
    await head.fit(trainSet.xs, trainSet.ys, {
      epochs: params.epochs,
      batchSize: params.batchSize,
      shuffle: true,
      validationData: valSet ? [valSet.xs, valSet.ys] : undefined,
      // tfjs' own 'auto' yielding is also rAF-based; we yield ourselves below.
      yieldEvery: 'never',
      callbacks: {
        onEpochBegin: async (epoch) => {
          currentEpoch = epoch;
        },
        onBatchEnd: async (batch) => {
          callbacks.onBatchEnd?.(currentEpoch, (batch + 1) / batchesPerEpoch);
          // Big datasets mean hundreds of batches per epoch: repaint every ~50 ms.
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
          callbacks.onEpochEnd?.(log);
          // Yield so the UI (progress bar, metrics) can repaint between epochs.
          await yieldToUI();
          lastYield = performance.now();
        },
      },
    });
  } catch (err) {
    head.dispose();
    throw err;
  } finally {
    tf.dispose([trainSet.xs, trainSet.ys]);
    if (valSet) tf.dispose([valSet.xs, valSet.ys]);
  }

  return {
    head,
    classIds: classes.map((c) => c.id),
    labels: classes.map((c) => c.name),
    featureDim: extractor.featureDim,
    trainedAt: Date.now(),
    sampleCount: data.features.length,
    finalMetrics: history[history.length - 1],
  };
}
