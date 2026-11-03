/**
 * Mode B STEP 6 + 7: dual-path inference with confidence gating, and the
 * hybrid "Detect + Classify" mode.
 *
 * Dual path (one MobileNet call, batch of 2):
 *   a) largest face -> 20%-margin crop -> features -> head -> probs_face
 *   b) whole frame  -> letterbox       -> features -> head -> probs_full
 *   c) keep the path whose top probability is higher.
 * A path only runs if the model has classes of its type: probs_face needs a
 * Person class (and a face in view), probs_full needs an Object class. A head
 * never trained on full frames of a person would otherwise produce confident
 * nonsense on that path.
 *
 * Gating: if the winning class is the Unknown class, or its confidence is
 * below the threshold (default 0.60), the answer is "Unknown". Softmax always
 * sums to 1, so without the gate every input would be force-matched.
 *
 * Hybrid: Mode A localizes (COCO-SSD boxes, BlazeFace faces), each region is
 * cropped and classified by the head, and the box gets the custom label.
 */
import * as tf from '@tensorflow/tfjs';
import { HYBRID_MAX_BOXES, NOT_SURE_LABEL } from '../config';
import { cropFace, detectAndCropFace, significantFaces, sourceSize, type Detectable } from './faceDetector';
import { expandBox, letterbox } from './preprocessing';
import { extractFeatures, loadMobilenet, type FeatureExtractor } from './mobilenet';
import { argmax } from './evaluator';
import type {
  ClassKind,
  CocoDetection,
  DualPathPrediction,
  FaceDetection,
  HybridBox,
  InferencePath,
  InferencePaths,
  PathResult,
  Prediction,
} from '../types';

export function inferencePaths(kinds: ClassKind[]): InferencePaths {
  return { face: kinds.includes('person'), full: kinds.includes('object') };
}

/** Class probabilities for a batch of 224x224 canvases. All intermediates are freed by tf.tidy. */
export async function classifyCrops(extractor: FeatureExtractor, head: tf.LayersModel, crops: HTMLCanvasElement[]): Promise<number[][]> {
  if (crops.length === 0) return [];
  const probs = tf.tidy(() => {
    const batch = tf.stack(crops.map((c) => tf.browser.fromPixels(c, 3))) as tf.Tensor4D;
    return head.predict(extractFeatures(extractor, batch)) as tf.Tensor2D;
  });
  try {
    return (await probs.array()) as number[][];
  } finally {
    probs.dispose();
  }
}

/** Applies the Unknown class and the confidence threshold to raw probabilities. */
export function predictWithConfidenceThreshold(
  probs: ArrayLike<number>,
  labels: string[],
  kinds: ClassKind[],
  threshold: number,
): Prediction {
  const topIndex = argmax(probs);
  const confidence = probs[topIndex];
  const allProbs: Record<string, number> = {};
  labels.forEach((l, i) => (allProbs[l] = probs[i]));

  if (kinds[topIndex] === 'unknown') {
    return { label: labels[topIndex], confidence, allProbs, topIndex, isUnknown: true, reason: 'unknown-class' };
  }
  if (confidence < threshold) {
    return { label: NOT_SURE_LABEL, confidence, allProbs, topIndex, isUnknown: true, reason: 'below-threshold' };
  }
  return { label: labels[topIndex], confidence, allProbs, topIndex, isUnknown: false, reason: 'match' };
}

export interface HeadOptions {
  head: tf.LayersModel;
  labels: string[];
  kinds: ClassKind[];
  threshold: number;
}

const fullCanvas = () => document.createElement('canvas');

/**
 * Full dual-path pipeline for one image or frame. Pass `faces` when BlazeFace
 * already ran on these exact pixels (the Detect tab does) to skip a second run.
 */
export async function predictDualPath(
  source: Detectable,
  opts: HeadOptions & { faces?: FaceDetection[] },
): Promise<DualPathPrediction> {
  const paths = inferencePaths(opts.kinds);
  const crops: { path: InferencePath; canvas: HTMLCanvasElement }[] = [];
  const out: DualPathPrediction = { prediction: null, chosen: null, faceCount: 0 };

  if (paths.face) {
    const face = await detectAndCropFace(source, 'largest', opts.faces);
    if (face.status === 'ok') {
      crops.push({ path: 'face', canvas: face.crop });
      out.faceBox = face.box;
      out.faceCount = face.faceCount;
    }
  }
  if (paths.full) crops.push({ path: 'full', canvas: letterbox(source, undefined, fullCanvas()) });
  if (crops.length === 0) return out;

  const extractor = await loadMobilenet();
  const probs = await classifyCrops(extractor, opts.head, crops.map((c) => c.canvas));
  let best: PathResult | null = null;
  crops.forEach(({ path }, i) => {
    const topIndex = argmax(probs[i]);
    const result: PathResult = { path, probs: probs[i], topIndex, confidence: probs[i][topIndex] };
    out[path] = result;
    if (!best || result.confidence > best.confidence) best = result;
  });
  const winner = best as PathResult | null;
  if (!winner) return out;
  out.chosen = winner.path;
  out.prediction = predictWithConfidenceThreshold(winner.probs, opts.labels, opts.kinds, opts.threshold);
  return out;
}

/** Hybrid crop margin: detector boxes are tight, a little context helps the classifier. */
const HYBRID_MARGIN = 0.1;

/**
 * "Detect + Classify": Mode A boxes in, the same boxes out with a Mode B label
 * each. Faces go through the face path, COCO boxes through the letterbox path.
 * Only the HYBRID_MAX_BOXES highest-scoring regions are classified per frame;
 * the rest keep their Mode A label (prediction null).
 */
export async function predictHybrid(
  source: Detectable,
  opts: HeadOptions & { coco: CocoDetection[]; faces: FaceDetection[] },
): Promise<HybridBox[]> {
  const paths = inferencePaths(opts.kinds);
  const { width, height } = sourceSize(source);
  const boxes: (HybridBox & { canvas?: HTMLCanvasElement })[] = [
    ...significantFaces(opts.faces).map((f) => ({ box: f.box, detector: 'face' as const, baseLabel: 'face', baseScore: f.score, prediction: null })),
    ...opts.coco.map((d) => ({ box: d.box, detector: 'coco' as const, baseLabel: d.label, baseScore: d.score, prediction: null })),
  ];

  // Pick which boxes to classify: faces first (if the model knows people), then COCO boxes by score.
  let budget = HYBRID_MAX_BOXES;
  for (const b of boxes) {
    if (budget === 0) break;
    if (b.detector === 'face' && paths.face) b.canvas = cropFace(source, b.box);
    else if (b.detector === 'coco' && paths.full) b.canvas = letterbox(source, expandBox(b.box, HYBRID_MARGIN, width, height));
    if (b.canvas) budget--;
  }
  const jobs = boxes.filter((b) => b.canvas);
  if (jobs.length) {
    const extractor = await loadMobilenet();
    const probs = await classifyCrops(extractor, opts.head, jobs.map((b) => b.canvas!));
    jobs.forEach((b, i) => (b.prediction = predictWithConfidenceThreshold(probs[i], opts.labels, opts.kinds, opts.threshold)));
  }
  return boxes.map(({ box, detector, baseLabel, baseScore, prediction }) => ({ box, detector, baseLabel, baseScore, prediction }));
}
