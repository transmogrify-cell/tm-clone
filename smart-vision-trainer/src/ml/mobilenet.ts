/**
 * Shared TF.js backend + Mode B STEP 2: the frozen MobileNet V2 feature extractor.
 *
 * MobileNet V2 (width 1.0, 224x224) was trained on ImageNet. We drop its
 * 1000-way classifier and read the layer just before it: the 1280-d global
 * average pool ("penultimate layer"). Those 1280 numbers describe a face crop
 * or an object photo well enough for a tiny head to tell your classes apart.
 *
 * Frozen by construction: it is a tfjs GraphModel (weights are graph
 * constants), and we only ever run inference on it.
 *
 * Normalization to [-1, 1]: this TF-Hub graph takes pixels in [0, 1] and its
 * first two ops (hub_input/Mul, hub_input/Sub) compute x * 2 - 1, i.e.
 * MobileNet's native [-1, 1] range. The package scales raw [0, 255] pixels
 * to the graph's [0, 1] input when we pass inputRange: [0, 1].
 */
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';

export const IMAGE_SIZE = 224;
export const MODELS_BASE = `${import.meta.env.BASE_URL}models`;
export const MOBILENET_URL = `${MODELS_BASE}/mobilenet/model.json`;
export const MOBILENET_INPUT_RANGE: [number, number] = [0, 1];
export const EMBEDDING_LAYER = 'module_apply_default/MobilenetV2/Logits/AvgPool';
export const MOBILENET_SOURCE = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2';

export interface FeatureExtractor {
  model: mobilenet.MobileNet;
  featureDim: number;
}

let backendPromise: Promise<string> | null = null;

/** Prefer WebGL (GPU); fall back to the CPU backend if it fails. Shared by all five models. */
export function initBackend(): Promise<string> {
  backendPromise ??= (async () => {
    try {
      // Pass tensor shapes as uniforms so shaders compile once per op instead of
      // once per input size: uploaded photos all differ in size, and without this
      // each new size costs ~1 s of recompilation on the first detection.
      tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true);
      if (await tf.setBackend('webgl')) {
        await tf.ready();
        tf.tidy(() => tf.add(tf.scalar(1), tf.scalar(1)).dataSync());
        return 'webgl';
      }
    } catch (err) {
      console.warn('[tfjs] WebGL backend failed, falling back to CPU', err);
    }
    await tf.setBackend('cpu');
    await tf.ready();
    return 'cpu';
  })();
  return backendPromise;
}

/** Wraps a model-load failure with the one command that fixes the usual cause. */
export function modelLoadError(what: string, url: string, err: unknown): Error {
  return new Error(`Could not load ${what} from ${url}. Run "npm run fetch-models". (${err instanceof Error ? err.message : err})`);
}

let extractorPromise: Promise<FeatureExtractor> | null = null;

/** Loads MobileNet once (from public/models/mobilenet) and returns the feature extractor. */
export function loadMobilenet(): Promise<FeatureExtractor> {
  extractorPromise ??= (async () => {
    await initBackend();
    let model: mobilenet.MobileNet;
    try {
      model = await mobilenet.load({ version: 2, alpha: 1.0, modelUrl: MOBILENET_URL, inputRange: MOBILENET_INPUT_RANGE });
    } catch (err) {
      throw modelLoadError('MobileNet', MOBILENET_URL, err);
    }
    // Probe the embedding size (and compile the shaders for that path).
    const probe = model.infer(tf.zeros([IMAGE_SIZE, IMAGE_SIZE, 3]), true);
    const featureDim = probe.shape[probe.shape.length - 1] as number;
    probe.dispose();
    return { model, featureDim };
  })().catch((err) => {
    extractorPromise = null;
    throw err;
  });
  return extractorPromise;
}

/**
 * Pixels -> features. `pixels` is [n, 224, 224, 3] (or a single [224, 224, 3])
 * with raw values in [0, 255]. Returns [n, 1280]. The caller owns both tensors.
 */
export function extractFeatures(extractor: FeatureExtractor, pixels: tf.Tensor3D | tf.Tensor4D): tf.Tensor2D {
  return extractor.model.infer(pixels, true) as tf.Tensor2D;
}
