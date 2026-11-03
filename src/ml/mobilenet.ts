/**
 * Frozen MobileNet V2 feature extractor.
 *
 * Transfer learning in one paragraph: MobileNet was trained on 1.2M ImageNet
 * photos to tell 1000 categories apart. Its last layers have learned a compact,
 * general-purpose description of "what is in this picture". We chop off the
 * 1000-way classifier, keep everything up to the penultimate representation,
 * and use the resulting 1280-number vector as the input to a tiny classifier
 * that we *do* train. MobileNet's own weights are never updated.
 *
 * The base model is served from /public/models/mobilenet (no network at runtime).
 */
import * as tf from '@tensorflow/tfjs';

/** MobileNet input resolution (height = width). */
export const IMAGE_SIZE = 224;
/** Width multiplier of the bundled MobileNet V2 (0.35 = small and fast; same as Teachable Machine). */
export const MOBILENET_ALPHA = 0.35;
/**
 * Layer we tap: the ReLU after the final 1x1 conv (7x7x1280). Global average
 * pooling turns it into a 1280-d vector, which is exactly the "penultimate
 * layer" of the full classifier (the one feeding its 1000-way softmax).
 */
export const FEATURE_LAYER = 'out_relu';
export const MOBILENET_URL = `${import.meta.env.BASE_URL}models/mobilenet/model.json`;

export interface FeatureExtractor {
  /** [n, 224, 224, 3] in [-1, 1]  ->  [n, featureDim]. */
  model: tf.LayersModel;
  featureDim: number;
  featureLayer: string;
}

let backendPromise: Promise<string> | null = null;

/** Prefer WebGL (GPU); fall back to the pure-JS CPU backend if it is unavailable. */
export function initBackend(): Promise<string> {
  backendPromise ??= (async () => {
    try {
      if (await tf.setBackend('webgl')) {
        await tf.ready();
        // Run one op: some drivers "init" fine but fail on first use.
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

let extractorPromise: Promise<FeatureExtractor> | null = null;

/**
 * Loads MobileNet once and returns the truncated, frozen feature extractor.
 * Subsequent calls return the same promise.
 */
export function loadMobileNet(onProgress?: (fraction: number) => void): Promise<FeatureExtractor> {
  extractorPromise ??= buildExtractor(onProgress).catch((err) => {
    extractorPromise = null; // allow a retry after a failure
    throw err;
  });
  return extractorPromise;
}

async function buildExtractor(onProgress?: (fraction: number) => void): Promise<FeatureExtractor> {
  await initBackend();

  let base: tf.LayersModel;
  try {
    base = await tf.loadLayersModel(MOBILENET_URL, { onProgress });
  } catch (err) {
    throw new Error(
      `Could not load MobileNet from ${MOBILENET_URL}. Run "npm run fetch-model" to download it. ` +
        `(${err instanceof Error ? err.message : String(err)})`,
    );
  }

  // Tap the penultimate representation and global-average-pool it to a vector.
  const tapped = base.getLayer(FEATURE_LAYER).output as tf.SymbolicTensor;
  const pooled = tf.layers
    .globalAveragePooling2d({ name: 'feature_pool' })
    .apply(tapped) as tf.SymbolicTensor;
  const model = tf.model({ inputs: base.inputs, outputs: pooled, name: 'mobilenet_v2_features' });

  // Freeze: we only ever call predict() on this model, but marking it
  // non-trainable makes the intent explicit and guards against accidental fit().
  model.trainable = false;

  // Warm-up pass compiles the WebGL shaders now, so the first real batch is fast.
  tf.tidy(() => {
    model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]));
  });

  const featureDim = pooled.shape[pooled.shape.length - 1] as number;
  return { model, featureDim, featureLayer: FEATURE_LAYER };
}

/**
 * Runs a normalized image batch through the frozen base.
 * `batch` must be [n, 224, 224, 3] with values in [-1, 1].
 * Returns [n, featureDim]; the caller owns both tensors.
 */
export function extractFeatures(extractor: FeatureExtractor, batch: tf.Tensor4D): tf.Tensor2D {
  return extractor.model.predict(batch) as tf.Tensor2D;
}
