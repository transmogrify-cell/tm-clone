/**
 * Inference (spec §3 step D):
 *   image -> crop/resize/normalize -> frozen MobileNet -> features -> head -> softmax
 *
 * Also `loadModel()`, which re-assembles an exported model: it loads MobileNet,
 * loads the exported head (model.json + weights.bin) and metadata.json, and
 * returns a predict(image) function.
 */
import * as tf from '@tensorflow/tfjs';
import JSZip from 'jszip';
import { extractFeatures, loadMobileNet, type FeatureExtractor } from './mobilenet';
import { tensorFromImage, type ImageLike } from '../utils/image';
import type { ModelMetadata, Prediction } from '../types';

/**
 * Class probabilities for one image, in the head's output order.
 * Everything up to the softmax output runs synchronously inside tf.tidy, so no
 * intermediate tensor can leak; only `probs` survives and is disposed below.
 */
export async function predictProbabilities(
  extractor: FeatureExtractor,
  head: tf.LayersModel,
  image: ImageLike,
  opts: { mirror?: boolean } = {},
): Promise<Float32Array> {
  const probs = tf.tidy(() => {
    const input = tensorFromImage(image, opts).expandDims(0) as tf.Tensor4D;
    const features = extractFeatures(extractor, input);
    return head.predict(features) as tf.Tensor2D;
  });
  try {
    return (await probs.data()) as Float32Array;
  } finally {
    probs.dispose();
  }
}

/** Pairs probabilities with labels, sorted from most to least likely. */
export function toPredictions(probs: ArrayLike<number>, labels: string[]): Prediction[] {
  return labels
    .map((className, i) => ({ className, probability: probs[i] ?? 0 }))
    .sort((a, b) => b.probability - a.probability);
}

// ───────────────────────────── loadModel() ─────────────────────────────

export interface LoadedModel {
  labels: string[];
  metadata: ModelMetadata;
  head: tf.LayersModel;
  /** Predictions sorted by probability (highest first). */
  predict(image: ImageLike, opts?: { mirror?: boolean }): Promise<Prediction[]>;
  dispose(): void;
}

/**
 * Where an exported model comes from:
 *  - `{ modelUrl }`: a hosted folder (metadata.json defaults to a sibling of model.json)
 *  - a `.zip` File, exactly as downloaded from the Export dialog
 *  - the individual files (model.json, weights.bin, metadata.json) from a file picker
 */
export type ModelSource = { modelUrl: string; metadataUrl?: string } | File | File[] | FileList;

/**
 * 1. Loads MobileNet  2. Loads the head (model.json + weights.bin)
 * 3. Re-attaches the head to MobileNet's features  4. Returns predict(image).
 */
export async function loadModel(source: ModelSource): Promise<LoadedModel> {
  const extractor = await loadMobileNet();
  const { head, metadata } = await loadHead(source);

  const inputDim = head.inputs[0].shape[1];
  if (inputDim !== extractor.featureDim) {
    head.dispose();
    throw new Error(
      `This model expects ${inputDim}-d features but the loaded MobileNet produces ${extractor.featureDim}-d. ` +
        `It was trained with a different base model.`,
    );
  }
  const outputDim = head.outputs[0].shape[1];
  if (outputDim !== metadata.labels.length) {
    head.dispose();
    throw new Error(`metadata.json lists ${metadata.labels.length} labels but the model has ${outputDim} outputs.`);
  }

  return {
    labels: metadata.labels,
    metadata,
    head,
    predict: async (image, opts) => toPredictions(await predictProbabilities(extractor, head, image, opts), metadata.labels),
    dispose: () => head.dispose(),
  };
}

async function loadHead(source: ModelSource): Promise<{ head: tf.LayersModel; metadata: ModelMetadata }> {
  if (!(source instanceof File) && !Array.isArray(source) && !(source instanceof FileList)) {
    // Hosted folder.
    const metadataUrl = source.metadataUrl ?? new URL('metadata.json', new URL(source.modelUrl, location.href)).href;
    const [head, metadata] = await Promise.all([
      tf.loadLayersModel(source.modelUrl),
      fetch(metadataUrl).then((r) => {
        if (!r.ok) throw new Error(`metadata.json: HTTP ${r.status}`);
        return r.json() as Promise<ModelMetadata>;
      }),
    ]);
    return { head, metadata };
  }

  if (source instanceof File && /\.zip$/i.test(source.name)) {
    const files = await filesFromZip(source);
    return loadHeadFromFiles(files);
  }

  return loadHeadFromFiles(source instanceof File ? [source] : Array.from(source));
}

/** Unpacks an exported zip into File objects (folder prefixes are ignored). */
async function filesFromZip(zipFile: File): Promise<File[]> {
  const zip = await JSZip.loadAsync(zipFile);
  const entries = Object.values(zip.files).filter((f) => !f.dir);
  return Promise.all(
    entries.map(async (entry) => new File([await entry.async('arraybuffer')], entry.name.split('/').pop()!)),
  );
}

async function loadHeadFromFiles(files: File[]) {
  const byName = new Map(files.map((f) => [f.name, f]));
  const modelFile = byName.get('model.json');
  const metadataFile = byName.get('metadata.json');
  if (!modelFile) throw new Error('model.json is missing.');
  if (!metadataFile) throw new Error('metadata.json is missing.');

  const modelJson = JSON.parse(await modelFile.text()) as tf.io.ModelJSON;
  const metadata = JSON.parse(await metadataFile.text()) as ModelMetadata;

  // Concatenate the weight shards in manifest order.
  const buffers: ArrayBuffer[] = [];
  for (const group of modelJson.weightsManifest) {
    for (const path of group.paths) {
      const shard = byName.get(path.split('/').pop()!);
      if (!shard) throw new Error(`Weights file "${path}" is missing.`);
      buffers.push(await shard.arrayBuffer());
    }
  }

  const head = await tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: modelJson.modelTopology,
      weightSpecs: modelJson.weightsManifest.flatMap((g) => g.weights),
      weightData: tf.io.concatenateArrayBuffers(buffers),
      format: modelJson.format,
      generatedBy: modelJson.generatedBy,
      convertedBy: modelJson.convertedBy,
    }),
  );
  return { head, metadata };
}
