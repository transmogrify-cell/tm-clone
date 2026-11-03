/**
 * Export. Only the trained head is saved; MobileNet and BlazeFace are the same
 * for every project and are loaded separately wherever the model is used.
 *
 *  - TensorFlow.js: model.json + weights.bin (the head) + metadata.json
 *    (labels, class types, preprocessing, threshold), as one .zip or as
 *    individual files.
 *  - Keras .h5: the browser can't write HDF5, so a Python snippet reads the
 *    TF.js weights.bin directly and rebuilds the head in tf.keras.
 */
import * as tf from '@tensorflow/tfjs';
import JSZip from 'jszip';
import { EMBEDDING_LAYER, IMAGE_SIZE, MOBILENET_INPUT_RANGE, MOBILENET_SOURCE } from './mobilenet';
import { FACE_MARGIN } from './faceDetector';
import { LETTERBOX_FILL } from './preprocessing';
import { inferencePaths } from './predictor';
import { slugify, triggerDownload } from '../utils/download';
import type { ModelMetadata, TrainedModel } from '../types';

export function buildMetadata(trained: TrainedModel, labels: string[], threshold: number): ModelMetadata {
  const unknownIndex = trained.kinds.indexOf('unknown');
  const paths = inferencePaths(trained.kinds);
  return {
    labels,
    classTypes: trained.kinds,
    imageSize: IMAGE_SIZE,
    normalization: '[-1,1]',
    baseModel: 'mobilenet_v2',
    confidenceThreshold: threshold,
    dualPathInference: paths.face && paths.full,
    paths,
    unknownLabel: unknownIndex >= 0 ? labels[unknownIndex] : null,
    faceDetector: 'mediapipe_blazeface_short',
    faceMargin: FACE_MARGIN,
    letterboxFill: LETTERBOX_FILL,
    featureDim: trained.featureDim,
    mobilenet: {
      version: 2,
      alpha: 1,
      inputRange: MOBILENET_INPUT_RANGE,
      embeddingLayer: EMBEDDING_LAYER,
      source: MOBILENET_SOURCE,
    },
    createdAt: new Date(trained.trainedAt).toISOString(),
    generator: 'smart-vision-trainer',
    version: 1,
  };
}

async function headArtifacts(head: tf.LayersModel): Promise<tf.io.ModelArtifacts> {
  let captured: tf.io.ModelArtifacts | null = null;
  await head.save(
    tf.io.withSaveHandler(async (artifacts) => {
      captured = artifacts;
      return { modelArtifactsInfo: tf.io.getModelArtifactsInfoForJSON(artifacts) };
    }),
  );
  if (!captured) throw new Error('Saving the model produced no data.');
  return captured;
}

export interface TfjsFiles {
  'model.json': Blob;
  'weights.bin': Blob;
  'metadata.json': Blob;
}

/** The three TF.js files, with model.json's manifest pointing at "weights.bin". */
export async function buildTfjsFiles(trained: TrainedModel, labels: string[], threshold: number): Promise<TfjsFiles> {
  const artifacts = await headArtifacts(trained.head);
  const weightData = artifacts.weightData;
  const modelJson: tf.io.ModelJSON = {
    modelTopology: artifacts.modelTopology!,
    format: artifacts.format ?? 'layers-model',
    generatedBy: artifacts.generatedBy,
    convertedBy: null,
    weightsManifest: [{ paths: ['weights.bin'], weights: artifacts.weightSpecs ?? [] }],
  };
  const weights = Array.isArray(weightData) ? tf.io.concatenateArrayBuffers(weightData) : weightData ?? new ArrayBuffer(0);
  const json = (v: unknown) => new Blob([JSON.stringify(v, null, 2)], { type: 'application/json' });
  return {
    'model.json': json(modelJson),
    'weights.bin': new Blob([weights], { type: 'application/octet-stream' }),
    'metadata.json': json(buildMetadata(trained, labels, threshold)),
  };
}

export async function buildTfjsZip(trained: TrainedModel, labels: string[], threshold: number): Promise<Blob> {
  const files = await buildTfjsFiles(trained, labels, threshold);
  const zip = new JSZip();
  for (const [name, blob] of Object.entries(files)) zip.file(name, blob);
  zip.file('convert_to_keras.py', kerasPythonSnippet());
  return zip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
}

export async function exportTfjsZip(trained: TrainedModel, labels: string[], threshold: number, name = 'vision-model'): Promise<void> {
  triggerDownload(await buildTfjsZip(trained, labels, threshold), `${slugify(name)}-tfjs.zip`);
}

export async function exportTfjsFile(trained: TrainedModel, labels: string[], threshold: number, file: keyof TfjsFiles): Promise<void> {
  const files = await buildTfjsFiles(trained, labels, threshold);
  triggerDownload(files[file], file);
}

/**
 * Python that rebuilds the head in tf.keras from the TF.js export and saves .h5.
 * It parses model.json's weight manifest and slices weights.bin with numpy, so
 * it needs nothing but TensorFlow.
 */
export function kerasPythonSnippet(): string {
  return `#!/usr/bin/env python3
"""
Rebuild the custom classification HEAD exported from Smart Vision Trainer as Keras .h5.

Put this file next to model.json, weights.bin and metadata.json (the unzipped
TF.js export) and run:

    pip install "tensorflow>=2.12"
    python convert_to_keras.py            # -> custom_head.h5 + labels.txt

This is only the head. Its input is the 1280-d MobileNet V2 feature vector,
not an image. To classify an image, reproduce the browser pipeline:
  * Person classes ("classTypes" == "person"): detect the face (BlazeFace /
    MediaPipe face detection), crop a square with a 20% margin per side,
    resize to 224x224.
  * Object classes ("object"): letterbox the whole image to 224x224 (keep the
    aspect ratio, pad with black).
  * Features: MobileNet V2 1.0 224 from TF-Hub
    "google/imagenet/mobilenet_v2_100_224/feature_vector/2" (the browser uses the
    matching classification/2 graph up to its 1280-d AvgPool) with pixels in [0, 1]
    (the module maps them to [-1, 1] internally).
  * probs = head.predict(features). With "dualPathInference", run both crops and
    keep the more confident one; below "confidenceThreshold" -> "Unknown".
"""
import json
import os

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
L2 = tf.keras.regularizers.l2(0.001)


def load_tfjs_weights(folder):
    """name -> numpy array, read from the TF.js weights manifest."""
    with open(os.path.join(folder, "model.json"), encoding="utf-8") as f:
        manifest = json.load(f)["weightsManifest"]
    weights = {}
    for group in manifest:
        data = b"".join(open(os.path.join(folder, p), "rb").read() for p in group["paths"])
        offset = 0
        for spec in group["weights"]:
            count = int(np.prod(spec["shape"])) if spec["shape"] else 1
            weights[spec["name"]] = np.frombuffer(data, np.float32, count, offset).reshape(spec["shape"])
            offset += count * 4
    return weights


def build_head(feature_dim, num_classes):
    """Same architecture as the browser head, with the same layer names."""
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(feature_dim,), name="mobilenet_features"),
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=L2, name="head_dense_1"),
            tf.keras.layers.BatchNormalization(name="head_bn"),
            tf.keras.layers.Dropout(0.4, name="head_dropout_1"),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=L2, name="head_dense_2"),
            tf.keras.layers.Dropout(0.3, name="head_dropout_2"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="head_output"),
        ],
        name="custom_head",
    )


def main():
    with open(os.path.join(HERE, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    labels = meta["labels"]
    w = load_tfjs_weights(HERE)
    head = build_head(meta["featureDim"], len(labels))

    head.get_layer("head_dense_1").set_weights([w["head_dense_1/kernel"], w["head_dense_1/bias"]])
    head.get_layer("head_bn").set_weights(
        [w["head_bn/gamma"], w["head_bn/beta"], w["head_bn/moving_mean"], w["head_bn/moving_variance"]]
    )
    head.get_layer("head_dense_2").set_weights([w["head_dense_2/kernel"], w["head_dense_2/bias"]])
    head.get_layer("head_output").set_weights([w["head_output/kernel"], w["head_output/bias"]])

    head.save(os.path.join(HERE, "custom_head.h5"))
    with open(os.path.join(HERE, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\\n".join(labels) + "\\n")
    types = ", ".join(f"{l} ({t})" for l, t in zip(labels, meta["classTypes"]))
    print(f"Saved custom_head.h5 with {len(labels)} classes: {types}")
    print(f"Suggested confidence threshold: {meta['confidenceThreshold']}")


if __name__ == "__main__":
    main()
`;
}

/** Plain TF.js usage of the export in any web page (no app code needed). */
export function jsUsageSnippet(): string {
  return `import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';

// The unzipped export, served next to your page.
const head = await tf.loadLayersModel('/my-model/model.json');
const meta = await (await fetch('/my-model/metadata.json')).json();
const net = await mobilenet.load({ version: 2, alpha: 1.0, inputRange: meta.mobilenet.inputRange });

/** Fit an image into a 224x224 canvas, black padding (the "object" path). */
function letterbox(img) {
  const c = Object.assign(document.createElement('canvas'), { width: 224, height: 224 });
  const ctx = c.getContext('2d');
  const w = img.naturalWidth ?? img.videoWidth ?? img.width, h = img.naturalHeight ?? img.videoHeight ?? img.height;
  const s = 224 / Math.max(w, h);
  ctx.fillRect(0, 0, 224, 224);
  ctx.drawImage(img, (224 - w * s) / 2, (224 - h * s) / 2, w * s, h * s);
  return c;
}

export async function classify(img) {
  // For person classes, crop the face first (BlazeFace, 20% margin) instead of letterboxing.
  const probs = tf.tidy(() => head.predict(net.infer(tf.browser.fromPixels(letterbox(img)), true)).dataSync());
  const top = probs.indexOf(Math.max(...probs));
  return probs[top] < meta.confidenceThreshold ? { label: 'Unknown', confidence: probs[top] }
                                               : { label: meta.labels[top], confidence: probs[top] };
}
`;
}
