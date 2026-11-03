/**
 * Model export (spec §6 and §7).
 *
 * Only the trained head is exported. MobileNet is not: it is identical for
 * every project, so consumers reload it and stitch the head back on top
 * (see loadModel() in predictor.ts, or the Python script below).
 *
 *  - TensorFlow.js : model.json + weights.bin (the head) + metadata.json, zipped.
 *  - Keras (.h5)   : the browser cannot write HDF5, so we export the head weights
 *                    as JSON plus a Python script that rebuilds the full model
 *                    (MobileNet V2 + head) with tf.keras and saves keras_model.h5.
 *  - TF Lite       : there is no in-browser tfjs -> TFLite converter
 *                    (tfjs-converter only goes the other way), so the same
 *                    script builds the Keras model and runs TFLiteConverter.
 */
import * as tf from '@tensorflow/tfjs';
import JSZip from 'jszip';
import { FEATURE_LAYER, IMAGE_SIZE, MOBILENET_ALPHA, MOBILENET_URL } from './mobilenet';
import { slugify, triggerDownload } from '../utils/download';
import type { ModelMetadata, TrainedModel } from '../types';

export type PythonTarget = 'keras' | 'tflite';
export type TfliteQuantization = 'none' | 'float16' | 'dynamic';

export const HEAD_WEIGHTS_FORMAT = 'tm-clone-head/v1';

export function buildMetadata(trained: TrainedModel, labels: string[]): ModelMetadata {
  return {
    labels,
    imageSize: IMAGE_SIZE,
    normalization: '[-1,1]',
    baseModel: 'mobilenet_v2',
    featureLayer: FEATURE_LAYER,
    pooling: 'global_average',
    alpha: MOBILENET_ALPHA,
    featureDim: trained.featureDim,
    baseModelUrl: MOBILENET_URL,
    createdAt: new Date(trained.trainedAt).toISOString(),
    generator: 'tm-clone',
    version: 1,
  };
}

function json(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

// ───────────────────────────── TensorFlow.js ─────────────────────────────

/** Serializes the head with tfjs' own save() machinery, captured in memory. */
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

function toArrayBuffer(data: tf.io.WeightData | undefined): ArrayBuffer {
  if (!data) return new ArrayBuffer(0);
  return Array.isArray(data) ? tf.io.concatenateArrayBuffers(data) : data;
}

/** model.json + weights.bin + metadata.json, as one zip. */
export async function buildTfjsZip(trained: TrainedModel, labels: string[]): Promise<Blob> {
  const artifacts = await headArtifacts(trained.head);
  const modelJson: tf.io.ModelJSON = {
    modelTopology: artifacts.modelTopology!,
    format: artifacts.format ?? 'layers-model',
    generatedBy: artifacts.generatedBy,
    convertedBy: null,
    trainingConfig: artifacts.trainingConfig,
    weightsManifest: [{ paths: ['weights.bin'], weights: artifacts.weightSpecs ?? [] }],
  };

  const zip = new JSZip();
  zip.file('model.json', json(modelJson));
  zip.file('weights.bin', toArrayBuffer(artifacts.weightData));
  zip.file('metadata.json', json(buildMetadata(trained, labels)));
  return zip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
}

export async function exportTfjs(trained: TrainedModel, labels: string[], projectName = 'model'): Promise<void> {
  triggerDownload(await buildTfjsZip(trained, labels), `${slugify(projectName)}-tfjs.zip`);
}

// ───────────────────────────── Keras / TF Lite kit ─────────────────────────────

/** 9 significant digits round-trip a float32 exactly, at half the JSON size of full doubles. */
function roundFloats(values: Float32Array): number[] {
  return Array.from(values, (v) => Number(v.toPrecision(9)));
}

/**
 * head_weights.json: the head architecture (in order) and its weights.
 * Kernels are flattened row-major with their shape, [inputDim, units], which is
 * the same layout Keras uses, so `np.reshape(kernel, kernelShape)` restores them.
 */
export async function buildHeadWeightsJson(trained: TrainedModel, labels: string[]): Promise<string> {
  const layers = [];
  for (const layer of trained.head.layers) {
    const cls = layer.getClassName();
    const config = layer.getConfig();
    if (cls === 'Dense') {
      // getWeights() returns the live variables: read them, don't dispose them.
      const [kernel, bias] = layer.getWeights();
      layers.push({
        type: 'dense',
        name: layer.name,
        units: config.units,
        activation: config.activation,
        kernelShape: kernel.shape,
        kernel: roundFloats((await kernel.data()) as Float32Array),
        bias: roundFloats((await bias.data()) as Float32Array),
      });
    } else if (cls === 'Dropout') {
      layers.push({ type: 'dropout', name: layer.name, rate: config.rate });
    } else {
      throw new Error(`Unsupported head layer: ${cls}`);
    }
  }
  const { labels: _labels, ...meta } = buildMetadata(trained, labels);
  return JSON.stringify({ format: HEAD_WEIGHTS_FORMAT, labels, ...meta, layers });
}

export function pythonFilename(target: PythonTarget): string {
  return target === 'keras' ? 'convert_to_keras.py' : 'convert_to_tflite.py';
}

/** The script shown (and bundled) in the Keras and TF Lite export tabs. */
export function pythonScript(target: PythonTarget, quantize: TfliteQuantization = 'none'): string {
  const name = pythonFilename(target);
  return `#!/usr/bin/env python3
"""
Rebuild the image classifier you trained in the browser as a full Keras model
and/or convert it to TensorFlow Lite. Runs 100% locally.

The browser only trained a small "head" on top of a frozen MobileNet V2. This
script recreates that same MobileNet V2 (alpha=${MOBILENET_ALPHA}, ImageNet weights,
${IMAGE_SIZE}x${IMAGE_SIZE}, global-average-pooled "${FEATURE_LAYER}" features), stacks the head
on top and loads the weights from head_weights.json. The result is one model:

    input : float32 [batch, ${IMAGE_SIZE}, ${IMAGE_SIZE}, 3], RGB scaled to [-1, 1] (pixel / 127.5 - 1)
    output: float32 [batch, num_classes] softmax probabilities, in labels.txt order

Setup : pip install "tensorflow>=2.12" pillow
Run   : python ${name}                       (writes ${target === 'keras' ? 'keras_model.h5' : 'model.tflite'} + labels.txt)
        python ${name} --format both         (keras_model.h5 and model.tflite)
        python ${name} --image photo.jpg     (also classify one image as a check)
        python ${name} --quantize float16    (smaller .tflite: none | float16 | dynamic)
"""
import argparse
import json
import os
import tempfile

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FORMAT = "${target}"
DEFAULT_QUANTIZE = "${quantize}"


def load_spec(path):
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    if spec.get("format") != "${HEAD_WEIGHTS_FORMAT}":
        raise SystemExit(f"{path} is not a head_weights.json exported by the app")
    return spec


def build_model(spec):
    size = spec["imageSize"]
    # The same frozen feature extractor the browser used. Keras downloads these
    # ImageNet weights once (~1.6 MB) and caches them in ~/.keras/models.
    base = tf.keras.applications.MobileNetV2(
        input_shape=(size, size, 3),
        alpha=spec["alpha"],
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(size, size, 3), name="image")
    x = base(inputs, training=False)
    for layer in spec["layers"]:
        if layer["type"] == "dense":
            x = tf.keras.layers.Dense(layer["units"], activation=layer["activation"], name=layer["name"])(x)
        elif layer["type"] == "dropout":
            x = tf.keras.layers.Dropout(layer["rate"], name=layer["name"])(x)
        else:
            raise SystemExit(f"Unsupported layer type {layer['type']!r}")
    model = tf.keras.Model(inputs, x, name="image_classifier")

    # Load the head weights trained in the browser. Kernels are [input_dim, units].
    for layer in spec["layers"]:
        if layer["type"] == "dense":
            kernel = np.asarray(layer["kernel"], dtype=np.float32).reshape(layer["kernelShape"])
            bias = np.asarray(layer["bias"], dtype=np.float32)
            model.get_layer(layer["name"]).set_weights([kernel, bias])
    return model


def _configure(converter, quantize):
    if quantize in ("float16", "dynamic"):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if quantize == "float16":
        converter.target_spec.supported_types = [tf.float16]
    return converter


def to_tflite(model, quantize):
    try:
        return _configure(tf.lite.TFLiteConverter.from_keras_model(model), quantize).convert()
    except Exception as err:  # some TF/Keras combinations need the SavedModel route
        print(f"from_keras_model failed ({err}); converting via SavedModel instead")
        saved = tempfile.mkdtemp()
        if hasattr(model, "export"):
            model.export(saved)  # Keras 3
        else:
            tf.saved_model.save(model, saved)
        return _configure(tf.lite.TFLiteConverter.from_saved_model(saved), quantize).convert()


def load_image(path, size):
    """Center-crop to a square, resize, scale to [-1, 1]: the same as the browser."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((size, size), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 127.5 - 1.0)[np.newaxis]


def run_tflite(tflite_bytes, x):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], x.astype(inp["dtype"]))
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=os.path.join(HERE, "head_weights.json"))
    parser.add_argument("--format", choices=["keras", "tflite", "both"], default=DEFAULT_FORMAT)
    parser.add_argument("--quantize", choices=["none", "float16", "dynamic"], default=DEFAULT_QUANTIZE)
    parser.add_argument("--out", default=HERE, help="output folder")
    parser.add_argument("--image", help="optional test image to classify")
    args = parser.parse_args()

    spec = load_spec(args.weights)
    labels = spec["labels"]
    model = build_model(spec)
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\\n".join(labels) + "\\n")

    if args.format in ("keras", "both"):
        path = os.path.join(args.out, "keras_model.h5")
        model.save(path)
        print(f"Saved {path}")

    tflite_bytes = None
    if args.format in ("tflite", "both"):
        tflite_bytes = to_tflite(model, args.quantize)
        path = os.path.join(args.out, "model.tflite")
        with open(path, "wb") as f:
            f.write(tflite_bytes)
        print(f"Saved {path} ({len(tflite_bytes) / 1024:.0f} KB, quantize={args.quantize})")

    if args.image:
        x = load_image(args.image, spec["imageSize"])
        results = [("keras", model.predict(x, verbose=0)[0])]
        if tflite_bytes is not None:
            results.append(("tflite", run_tflite(tflite_bytes, x)[0]))
        for name, probs in results:
            ranked = sorted(zip(labels, probs), key=lambda p: -p[1])
            print(f"{name}: " + ", ".join(f"{label} {p:.1%}" for label, p in ranked))


if __name__ == "__main__":
    main()
`;
}

/** head_weights.json + metadata.json + labels.txt + the conversion script. */
export async function buildPythonKit(
  trained: TrainedModel,
  labels: string[],
  target: PythonTarget,
  quantize: TfliteQuantization = 'none',
): Promise<Blob> {
  const zip = new JSZip();
  zip.file('head_weights.json', await buildHeadWeightsJson(trained, labels));
  zip.file('metadata.json', json(buildMetadata(trained, labels)));
  zip.file('labels.txt', labels.join('\n') + '\n');
  zip.file(pythonFilename(target), pythonScript(target, quantize));
  return zip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
}

export async function exportKeras(trained: TrainedModel, labels: string[], projectName = 'model'): Promise<void> {
  triggerDownload(await buildPythonKit(trained, labels, 'keras'), `${slugify(projectName)}-keras.zip`);
}

export async function exportTflite(
  trained: TrainedModel,
  labels: string[],
  quantize: TfliteQuantization = 'none',
  projectName = 'model',
): Promise<void> {
  triggerDownload(await buildPythonKit(trained, labels, 'tflite', quantize), `${slugify(projectName)}-tflite.zip`);
}
