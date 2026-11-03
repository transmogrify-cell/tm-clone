#!/usr/bin/env node
// One-time setup helper: downloads the pre-trained MobileNet V2 (alpha 0.35,
// 224x224, ImageNet, no classification top) in TensorFlow.js Layers format
// into public/models/mobilenet so the app never touches the network at runtime.
//
// These are the Keras `mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top`
// ImageNet weights converted to tfjs, which is also what
// `tf.keras.applications.MobileNetV2(alpha=0.35, include_top=False)` loads in
// Python. That match is what lets the Keras / TF Lite export rebuild the exact
// same feature extractor.
//
// Usage: npm run fetch-model   (add --force to re-download)

import { mkdir, writeFile, access } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const BASE =
  'https://storage.googleapis.com/teachable-machine-models/' +
  'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top';
const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'models', 'mobilenet');
const force = process.argv.includes('--force');

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function download(name) {
  const res = await fetch(`${BASE}/${name}`);
  if (!res.ok) throw new Error(`GET ${name}: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await writeFile(join(OUT_DIR, name), buf);
  console.log(`  ${name}  ${(buf.length / 1024).toFixed(0)} KB`);
  return buf;
}

await mkdir(OUT_DIR, { recursive: true });

if (!force && (await exists(join(OUT_DIR, 'model.json')))) {
  console.log(`MobileNet already present in ${OUT_DIR} (use --force to re-download).`);
  process.exit(0);
}

console.log(`Downloading MobileNet V2 0.35 into ${OUT_DIR}`);
const manifest = JSON.parse((await download('model.json')).toString('utf8'));
for (const group of manifest.weightsManifest) {
  for (const path of group.paths) await download(path);
}
console.log('Done.');
