#!/usr/bin/env node
// One-time setup: downloads every pre-trained model into public/models so the
// app runs fully offline afterwards (no CDN is contacted at runtime).
//
//   public/models/coco-ssd/       COCO-SSD lite_mobilenet_v2 (80 COCO classes, tfjs graph)
//   public/models/blazeface/      MediaPipe BlazeFace short-range face detector (tfjs graph)
//   public/models/hand-detector/  MediaPipe Hands palm detector, lite (tfjs graph)
//   public/models/hand-landmark/  MediaPipe Hands 21-keypoint landmark model, lite (tfjs graph)
//   public/models/mobilenet/      MobileNet V2 1.0 224 ImageNet; its 1280-d global-average-pool
//                                 layer is the frozen feature extractor for custom training
//
// These are exactly the files the @tensorflow-models/* packages fetch from
// TF Hub / Google Cloud Storage by default.
//
// Usage: npm run fetch-models          (add --force to re-download)
//        node scripts/fetch-models.mjs --soft   (postinstall: warn instead of failing when offline)

import { access, mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'models');
const TFHUB = (path) => ({ base: `https://tfhub.dev/${path}`, query: '?tfjs-format=file' });
const MODELS = {
  'coco-ssd': { base: 'https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2', query: '' },
  blazeface: TFHUB('mediapipe/tfjs-model/face_detection/short/1'),
  'hand-detector': TFHUB('mediapipe/tfjs-model/handpose_3d/detector/lite/1'),
  'hand-landmark': TFHUB('mediapipe/tfjs-model/handpose_3d/landmark/lite/1'),
  mobilenet: TFHUB('google/imagenet/mobilenet_v2_100_224/classification/2'),
};
const force = process.argv.includes('--force');
const soft = process.argv.includes('--soft');

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function get(url, attempts = 3) {
  for (let i = 1; ; i++) {
    try {
      const res = await fetch(url, { redirect: 'follow' });
      if (!res.ok) throw new Error(`GET ${url}: HTTP ${res.status}`);
      return Buffer.from(await res.arrayBuffer());
    } catch (err) {
      // Large shards occasionally drop mid-stream; a retry almost always works.
      if (i >= attempts) throw err;
      await new Promise((r) => setTimeout(r, 1000 * i));
    }
  }
}

let failed = 0;
for (const [name, { base, query }] of Object.entries(MODELS)) {
  const dir = join(ROOT, name);
  if (!force && (await exists(join(dir, 'model.json')))) {
    console.log(`${name}: already present (use --force to re-download)`);
    continue;
  }
  try {
    await mkdir(dir, { recursive: true });
    const modelJson = await get(`${base}/model.json${query}`);
    const manifest = JSON.parse(modelJson.toString('utf8'));
    let bytes = modelJson.length;
    for (const group of manifest.weightsManifest) {
      for (const path of group.paths) {
        const shard = await get(`${base}/${path}${query}`);
        await writeFile(join(dir, path), shard);
        bytes += shard.length;
      }
    }
    // model.json last, so a half-finished download is retried next time.
    await writeFile(join(dir, 'model.json'), modelJson);
    console.log(`${name}: ${(bytes / 1024 / 1024).toFixed(1)} MB -> ${dir}`);
  } catch (err) {
    failed++;
    console.error(`${name}: download failed (${err.message})`);
  }
}

if (failed) {
  const msg = `${failed} model(s) missing. Run "npm run fetch-models" when online; the app shows which detector is unavailable.`;
  if (soft) console.warn(`[fetch-models] ${msg}`);
  else {
    console.error(msg);
    process.exit(1);
  }
}
