/**
 * Mode A: COCO-SSD object detection (80 classes, boxes + scores), zero-shot.
 *
 * Base model: lite_mobilenet_v2 (SSDLite + MobileNet V2), the fastest of the
 * three COCO-SSD variants. Served from public/models/coco-ssd.
 */
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { COCO_MAX_BOXES, COCO_MIN_SCORE } from '../config';
import { initBackend, modelLoadError, MODELS_BASE } from './mobilenet';
import type { Detectable } from './faceDetector';
import type { CocoDetection } from '../types';

export const COCO_URL = `${MODELS_BASE}/coco-ssd/model.json`;

/** The 80 COCO-SSD labels, exactly as the model returns them. */
export const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
  'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
  'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
  'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
] as const;

let modelPromise: Promise<cocoSsd.ObjectDetection> | null = null;

export function loadCocoSsd(): Promise<cocoSsd.ObjectDetection> {
  modelPromise ??= (async () => {
    await initBackend();
    let model: cocoSsd.ObjectDetection;
    try {
      model = await cocoSsd.load({ base: 'lite_mobilenet_v2', modelUrl: COCO_URL });
    } catch (err) {
      throw modelLoadError('COCO-SSD', COCO_URL, err);
    }
    // Warm-up: the first call compiles the WebGL programs.
    const blank = document.createElement('canvas');
    blank.width = blank.height = 64;
    await model.detect(blank, 1);
    return model;
  })().catch((err) => {
    modelPromise = null;
    throw err;
  });
  return modelPromise;
}

/** Boxes for the 80 COCO classes, highest score first. */
export async function detectCoco(
  source: Detectable,
  maxBoxes = COCO_MAX_BOXES,
  minScore = COCO_MIN_SCORE,
): Promise<CocoDetection[]> {
  const model = await loadCocoSsd();
  const found = await model.detect(source, maxBoxes, minScore);
  return found
    .map(({ bbox: [x, y, width, height], class: label, score }) => ({ box: { x, y, width, height }, label, score }))
    .sort((a, b) => b.score - a.score);
}
