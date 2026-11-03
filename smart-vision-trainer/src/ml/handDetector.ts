/**
 * Mode A: MediaPipe Hands (palm detector + 21-keypoint landmark model), lite,
 * tfjs runtime via @tensorflow-models/hand-pose-detection. Served from
 * public/models/hand-detector and public/models/hand-landmark.
 *
 * On video the detector tracks: it reuses the previous frame's hand regions
 * and only re-runs palm detection when a hand is lost (much faster). Still
 * images use staticImageMode so every call starts fresh.
 */
import * as handPose from '@tensorflow-models/hand-pose-detection';
import { initBackend, modelLoadError, MODELS_BASE } from './mobilenet';
import type { Detectable } from './faceDetector';
import type { Box, HandDetection, Keypoint } from '../types';

export const HAND_DETECTOR_URL = `${MODELS_BASE}/hand-detector/model.json`;
export const HAND_LANDMARK_URL = `${MODELS_BASE}/hand-landmark/model.json`;

/** Bones between the 21 landmarks (wrist = 0, then 4 per finger from thumb to pinky). */
export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

let detectorPromise: Promise<handPose.HandDetector> | null = null;

export function loadHandDetector(): Promise<handPose.HandDetector> {
  detectorPromise ??= (async () => {
    await initBackend();
    let detector: handPose.HandDetector;
    try {
      detector = await handPose.createDetector(handPose.SupportedModels.MediaPipeHands, {
        runtime: 'tfjs',
        modelType: 'lite',
        maxHands: 2,
        detectorModelUrl: HAND_DETECTOR_URL,
        landmarkModelUrl: HAND_LANDMARK_URL,
      });
    } catch (err) {
      throw modelLoadError('MediaPipe Hands', HAND_DETECTOR_URL, err);
    }
    const blank = document.createElement('canvas');
    blank.width = blank.height = 64;
    await detector.estimateHands(blank, { staticImageMode: true });
    return detector;
  })().catch((err) => {
    detectorPromise = null;
    throw err;
  });
  return detectorPromise;
}

/** Bounding box of the keypoints, padded by 10% so the fingertips sit inside it. */
export function keypointBox(points: Keypoint[], pad = 0.1): Box {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const x0 = Math.min(...xs);
  const y0 = Math.min(...ys);
  const w = Math.max(...xs) - x0;
  const h = Math.max(...ys) - y0;
  return { x: x0 - w * pad, y: y0 - h * pad, width: w * (1 + 2 * pad), height: h * (1 + 2 * pad) };
}

export async function detectHands(source: Detectable, staticImage = false): Promise<HandDetection[]> {
  const detector = await loadHandDetector();
  const hands = await detector.estimateHands(source, { flipHorizontal: false, staticImageMode: staticImage });
  return hands
    .filter((h) => h.keypoints?.length === 21)
    .map((h) => {
      const keypoints = h.keypoints.map(({ x, y, name }) => ({ x, y, name }));
      // MediaPipe labels handedness assuming a mirrored (selfie) input. We pass
      // the true, unmirrored pixels, so swap to get the person's actual hand.
      const handedness = h.handedness === 'Left' ? 'Right' : 'Left';
      return { box: keypointBox(keypoints), keypoints, handedness, score: h.score };
    });
}

/** Forget tracked hands (call when switching between video and a still image). */
export async function resetHandTracking(): Promise<void> {
  if (detectorPromise) (await detectorPromise.catch(() => null))?.reset();
}
