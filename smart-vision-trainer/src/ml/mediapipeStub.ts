// @tensorflow-models/face-detection and @tensorflow-models/hand-pose-detection
// statically import @mediapipe/face_detection and @mediapipe/hands for their
// optional "mediapipe" runtime (a WASM build fetched from a CDN). This app only
// uses the "tfjs" runtime, so vite.config.ts aliases both imports here: no CDN
// code is bundled, and choosing the mediapipe runtime fails loudly.
const fail = (): never => {
  throw new Error('The MediaPipe runtime is not bundled; use runtime: "tfjs".');
};

export class FaceDetection {
  constructor() {
    fail();
  }
}

export class Hands {
  constructor() {
    fail();
  }
}
