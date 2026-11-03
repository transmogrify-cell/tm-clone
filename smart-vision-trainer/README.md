# Smart Vision Trainer

A local, browser-only, Teachable-Machine-style vision lab that does two things side by side:

- **Mode A: zero-shot pretrained detection.** Works the moment the page loads. COCO-SSD draws boxes around 80 everyday object types (person, car, bicycle, motorcycle, bottle, cup, cell phone, chair, book, laptop…), BlazeFace finds faces, and MediaPipe Hands tracks 21 keypoints per hand.
- **Mode B: custom training.** Add your own classes ("Alice", "MyBottle", "RedCar"), each typed **Person** (learns a face crop) or **Object** (learns the whole image). A small head trains in the browser on frozen MobileNet V2 features, then you get a per-class accuracy report.

100% local: no server, no cloud, no CDN at runtime. Images never leave the device. All five models are served from `public/models`.

```bash
npm install      # also downloads the 5 pretrained models into public/models (~35 MB, once)
npm run dev      # http://localhost:5175
```

Node 20.19+ is required. If the model download failed during install (offline, proxy), run `npm run fetch-models` later. The app shows which model is missing. `npm run build` produces a static `dist/` you can host anywhere (`base: './'`). The webcam needs a secure context: `localhost` or HTTPS.

---

## Contents

1. [Architecture](#architecture)
2. [Mode A vs Mode B](#mode-a-vs-mode-b)
3. [Mode A: pretrained detectors](#mode-a-pretrained-detectors)
4. [Mode B: the custom training pipeline](#mode-b-the-custom-training-pipeline)
5. [Accuracy report](#accuracy-report)
6. [Preview: dual-path inference and Detect + Classify](#preview-dual-path-inference-and-detect--classify)
7. [Export](#export)
8. [Using the pretrained detectors standalone](#using-the-pretrained-detectors-standalone)
9. [Limitations](#limitations)
10. [Performance and robustness](#performance-and-robustness)
11. [Demo test](#demo-test)
12. [File structure](#file-structure)

---

## Architecture

```
                         ┌──────────────── public/models (served by the app itself) ────────────────┐
                         │ coco-ssd/  blazeface/  hand-detector/  hand-landmark/  mobilenet/         │
                         └───────────────────────────────────────────────────────────────────────────┘
 webcam (getUserMedia)                                │ loaded once, on app start (TF.js WebGL, CPU fallback)
 or uploaded photo ──► snapshot canvas ──┬────────────┴───────────────────────────────┐
                                         │                                            │
               MODE A (every frame, ≤10 fps, Promise.all)           MODE B (only after you train)
         ┌──────────────┬──────────────┬──────────────┐       ┌──────────────────────────────────────────┐
         │  COCO-SSD    │  BlazeFace   │ MediaPipe    │       │ face crop (Person) / letterbox (Object)   │
         │  80 classes  │  face boxes  │ Hands, 21 kp │       │   ─► frozen MobileNet V2 ─► 1280-d        │
         └──────┬───────┴──────┬───────┴──────┬───────┘       │   ─► trained head ─► softmax ─► threshold │
                └──────── canvas overlay ─────┘               └──────────────────────────────────────────┘
                   (boxes, labels, skeleton)                     "Detect + Classify": Mode A boxes are
                                                                  cropped and labelled by Mode B
```

- **React 18 + Vite + TypeScript + Tailwind**, **Zustand** for state (`src/store/useProjectStore.ts`).
- **TensorFlow.js 4.22** (`@tensorflow/tfjs`, WebGL backend, CPU fallback) shared by all models.
- `@tensorflow-models/coco-ssd`, `@tensorflow-models/face-detection` (BlazeFace), `@tensorflow-models/hand-pose-detection` (MediaPipe Hands) and `@tensorflow-models/mobilenet`, all on their **tfjs runtimes**. The optional MediaPipe WASM runtimes would be fetched from a CDN, so Vite aliases `@mediapipe/face_detection` and `@mediapipe/hands` to a stub (`src/ml/mediapipeStub.ts`) and nothing from a CDN is bundled.
- The camera stream is shared and reference-counted (`src/utils/camera.ts`), so the Detect view and a class card don't fight over it.
- Samples, features and the trained head live in memory only. A reload warns before discarding them.

UI: tabs **Detect** (Mode A live, plus a Mode B summary panel and the live custom answer), **Train** (class cards, training, accuracy report), **Preview** (Mode B inference, threshold, Detect + Classify) and **Export**.

## Mode A vs Mode B

| | Mode A: pretrained detection | Mode B: custom training |
|---|---|---|
| Needs training | No | Yes, ≥20 samples per class (≥10 for Unknown) |
| Output | **Boxes** + label + score (and hand keypoints) | **One label** for the image (or for a crop) + confidence |
| Vocabulary | Fixed: 80 COCO classes, "face", "hand" | Anything you collect samples for |
| Models | COCO-SSD lite_mobilenet_v2, BlazeFace short, MediaPipe Hands lite | Frozen MobileNet V2 1.0/224 + a ~360k-parameter head |
| Good for | Demos, baselines, "is there a person / car / bottle?" | "Is this *my* bottle?", "Is this Alice or Bob?" |

They combine: **Detect + Classify** (Preview tab) uses Mode A to find *where* things are and Mode B to say *which* of your classes each one is.

## Mode A: pretrained detectors

All three models load **in parallel** on app start (`useDetectors`), then MobileNet. Every frame (webcam) or once per photo:

1. The video frame is copied to a canvas so all models see identical pixels.
2. The enabled detectors run concurrently (`Promise.all` in `runDetectors`), throttled to **10 fps** by a `requestAnimationFrame` loop with a timestamp check. A new frame never starts while the previous one is running.
3. Boxes are drawn on a `<canvas>` overlay (`src/utils/canvasOverlay.ts`) at source resolution: COCO objects in per-class colors (person red, car blue, bottle green, bicycle orange…) with label + score, faces in yellow, hands in purple with the 21-keypoint skeleton. The webcam view is mirrored (selfie). The overlay flips box coordinates rather than the canvas, so text stays readable.
4. The side panel lists everything found, e.g. `Detected (3): person 0.94, car 0.88, bottle 0.76`. A checkbox turns each detector on or off.

| Detector | Package / model | Output |
|---|---|---|
| COCO-SSD | `coco-ssd`, `lite_mobilenet_v2` base (SSDLite, the fastest variant; 17.7 MB of weights) | up to 20 boxes, score ≥ 0.50 |
| Faces | `face-detection`, MediaPipe BlazeFace short-range, tfjs | face boxes (up to 5) |
| Hands | `hand-pose-detection`, MediaPipe Hands lite (palm detector + landmark model), tfjs | up to 2 hands, 21 keypoints, handedness |

Hands **track** across video frames (palm detection only reruns when a hand is lost). Still photos use `staticImageMode`. MediaPipe reports handedness for a mirrored image; we pass true pixels, so the label is swapped to match the person's actual hand.

### The 80 COCO classes

person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

Anything else (a specific mug, a brand, a person's identity) is invisible to Mode A. That is what Mode B is for.

## Mode B: the custom training pipeline

Each step is commented inline in `src/ml/*`.

**STEP 1: preprocessing by class type** (`preprocessing.ts`, `faceDetector.ts`)

| Class type | What is kept | Why |
|---|---|---|
| **Person** | BlazeFace → exactly one face → square crop with **20% margin** per side → 224×224 (gray where the margin leaves the frame) | A full frame is mostly background and clothes. A face crop makes the head learn identity, not scenery. Photos with no face or several faces are skipped (and counted in a toast). |
| **Object** | The whole image **letterboxed** to 224×224 (aspect ratio kept, black padding) | No stretching, so shape cues survive. |
| **Unknown** (optional) | The letterboxed frame, **plus** the face crop when there is exactly one face | Inference is dual-path, so the rejection class needs examples of what both paths see. |

Both views are normalized to **[-1, 1]**. The pixels go in as [0, 1], and the MobileNet graph's own first two ops (`hub_input/Mul`, `hub_input/Sub`) compute `x·2 − 1`. Samples are stored as 224×224 JPEG data URLs, up to **500 per class**.

**STEP 2: frozen MobileNet V2** (`mobilenet.ts`). This is MobileNet V2 1.0/224 (ImageNet) with the classifier cut off. Its 1280-d global-average-pool output is the feature vector. It is a GraphModel and only ever runs inference, so it is frozen by construction.

**STEP 3: 6× augmentation** (`augmentation.ts`, GPU tf ops, before feature extraction): original, horizontal flip, brightness ±15%, rotation ±10°, zoom 0.9×, zoom 1.1×. Brightness and rotation alternate direction per sample (seeded by the sample id), so they are deterministic and their features can be cached. Out-of-frame pixels use the sample's own padding color.

**STEP 4: trainable head** (`trainer.ts`)

```
1280 ─► Dense 256 relu (L2 0.001) ─► BatchNorm ─► Dropout 0.4
     ─► Dense 128 relu (L2 0.001) ─► Dropout 0.3 ─► Dense N softmax

adam(0.0005) · categoricalCrossentropy · 40 epochs · batch 16 · validation 20%
EarlyStopping(monitor val_loss, patience 5, restoreBestWeights)
```

- Features are extracted once per augmented image and **cached by sample id**, so retraining after adding samples only processes the new ones.
- The 20% validation split is **stratified per class and made by sample, before augmentation**. If augmented variants were split instead, a flipped copy of a validation image would sit in the training set and inflate the report. Validation uses only the original, un-augmented sample.
- Balanced class weights stop a smaller class (typically Unknown) from being drowned out.
- `tf.callbacks.earlyStopping` in TF.js cannot restore weights, so the trainer keeps a copy of the best epoch's weights and restores them. The UI shows which epoch was kept. **Stop** ends training early and still keeps the best weights.
- Epochs, batch size, learning rate and patience are editable under *Settings*.

**STEP 5: evaluation.** See [Accuracy report](#accuracy-report).

**STEP 6: dual-path inference.** See [Preview](#preview-dual-path-inference-and-detect--classify).

**STEP 7: Unknown class + sample minimums.** **Train** stays disabled until there are at least 2 classes, every class has **≥ 20 samples**, the Unknown class (if added) has **≥ 10**, and names are non-empty and unique. The button shows exactly what is missing (e.g. `Class 1 12/20`). "Good" quality starts at 40 (20 for Unknown).

## Accuracy report

Shown automatically after training (`Train → Accuracy report`), computed on the held-out validation samples (`evaluator.ts`):

- **Overall accuracy**: correct ÷ held-out samples, plus which epoch's weights were kept and whether early stopping fired.
- **Per-class table**: **Precision** (when it predicts *X*, how often is it *X*: diagonal ÷ column), **Recall** (of the *X* samples, how many it got: diagonal ÷ row), F1, **Support** (held-out samples behind the row) and total samples. Values below 70% are red.
- **Confusion matrix heatmap**: rows are the actual class, columns the prediction, and color is the share of the row. Hover a cell to see e.g. "Alice samples predicted as Bob: 2 of 9".
- **Warnings**: any class with **recall < 0.70** (or precision < 0.70) gets a concrete suggestion, e.g. *"MyBottle: recall is 55% (below 70%): only 6 of 11 test samples were recognised. Add more photos of MyBottle, from different angles, distances and backgrounds."* A small test set is flagged too, since each sample then moves recall by many points.
- **Threshold impact**: for the current threshold, the share of held-out samples that get a confident answer, the accuracy among those, and how many of your own samples would turn into "Unknown".

## Preview: dual-path inference and Detect + Classify

**Dual path** (`predictor.ts → predictDualPath`), for each frame or photo:

```
a) largest face ─► 20%-margin crop ─► MobileNet ─► head ─► probs_face
b) whole frame  ─► letterbox       ─► MobileNet ─► head ─► probs_full      (a and b share one batched MobileNet call)
c) keep the path with the higher top probability
d) top class is the Unknown class, or confidence < threshold (default 0.60, slider 0.30–0.90)  ─►  "Unknown"
```

A path only runs if the model has classes of its type. The face path needs a Person class and a face in view. The full path needs an Object class. A head that never saw full frames of people would otherwise give confident nonsense on that path. The Preview panel shows both paths' probabilities and which one won. The Detect tab can also show the custom answer live, next to Mode A.

**Detect + Classify** (the hybrid toggle in Preview, `predictHybrid`): COCO-SSD boxes and BlazeFace faces localize. Each region is cropped (COCO boxes +10% margin, letterboxed; faces through the Person crop) and classified by the head. A box gets your label (in your class color) when the head is confident. Otherwise it keeps its COCO label, dashed. Up to 6 regions are classified per frame. This is how a custom class gets a box. It only works for things Mode A can localize: a "MyBottle" class works because COCO-SSD finds bottles. Tip: capture Object samples close-up, so the object fills most of the frame, as a hybrid crop does.

## Export

**Export tab → TensorFlow.js**: a `.zip` (or each file on its own):

| File | Content |
|---|---|
| `model.json` | Layers-model topology of the **head** + weight manifest |
| `weights.bin` | Head weights (float32) |
| `metadata.json` | Everything needed to reproduce the preprocessing |
| `convert_to_keras.py` | (zip only) the Keras converter below |

```json
{
  "labels": ["Alice", "MyBottle", "Unknown"],
  "classTypes": ["person", "object", "unknown"],
  "imageSize": 224,
  "normalization": "[-1,1]",
  "baseModel": "mobilenet_v2",
  "confidenceThreshold": 0.6,
  "dualPathInference": true,
  "paths": { "face": true, "full": true },
  "unknownLabel": "Unknown",
  "faceDetector": "mediapipe_blazeface_short",
  "faceMargin": 0.2,
  "letterboxFill": 0,
  "featureDim": 1280,
  "mobilenet": { "version": 2, "alpha": 1, "inputRange": [0, 1], "embeddingLayer": "module_apply_default/MobilenetV2/Logits/AvgPool",
                 "source": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2" },
  "createdAt": "…", "generator": "smart-vision-trainer", "version": 1
}
```

`dualPathInference` is true when the model has both Person and Object classes. `paths` says which path(s) to run. MobileNet and BlazeFace are not in the export. Load them from their packages wherever you use the head. The Export tab shows a copyable `classify.js` that does this in plain TF.js.

**Keras (.h5)**: browsers can't write HDF5, so **Show Python snippet** opens a copyable `convert_to_keras.py`. It reads `model.json`'s weight manifest, slices `weights.bin` with numpy, rebuilds the identical head in `tf.keras` (same layer names) and saves `custom_head.h5` + `labels.txt`. It needs only `pip install "tensorflow>=2.12"`. The head's input is the 1280-d MobileNet V2 feature vector, so in Python use TF-Hub `google/imagenet/mobilenet_v2_100_224/feature_vector/2` with pixels in [0, 1], after the same face crop or letterbox.

## Using the pretrained detectors standalone

These snippets don't need this app. Without `modelUrl` / `detectorModelUrl`, the packages download the models from Google's TF Hub / GCS. Point them at your own copy (as this app does with `public/models`) to stay offline. `npm run fetch-models` downloads them.

**COCO-SSD**

```js
import '@tensorflow/tfjs-backend-webgl';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const model = await cocoSsd.load({ base: 'lite_mobilenet_v2' /*, modelUrl: '/models/coco-ssd/model.json' */ });
const detections = await model.detect(imgOrVideoOrCanvas, 20, 0.5); // maxBoxes, minScore
// [{ bbox: [x, y, width, height], class: 'bottle', score: 0.76 }, ...]
```

**BlazeFace (MediaPipe face detection)**

```js
import '@tensorflow/tfjs-backend-webgl';
import * as faceDetection from '@tensorflow-models/face-detection';

const detector = await faceDetection.createDetector(faceDetection.SupportedModels.MediaPipeFaceDetector, {
  runtime: 'tfjs', modelType: 'short', maxFaces: 5,
  // detectorModelUrl: '/models/blazeface/model.json',
});
const faces = await detector.estimateFaces(imgOrVideoOrCanvas);
// [{ box: { xMin, yMin, width, height, ... }, keypoints: [6 points: eyes, nose, mouth, ears] }]
```

**MediaPipe Hands**

```js
import '@tensorflow/tfjs-backend-webgl';
import * as handPose from '@tensorflow-models/hand-pose-detection';

const detector = await handPose.createDetector(handPose.SupportedModels.MediaPipeHands, {
  runtime: 'tfjs', modelType: 'lite', maxHands: 2,
  // detectorModelUrl: '/models/hand-detector/model.json', landmarkModelUrl: '/models/hand-landmark/model.json',
});
const hands = await detector.estimateHands(video, { flipHorizontal: false, staticImageMode: false });
// [{ keypoints: [21 × { x, y, name: 'wrist' | 'thumb_tip' | ... }], handedness: 'Left' | 'Right', score }]
// Note: handedness assumes a mirrored (selfie) image; swap it for unmirrored input.
```

With a bundler, alias `@mediapipe/face_detection` / `@mediapipe/hands` to a stub when you only use the `tfjs` runtime (see `vite.config.ts`). Otherwise their WASM runtimes are bundled.

## Limitations

- **Mode A (COCO-SSD) only detects the 80 COCO classes.** A custom object ("my red mug") won't be detected without Mode B training, and even then only as a classification (next point).
- **Mode B is classification, not detection.** It says "this image is mostly class X" but does **not** draw a bounding box. A box for a custom class needs a real detector (YOLOv8, EfficientDet…) trained on annotated boxes, which is out of scope. The **Detect + Classify** toggle is the practical middle ground: Mode A localizes (e.g. a bottle), the crop is classified by Mode B ("MyBottle"). It only finds what Mode A can find.
- BlazeFace (short range) is tuned for faces within ~2 m of a webcam. It misses small, distant faces and sometimes fires on animal faces or clutter. Glance over a Person class's crops and delete the strays.
- A Person class learns faces, so the same person facing away, masked or far away won't match. Lookalikes and siblings can be confused. Lighting and camera changes between training and use hurt accuracy.
- Accuracy numbers come from a small held-out set (20% of your samples, from the same sessions). Real-world accuracy is usually lower. Collect varied samples (angles, lighting, backgrounds, distances).
- COCO-SSD lite trades accuracy for speed: small or partially hidden objects are often missed at the default 0.5 score.
- **Not for security, access control or authentication.** It is easy to fool with a photo. This is an educational / demo tool.
- Everything lives in memory: reloading the page loses samples and the trained model (export it first).

## Performance and robustness

- COCO-SSD `lite_mobilenet_v2` base; detection throttled to 10 fps with rAF + timestamp; no overlapping frames.
- Detectors run in parallel with `Promise.all`; boxes are drawn on a canvas, not DOM nodes.
- Each model warms up once on load (shader compilation) so the first real frame isn't slow. `WEBGL_USE_SHAPES_UNIFORMS` makes shaders shape-agnostic, so a photo of a new size doesn't trigger recompilation (measured: ~1 s → ~150 ms for the first detection on a new size).
- Every detection or classification run takes a ticket, and a run that finishes after a newer one started is dropped. A slow result can never overwrite the boxes for a newer photo or setting.
- `tf.tidy()` wraps every inference, augmentation and feature-extraction step; tensors outside a tidy (training data, best-weight copies, batched outputs) are disposed in `finally` blocks. An old head is disposed when a new one replaces it.
- 500 samples per class cap; a 1280-px cap on decoded uploads.
- WebGL backend with automatic CPU fallback (with a warning that it is slow).
- Handled cases: camera missing / denied / busy / unplugged (with "Use a photo instead"), each model failing to load independently (status chip + Retry; the rest keep working), out-of-memory during training (friendly message), photos with no face / several faces for Person classes, empty or duplicate class names, too few samples, data changed after training ("retrain" banner).

## Demo test

Recorded on 2026-11-03 in the Claude desktop app's browser pane (Chromium, WebGL backend), against `npm run dev`. The pane blocks webcams, which exercised the **no-camera path**: the error message and "Use a photo instead" appear, and everything below ran on uploaded photos through the real UI (file input → same pipeline as the webcam). Test photos came from COCO val2017 and Wikimedia Commons. They were kept outside the repo.

**Mode A: zero-shot detection on photos** (Detect tab, all three detectors on, score ≥ 0.50):

| Photo | Detected (as listed in the side panel) |
|---|---|
| COCO `000000303818` (street) | **car 0.94**, bus 0.90, **person** 0.81 / 0.80 / 0.79 / 0.73 / 0.68 / 0.59 |
| Wikimedia "VIP boards and water bottles" | **person** 0.96 / 0.95, dining table 0.68, **bottle 0.55**, left hand 0.99 (21-keypoint skeleton drawn) |
| Wikimedia "baby in a car seat with a bottle" | **bottle 0.72**, left hand 0.96 |
| COCO `000000463730` (street) | bus 0.99, person 0.92 / 0.84 / 0.77 / 0.76 / 0.75 / 0.69, bus 0.79, **car 0.77**, face ×2 |
| COCO `000000174482` | bicycle 0.93, car 0.82 / 0.72 / 0.57 / 0.54 |
| COCO `000000480985` | person 0.62, motorcycle 0.56 |

Person, car and bottle are each detected with boxes, labels and scores. The overlay was checked by compositing photo + canvas: per-class colors, yellow faces, purple hand skeleton. No single test photo contained all three at once, and a two-photo collage loses the bottle: COCO-SSD shrinks the whole input to 300×300, so a bottle in half a collage gets too small. Per photo: ~120–250 ms for the three detectors together; a repeated size ~120 ms.

**Mode B: custom training with an accuracy report** (Train tab):

- Classes: `RedMug` (Object, 30 samples), `BlueBottle` (Object, 30), `Unknown` (14). The images were generated in the page: random backgrounds, clutter and noise, with a red mug or a blue bottle at a random position, scale, rotation and hue.
- Gate: with Unknown at 0 samples, Train was disabled with *"Need 20+ samples per class and 10+ for Unknown (Unknown 0/10)"*. It unlocked at 14.
- Training: 74 samples × 6 augmentations; **early stopping after 26 epochs, weights restored from epoch 21**; ~35 s.
- Report: **overall accuracy 100.0% on 15 held-out samples**. Precision, recall and F1 were 100% for each class (support 6 / 6 / 3); the confusion matrix was diagonal; the small-test-set warning fired. The recall < 0.70 warning was checked separately on a fabricated evaluation: *"MyBottle: recall is 50% (below 70%): only 2 of 4 test samples were recognised…"*.
- Preview (dual path): 4 freshly generated images were each classified correctly at 1.00 via the full-frame path. That is the only path, since the model has no Person classes.
- Detect + Classify on the water-bottle photo: COCO boxes were re-labelled by the head. The bottle and table came out *Unknown* (dashed, COCO label kept), while the two people came out "RedMug" 0.64 / 0.83. That is a classifier being forced onto inputs unlike anything it trained on, exactly the limitation described above.
- Export: the zip contains `model.json`, `weights.bin` (1.45 MB), `metadata.json` and `convert_to_keras.py`. The head reloaded from the zip gives **identical** predictions (max abs diff 0). The weight names match the ones the Keras script expects.
- Person preprocessing: 5 photos uploaded to a Person class gave 3 face crops (20% margin, square). The toast read *"Skipped: 1 no face found, 1 multiple faces"*. Two of the three "faces" were a cat and a dog (see Limitations).
- Memory: `tf.memory().numTensors` stayed at 1001 across repeated Detect, dual-path and hybrid runs (no leaks).

Not covered here: live webcam capture and live webcam detection (no camera in the test browser). Those paths share the same detection, preprocessing and drawing code as photos, plus the camera hook carried over from this repo's earlier apps.

To reproduce Mode A: open the Detect tab, click **Upload image**, pick any street or kitchen photo. To reproduce Mode B: add two Object classes, drop 20+ photos on each card (or hold **Hold to capture** in front of the webcam), click **Train model**.

## File structure

```
smart-vision-trainer/
├── index.html
├── package.json             # postinstall downloads the models
├── vite.config.ts           # @mediapipe/* aliased to a stub (tfjs runtimes only)
├── tsconfig.json
├── tailwind.config.js · postcss.config.js
├── scripts/fetch-models.mjs # downloads the 5 models into public/models
├── public/models/           # coco-ssd · blazeface · hand-detector · hand-landmark · mobilenet
└── src/
    ├── main.tsx · App.tsx · index.css · config.ts
    ├── types/index.ts
    ├── store/useProjectStore.ts · useToastStore.ts
    ├── ml/
    │   ├── cocoDetector.ts      # loadCocoSsd(), detectCoco(), COCO_CLASSES
    │   ├── faceDetector.ts      # BlazeFace + 20%-margin face crop
    │   ├── handDetector.ts      # MediaPipe Hands, keypoint boxes, skeleton bones
    │   ├── mobilenet.ts         # backend init + frozen feature extractor
    │   ├── preprocessing.ts     # person (face crop) vs object (letterbox) vs unknown (both)
    │   ├── augmentation.ts      # 6x variants
    │   ├── trainer.ts           # feature cache, stratified split, head, early stopping
    │   ├── evaluator.ts         # accuracy, precision/recall/F1, confusion matrix, warnings
    │   ├── predictor.ts         # dual-path + threshold, hybrid Detect + Classify
    │   ├── exporter.ts          # TF.js files/zip, Keras + JS snippets
    │   └── mediapipeStub.ts
    ├── utils/  image.ts · camera.ts · canvasOverlay.ts · download.ts
    ├── hooks/  useDetectors.ts (load models, runDetectors) · useTraining.ts · useInference.ts (frame source + throttled loop)
    └── components/
        ├── Landing.tsx · AppHeader.tsx · SourceView.tsx
        ├── DetectTab.tsx        # Mode A live + Mode B summary
        ├── TrainTab.tsx · ClassCard.tsx · SampleGallery.tsx · WebcamCapture.tsx
        ├── TrainingProgress.tsx · AccuracyReport.tsx
        ├── PreviewTab.tsx       # dual-path result, threshold, Detect + Classify
        ├── ExportTab.tsx
        └── ui/                  # Button, Card, Modal, CodeBlock, ProgressBar, LineChart, Toaster, icons
```
