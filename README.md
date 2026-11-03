# tm-clone: a local, in-browser image classifier trainer

A self-contained clone of the Teachable Machine **Image Project**. Collect images from your webcam or from files, train a classifier with **transfer learning** directly in the browser, preview it live, and export it to TensorFlow.js, TensorFlow Lite or Keras.

Everything runs locally. There is no backend, no cloud API and no telemetry, and images never leave the browser tab. Even the pre-trained MobileNet weights are served from this repo (`public/models/mobilenet`).

## Quick start

```bash
npm install
npm run dev        # http://localhost:5173
```

Requirements: Node 20.19+ and a modern Chromium, Firefox or Safari. Webcam access needs a secure context, and `localhost` counts as one.

```bash
npm run build      # type-check + production build into dist/ (a static site)
npm run preview    # serve dist/ locally
npm run fetch-model -- --force   # re-download the MobileNet weights (already committed)
```

## Using it

1. **Image Project** → you start with two classes. Rename them by clicking the name.
2. Add samples to each class:
   - **Webcam**: press and hold **Hold to Record** (mouse, touch, or Space/Enter) to capture about 10 frames per second.
   - **Upload**: pick multiple images, or drag and drop them onto a card.
   - Hover a thumbnail to delete it. Each class holds up to 500 samples.
3. **Train Model** unlocks once every class has at least 5 samples. A progress bar tracks feature extraction (per image), then training (per epoch, with live loss and accuracy).
4. **Preview** shows confidence bars per class, either live from the webcam or for an uploaded image.
5. **Export Model** opens the TensorFlow.js / TensorFlow Lite / Keras tabs.

**Advanced** (bottom bar) sets epochs (default 20), batch size (16) and learning rate (0.001), and shows a per-epoch metrics table. Previously exported TF.js models can be reopened from the start screen (**Open an exported TensorFlow.js model**).

## How it works

This is **transfer learning with a frozen feature extractor**, not training from scratch.

```
                     ┌──────────── frozen, never trained ────────────┐   ┌────── trained in the browser ──────┐
image ─► crop+resize ─► MobileNet V2 (α=0.35, ImageNet) ─► out_relu   ─► Dense(128, relu) ─► Dropout(0.2) ─► Dense(N, softmax)
         224×224, [-1,1]                                   7×7×1280      ▲
                                                           └─ GlobalAvgPool ─► 1280-d feature vector (cached per sample)
```

**Step A: load once.** `src/ml/mobilenet.ts` loads MobileNet V2 (α = 0.35, 224×224, ImageNet, no top) as a tfjs Layers model, taps layer `out_relu` and adds global average pooling. That 1280-d vector is the penultimate representation of the full ImageNet classifier. The model is marked non-trainable and is only ever used for `predict()`. A warm-up pass compiles the WebGL shaders up front.

**Step B: features, once per image.** On **Train**, `getFeatures()` in `src/ml/trainer.ts` pushes samples through MobileNet in batches of 16 inside `tf.tidy()`, downloads the `[16, 1280]` result and disposes it. Vectors are **cached by sample id**, so retraining after adding samples only processes the new ones, and MobileNet never runs per epoch.

**Step C: the head.** `buildHead()` builds `Dense(128, relu) → Dropout(0.2) → Dense(N, softmax)`, compiled with Adam (lr 0.001) and categorical cross-entropy. `trainClassifier()` fits it for 20 epochs, batch 16, shuffled, holding out 15% for validation, and streams `onEpochEnd` metrics into the UI.

> **Validation split.** tfjs's `validationSplit` takes the *last* 15% of the arrays before shuffling. Samples are stored class by class, so that would validate on, and barely train on, only the last class. The app therefore makes a **stratified** 85/15 split per class and passes it as `validationData`. Every class with 2 or more samples appears in both sets.

**Step D: inference.** `predictProbabilities()` in `src/ml/predictor.ts` runs crop/resize/normalize → MobileNet → head → softmax synchronously inside one `tf.tidy()`.

Other implementation details:

- **Memory.** Every tensor is created inside `tf.tidy()` or explicitly disposed. After training, the only new tensors are the head's 4 weights and Adam's 10 state tensors. The preview loop and `loadModel()` round-trips leak nothing, which was verified with `tf.memory().numTensors`.
- **Backends.** WebGL is tried first, with a CPU fallback and a warning toast if it fails.
- **Responsive UI.** Training yields to the browser between batches and epochs. `tf.nextFrame()` (requestAnimationFrame) doesn't fire in background tabs, so the app yields with a `MessageChannel` when the tab is hidden and caps each yield at 50 ms otherwise. Training keeps going if you switch tabs.
- **Locking.** The UI is locked while training (no recording, uploads, renames or class changes) and card webcams are closed. The preview pauses while the head is swapped.
- **Camera.** A single `getUserMedia` stream is shared by reference count between the class cards and the preview, and switched off 1.5 s after the last viewer closes. Webcam frames are mirrored for display *and* for capture/preview, so training and inference see the same orientation.

## Export formats

Only the trained **head** is exported. MobileNet is identical for every project, so it is reloaded at load time and the head is attached to its features.

### TensorFlow.js (`model-tfjs.zip`)

| file            | contents                                                         |
| --------------- | ---------------------------------------------------------------- |
| `model.json`    | head architecture (Layers format) + weights manifest             |
| `weights.bin`   | head weights (~650 KB: `head_dense` 1280×128, `head_output` 128×N) |
| `metadata.json` | labels and preprocessing contract (below)                        |

```json
{
  "labels": ["Cat", "Dog"],
  "imageSize": 224,
  "normalization": "[-1,1]",
  "baseModel": "mobilenet_v2",
  "featureLayer": "out_relu",
  "pooling": "global_average",
  "alpha": 0.35,
  "featureDim": 1280,
  "baseModelUrl": "/models/mobilenet/model.json",
  "createdAt": "2026-11-03T05:09:35.460Z",
  "generator": "tm-clone",
  "version": 1
}
```

Load it with the helper in `src/ml/predictor.ts`:

```ts
import { loadModel } from './ml/predictor';

const model = await loadModel({ modelUrl: '/my-model/model.json' }); // metadata.json next to it
// or: await loadModel(zipFile)                       (the downloaded .zip, from <input type=file>)
// or: await loadModel([modelJson, weightsBin, metadataJson])  (individual Files)

const predictions = await model.predict(imgOrVideoOrCanvas); // sorted, highest first
// [{ className: 'Cat', probability: 0.97 }, { className: 'Dog', probability: 0.03 }]
model.dispose();
```

`loadModel()` (1) loads MobileNet, (2) loads `model.json` + `weights.bin` + `metadata.json`, (3) checks that the head's input size matches MobileNet's feature size and that its output size matches the labels, and (4) returns `predict(image)`. Non-square inputs are center-cropped. Pass `{ mirror: true }` for selfie-webcam frames.

The Export dialog also shows a plain-tfjs snippet for use in a project that doesn't include this code.

### Keras `.h5` and TensorFlow Lite (`model-keras.zip` / `model-tflite.zip`)

The browser can't write HDF5, and there is no in-browser tfjs → TFLite converter (`@tensorflow/tfjs-converter` only converts *to* tfjs). Both tabs therefore download a small kit:

| file                                          | contents                                                         |
| --------------------------------------------- | ---------------------------------------------------------------- |
| `head_weights.json`                           | head layers in order; kernels flattened row-major with `kernelShape` |
| `metadata.json`, `labels.txt`                 | as above                                                         |
| `convert_to_keras.py` / `convert_to_tflite.py` | rebuilds the full model and saves it (also shown in the UI with a Copy button) |

```bash
pip install "tensorflow>=2.12" pillow
python convert_to_keras.py                    # -> keras_model.h5 + labels.txt
python convert_to_tflite.py                   # -> model.tflite (quantization picked in the UI)
python convert_to_keras.py --format both --quantize float16 --image test.jpg
```

The script builds `tf.keras.applications.MobileNetV2(alpha=0.35, include_top=False, weights="imagenet", pooling="avg")`, the same weights the browser uses, adds the head and loads `head_weights.json`. The result is one end-to-end model:

- input: `float32 [batch, 224, 224, 3]`, RGB scaled to [-1, 1] (`pixel / 127.5 - 1`)
- output: softmax probabilities in `labels.txt` order

TFLite quantization options are `none` (float32, ~2.2 MB), `float16` (~1.1 MB) and `dynamic` (int8 weights).

### Verified

These checks were run against real exports from a 2-class, 10-images-per-class project:

- Browser MobileNet features match Keras `MobileNetV2(alpha=0.35)` to 4 decimal places on the same image.
- `loadModel(zip)` reproduces the in-app probabilities exactly.
- `keras_model.h5` (saved, then reloaded), float32 `.tflite` and float16 `.tflite` all agree with the browser to within 1e-4 on unseen and blended images. The small difference comes from canvas vs. PIL resizing.

## Project structure

```
tm-clone/
├── public/models/mobilenet/      # MobileNet V2 0.35 (tfjs Layers): model.json + group1-shard1of1.bin (1.6 MB)
├── scripts/fetch-mobilenet.mjs   # re-downloads the weights above
└── src/
    ├── config.ts                 # limits & defaults (classes, samples, fps, training params)
    ├── types/index.ts            # Sample, ImageClass, TrainedModel, ModelMetadata, ...
    ├── store/
    │   ├── useProjectStore.ts    # Zustand: classes, samples, training progress, trained head
    │   └── useToastStore.ts      # error/success toasts
    ├── ml/
    │   ├── mobilenet.ts          # initBackend(), loadMobileNet(), extractFeatures()
    │   ├── trainer.ts            # getFeatures() + cache, buildHead(), stratifiedSplit(), trainClassifier()
    │   ├── predictor.ts          # predictProbabilities(), loadModel()
    │   └── exporter.ts           # exportTfjs(), exportKeras(), exportTflite(), pythonScript()
    ├── utils/
    │   ├── image.ts              # cropToSquare, normalize, tensorFromImage, fileToSample
    │   ├── camera.ts             # useCamera() hook, shared ref-counted stream, error mapping
    │   └── download.ts           # triggerDownload()
    ├── hooks/
    │   ├── useModel.ts           # loads MobileNet on mount
    │   └── useTraining.ts        # orchestrates extraction + training, error handling
    └── components/
        ├── Landing, ProjectWorkspace, ClassCard, SampleGallery, WebcamCapture,
        ├── TrainButton, TrainingProgress, TrainingSettings, PreviewPanel,
        ├── ExportModal, HowItWorks
        └── ui/                   # Button, Card, Modal, Tabs, ProgressBar, CodeBlock, Toaster, icons
```

## Error handling

| situation                                   | behaviour                                                               |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| camera permission denied / no camera / busy | toast + message in the webcam box with **Try again**                    |
| insecure origin or no `getUserMedia`        | explains that the camera needs localhost/HTTPS or a newer browser       |
| MobileNet fails to load                     | toast with the URL and the `npm run fetch-model` fix                    |
| WebGL unavailable                           | falls back to the CPU backend, with a warning toast                     |
| out of memory while training                | toast suggesting fewer samples; the previous model stays usable          |
| class below 5 samples, duplicate or blank names | Train disabled, with the reason shown under the button              |
| non-image or unreadable uploads             | skipped, with a count in a toast                                        |
| over 500 samples in a class                 | extra images are dropped and recording stops, with a toast              |

## Known limitations

- **Nothing is persisted.** Samples live in memory, so a reload loses them (the app warns before unload). There is no project save/load, only model export and re-import.
- **Keras/TFLite need Python** (see above). The TF.js export is the only fully in-browser format.
- **MobileNet α = 0.35** keeps the download to 1.6 MB and inference fast (it is also what Teachable Machine uses), but it is less accurate than α = 1.0 on fine-grained classes. Swapping it means changing `public/models/mobilenet`, `MOBILENET_ALPHA` in `src/ml/mobilenet.ts`, and the `alpha` recorded in exports.
- **Single-label classification only.** Every image belongs to exactly one class, and the softmax always sums to 1, so there is no "none of the above" unless you train a background class.
- **No data augmentation.** Vary angle, lighting and background yourself when recording.
- **Precision varies by GPU.** Some GPUs only support float16 WebGL textures. tfjs handles this, but reported losses bottom out at about 1e-4 instead of about 1e-7.
- Exports are **not** compatible with the original Teachable Machine's project files, and the TF.js export expects this project's `model.json` + `metadata.json` pairing.
- `@tensorflow/tfjs` bundles `tfjs-layers` and `tfjs-converter`, so they aren't listed as separate dependencies.

## Credits

MobileNet V2 ImageNet weights are from [Keras Applications](https://keras.io/api/applications/mobilenet/) (Apache 2.0), in the tfjs-converted form published for Teachable Machine. Runtime: [TensorFlow.js](https://www.tensorflow.org/js).
