import { useMemo, useState } from 'react';
import { exportKeras, exportTfjs, exportTflite, pythonFilename, pythonScript, type TfliteQuantization } from '../ml/exporter';
import { resolveLabels, useProjectStore } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';
import { Button } from './ui/Button';
import { CodeBlock } from './ui/CodeBlock';
import { DownloadIcon, InfoIcon, SpinnerIcon } from './ui/icons';
import { Modal } from './ui/Modal';
import { Tabs, type TabItem } from './ui/Tabs';

type ExportTab = 'tfjs' | 'tflite' | 'keras';

const TABS: TabItem<ExportTab>[] = [
  { id: 'tfjs', label: 'TensorFlow.js' },
  { id: 'tflite', label: 'TensorFlow Lite' },
  { id: 'keras', label: 'Keras' },
];

const APP_SNIPPET = `import { loadModel } from './ml/predictor';

// Unzip the export into public/my-model/ (model.json, weights.bin, metadata.json).
// MobileNet itself is loaded from public/models/mobilenet/.
const model = await loadModel({ modelUrl: '/my-model/model.json' });

// ...or pass the downloaded .zip straight from an <input type="file">:
// const model = await loadModel(zipFile);

const predictions = await model.predict(document.querySelector('img'));
console.log(predictions[0]); // { className: 'Cat', probability: 0.97 }`;

const PLAIN_TFJS_SNIPPET = `import * as tf from '@tensorflow/tfjs';

// 1. The frozen base: MobileNet V2 (alpha 0.35), cut at 'out_relu' + global average pooling.
//    Copy public/models/mobilenet/ from this project next to your app.
const base = await tf.loadLayersModel('/models/mobilenet/model.json');
const pooled = tf.layers.globalAveragePooling2d({}).apply(base.getLayer('out_relu').output);
const features = tf.model({ inputs: base.inputs, outputs: pooled });

// 2. The head you trained, and its labels.
const head = await tf.loadLayersModel('/my-model/model.json');
const { labels, imageSize } = await (await fetch('/my-model/metadata.json')).json();

// 3. A square-cropped RGB image, resized and scaled to [-1, 1], through both models.
const probs = tf.tidy(() => {
  const pixels = tf.browser.fromPixels(img);
  const x = tf.image.resizeBilinear(pixels, [imageSize, imageSize]).div(127.5).sub(1).expandDims(0);
  return head.predict(features.predict(x));
});
const scores = await probs.data();
probs.dispose();`;

export function ExportModal({ open, onClose }: { open: boolean; onClose(): void }) {
  const trained = useProjectStore((s) => s.trained);
  const classes = useProjectStore((s) => s.classes);
  const [tab, setTab] = useState<ExportTab>('tfjs');
  const [quantize, setQuantize] = useState<TfliteQuantization>('none');
  const [working, setWorking] = useState(false);
  const labels = useMemo(() => (trained ? resolveLabels(trained, classes) : []), [trained, classes]);

  const download = async () => {
    if (!trained) return;
    setWorking(true);
    try {
      if (tab === 'tfjs') await exportTfjs(trained, labels);
      else if (tab === 'tflite') await exportTflite(trained, labels, quantize);
      else await exportKeras(trained, labels);
      toast.success('Export downloaded.');
    } catch (err) {
      toast.error(`Export failed: ${err instanceof Error ? err.message : err}`);
    } finally {
      setWorking(false);
    }
  };

  const downloadButton = (label: string) => (
    <Button variant="primary" onClick={() => void download()} disabled={!trained || working}>
      {working ? <SpinnerIcon size={16} /> : <DownloadIcon size={16} />}
      {label}
    </Button>
  );

  return (
    <Modal open={open} onClose={onClose} title="Export your model">
      <Tabs<ExportTab> tabs={TABS} value={tab} onChange={setTab}>
        <div className="flex flex-col gap-5 p-5 text-sm text-slate-600">
          {tab === 'tfjs' && (
            <>
              <p>
                A <code className="rounded bg-slate-100 px-1">.zip</code> with the trained head and its labels. Use it in any web page with
                TensorFlow.js; MobileNet is loaded separately and the head is attached on top of its features.
              </p>
              <FileList
                files={[
                  ['model.json', 'architecture of the head + weight manifest'],
                  ['weights.bin', 'trained head weights (~650 KB)'],
                  ['metadata.json', `labels (${labels.join(', ')}), image size, normalization, base model`],
                ]}
              />
              <div>{downloadButton('Download TensorFlow.js model')}</div>
              <CodeBlock label="with this project's loadModel() helper" code={APP_SNIPPET} />
              <CodeBlock label="plain TensorFlow.js, any project" code={PLAIN_TFJS_SNIPPET} />
            </>
          )}

          {tab === 'tflite' && (
            <>
              <Note>
                Browsers can't produce <code>.tflite</code> files: tfjs-converter only converts <em>to</em> TensorFlow.js, and the TF Lite
                converter is Python-only. This kit contains your head weights and a script that rebuilds the full model (MobileNet + head) and
                converts it locally.
              </Note>
              <label className="flex max-w-xs flex-col gap-1 text-xs font-medium text-slate-600">
                Quantization
                <select
                  value={quantize}
                  onChange={(e) => setQuantize(e.target.value as TfliteQuantization)}
                  className="h-9 rounded-lg border-0 px-2.5 text-sm text-slate-900 ring-1 ring-inset ring-slate-300 focus:ring-2 focus:ring-brand-500"
                >
                  <option value="none">None (float32)</option>
                  <option value="float16">Float16 (about half the size)</option>
                  <option value="dynamic">Dynamic range int8 (smallest)</option>
                </select>
              </label>
              <div>{downloadButton('Download TF Lite conversion kit')}</div>
              <Steps target="tflite" />
              <CodeBlock label={pythonFilename('tflite')} code={pythonScript('tflite', quantize)} maxHeight={360} />
            </>
          )}

          {tab === 'keras' && (
            <>
              <Note>
                JavaScript can't write HDF5 (<code>.h5</code>) files, so this kit contains your head weights as JSON and a Python script that
                rebuilds the full model with <code>tf.keras</code> and saves <code>keras_model.h5</code>.
              </Note>
              <div>{downloadButton('Download Keras conversion kit')}</div>
              <Steps target="keras" />
              <CodeBlock label={pythonFilename('keras')} code={pythonScript('keras')} maxHeight={360} />
            </>
          )}
        </div>
      </Tabs>
    </Modal>
  );
}

function FileList({ files }: { files: [string, string][] }) {
  return (
    <ul className="divide-y divide-slate-100 rounded-xl ring-1 ring-slate-200">
      {files.map(([name, desc]) => (
        <li key={name} className="flex flex-col gap-0.5 px-3 py-2 sm:flex-row sm:gap-3">
          <code className="shrink-0 font-mono text-xs font-semibold text-slate-800 sm:w-28">{name}</code>
          <span className="text-xs text-slate-500">{desc}</span>
        </li>
      ))}
    </ul>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return (
    <p className="flex gap-2 rounded-xl bg-brand-50 px-3 py-2.5 text-[13px] leading-relaxed text-brand-900">
      <InfoIcon size={16} className="mt-0.5 shrink-0" />
      <span>{children}</span>
    </p>
  );
}

function Steps({ target }: { target: 'keras' | 'tflite' }) {
  const script = pythonFilename(target);
  const output = target === 'keras' ? 'keras_model.h5' : 'model.tflite';
  return (
    <CodeBlock
      label="run locally"
      code={`unzip model-${target}.zip -d my-model && cd my-model
pip install "tensorflow>=2.12" pillow
python ${script}                    # -> ${output} + labels.txt
python ${script} --image test.jpg   # optional: classify one image as a check`}
    />
  );
}
