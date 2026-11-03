import { useMemo, useState } from 'react';
import { buildMetadata, exportTfjsFile, exportTfjsZip, jsUsageSnippet, kerasPythonSnippet, type TfjsFiles } from '../ml/exporter';
import { resolveLabels, useProjectStore } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { CodeBlock } from './ui/CodeBlock';
import { DownloadIcon, SpinnerIcon } from './ui/icons';
import { Modal } from './ui/Modal';

/** Export the custom head (TF.js files, Keras via a Python snippet). Mode A models need no export. */
export function ExportTab() {
  const trained = useProjectStore((s) => s.trained);
  const classes = useProjectStore((s) => s.classes);
  const threshold = useProjectStore((s) => s.threshold);
  const [busy, setBusy] = useState<string | null>(null);
  const [kerasOpen, setKerasOpen] = useState(false);
  const labels = useMemo(() => (trained ? resolveLabels(trained, classes) : []), [trained, classes]);
  const metadata = useMemo(() => (trained ? buildMetadata(trained, labels, threshold) : null), [trained, labels, threshold]);

  if (!trained || !metadata) {
    return (
      <main className="mx-auto max-w-3xl px-4 py-10">
        <Card className="flex flex-col items-start gap-3 p-6 text-sm text-slate-600">
          <p className="font-semibold text-slate-900">Nothing to export yet.</p>
          <p>Train a custom model first. The pretrained detectors (COCO-SSD, BlazeFace, MediaPipe Hands) are used as-is; see the README for standalone snippets.</p>
          <Button variant="primary" onClick={() => useProjectStore.getState().setTab('train')}>
            Go to Train
          </Button>
        </Card>
      </main>
    );
  }

  const run = async (key: string, job: () => Promise<void>) => {
    setBusy(key);
    try {
      await job();
    } catch (err) {
      toast.error(`Export failed: ${err instanceof Error ? err.message : err}`);
    } finally {
      setBusy(null);
    }
  };
  const file = (name: keyof TfjsFiles) => (
    <Button size="sm" disabled={busy !== null} onClick={() => run(name, () => exportTfjsFile(trained, labels, threshold, name))}>
      {busy === name ? <SpinnerIcon size={14} /> : <DownloadIcon size={14} />} {name}
    </Button>
  );

  return (
    <main className="mx-auto flex max-w-5xl flex-col gap-4 px-4 py-6">
      <Card className="flex flex-col gap-4 p-5">
        <div>
          <h1 className="text-lg font-semibold text-slate-900">TensorFlow.js</h1>
          <p className="text-sm text-slate-600">
            The trained head ({labels.length} classes: {labels.join(', ')}) plus metadata. MobileNet V2 and BlazeFace are not included; load them
            with the same packages wherever you use the model.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant="primary"
            disabled={busy !== null}
            onClick={() => run('zip', () => exportTfjsZip(trained, labels, threshold))}
            data-testid="export-zip"
          >
            {busy === 'zip' ? <SpinnerIcon /> : <DownloadIcon />} Download .zip
          </Button>
          <span className="text-xs text-slate-400">or</span>
          {file('model.json')}
          {file('weights.bin')}
          {file('metadata.json')}
        </div>
        <CodeBlock code={JSON.stringify(metadata, null, 2)} label="metadata.json" maxHeight={260} />
      </Card>

      <Card className="flex flex-col gap-3 p-5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Keras (.h5)</h2>
            <p className="text-sm text-slate-600">Browsers can't write HDF5. A short Python script rebuilds the head from the TF.js files.</p>
          </div>
          <Button onClick={() => setKerasOpen(true)}>Show Python snippet</Button>
        </div>
      </Card>

      <Card className="flex flex-col gap-3 p-5">
        <h2 className="text-lg font-semibold text-slate-900">Use it in a web page</h2>
        <CodeBlock code={jsUsageSnippet()} label="classify.js" maxHeight={360} />
      </Card>

      <Modal open={kerasOpen} onClose={() => setKerasOpen(false)} title="Keras .h5 export">
        <div className="flex flex-col gap-3 p-5 text-sm text-slate-600">
          <p>
            Unzip the TF.js export, save this next to <code>model.json</code> as <code>convert_to_keras.py</code> (it is also inside the .zip), and
            run it. Output: <code>custom_head.h5</code> + <code>labels.txt</code>.
          </p>
          <CodeBlock code={kerasPythonSnippet()} label="convert_to_keras.py" maxHeight={460} />
        </div>
      </Modal>
    </main>
  );
}
