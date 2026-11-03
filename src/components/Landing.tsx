import { useRef, useState, type ChangeEvent } from 'react';
import { loadModel } from '../ml/predictor';
import { useProjectStore } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';
import { HowItWorks } from './HowItWorks';
import { FolderIcon, ImageIcon, MicIcon, SpinnerIcon } from './ui/icons';
import { ProgressBar } from './ui/ProgressBar';

export function Landing() {
  const goTo = useProjectStore((s) => s.goTo);
  const loadTrained = useProjectStore((s) => s.loadTrained);
  const model = useProjectStore((s) => s.model);
  const hasWork = useProjectStore((s) => s.classes.some((c) => c.samples.length > 0) || s.trained !== null);
  const [opening, setOpening] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  /** Loads an exported TensorFlow.js model (zip, or model.json + weights.bin + metadata.json) for preview. */
  const openModel = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    e.target.value = '';
    if (files.length === 0) return;
    setOpening(true);
    try {
      const source = files.length === 1 && /\.zip$/i.test(files[0].name) ? files[0] : files;
      const loaded = await loadModel(source);
      loadTrained({
        head: loaded.head,
        labels: loaded.labels,
        featureDim: loaded.metadata.featureDim,
        trainedAt: Date.parse(loaded.metadata.createdAt) || Date.now(),
        sampleCount: 0,
      });
      toast.success(`Opened a model with ${loaded.labels.length} classes. Add samples to retrain it.`);
    } catch (err) {
      toast.error(`Could not open the model: ${err instanceof Error ? err.message : err}`);
    } finally {
      setOpening(false);
    }
  };

  return (
    <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-4 py-10 sm:py-16">
      <header className="mb-10 flex items-center gap-2.5">
        <Logo />
        <span className="text-lg font-semibold text-slate-900">Image Trainer</span>
      </header>

      <h1 className="max-w-2xl text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl">Teach your computer to recognise images.</h1>
      <p className="mt-4 max-w-2xl text-lg text-slate-600">
        Collect examples with your webcam or from files, train a classifier in seconds, and export it, all inside this browser tab. Your
        images never leave your device.
      </p>

      <div className="mt-10 grid gap-4 sm:grid-cols-2">
        <button
          type="button"
          onClick={() => goTo('workspace')}
          className="group flex flex-col items-start gap-4 rounded-2xl bg-white p-6 text-left shadow-sm ring-1 ring-slate-200 transition hover:-translate-y-0.5 hover:shadow-md hover:ring-brand-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-500"
        >
          <span className="flex h-12 w-12 items-center justify-center rounded-xl bg-brand-600 text-white">
            <ImageIcon size={24} />
          </span>
          <span>
            <span className="block text-xl font-semibold text-slate-900">Image Project</span>
            <span className="mt-1 block text-slate-600">Teach based on images, from files or your webcam.</span>
          </span>
          <span className="mt-auto text-sm font-semibold text-brand-700 group-hover:underline">{hasWork ? 'Continue project →' : 'Get started →'}</span>
        </button>

        <div aria-disabled="true" className="flex cursor-not-allowed flex-col items-start gap-4 rounded-2xl bg-slate-50 p-6 ring-1 ring-slate-200">
          <span className="flex h-12 w-12 items-center justify-center rounded-xl bg-slate-200 text-slate-400">
            <MicIcon size={24} />
          </span>
          <span>
            <span className="flex items-center gap-2 text-xl font-semibold text-slate-400">
              Audio Project
              <span className="rounded-full bg-slate-200 px-2 py-0.5 text-xs font-medium text-slate-500">Coming soon</span>
            </span>
            <span className="mt-1 block text-slate-400">Teach based on one-second sound clips.</span>
          </span>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
        <button
          type="button"
          onClick={() => fileInput.current?.click()}
          disabled={opening || model.status !== 'ready'}
          className="inline-flex items-center gap-2 font-medium text-slate-700 hover:text-brand-700 disabled:text-slate-400"
        >
          {opening ? <SpinnerIcon size={16} /> : <FolderIcon size={16} />}
          Open an exported TensorFlow.js model
        </button>
        <input ref={fileInput} type="file" accept=".zip,.json,.bin" multiple hidden onChange={openModel} />
        <ModelStatusLine />
      </div>

      <div className="mt-10">
        <HowItWorks />
      </div>

      <footer className="mt-auto pt-12 text-xs text-slate-400">Runs locally with TensorFlow.js · MobileNet V2 (ImageNet) · No data leaves your browser</footer>
    </div>
  );
}

export function ModelStatusLine() {
  const model = useProjectStore((s) => s.model);
  if (model.status === 'loading') {
    return (
      <span className="flex items-center gap-2 text-slate-500">
        <SpinnerIcon size={14} /> Loading MobileNet…
        <ProgressBar value={model.progress} className="w-24" label="MobileNet download" />
      </span>
    );
  }
  if (model.status === 'error') return <span className="text-red-600">MobileNet failed to load: {model.error}</span>;
  if (model.status === 'ready') {
    return (
      <span className="flex items-center gap-1.5 text-slate-500">
        <span className="h-2 w-2 rounded-full bg-emerald-500" /> MobileNet V2 ready · {model.backend === 'webgl' ? 'GPU (WebGL)' : 'CPU'}
      </span>
    );
  }
  return null;
}

export function Logo() {
  return (
    <svg width="32" height="32" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="#2547eb" />
      <path d="M9 22l5-12 4 8 2-4 3 8" stroke="white" strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
