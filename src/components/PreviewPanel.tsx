import { useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';
import { CLASS_COLORS } from '../config';
import { loadMobileNet, type FeatureExtractor } from '../ml/mobilenet';
import { predictProbabilities } from '../ml/predictor';
import { isBusy, resolveLabels, useProjectStore } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';
import { useCamera } from '../utils/camera';
import { loadImage } from '../utils/image';
import { Button } from './ui/Button';
import { Card, CardHeader } from './ui/Card';
import { AlertIcon, CameraIcon, DownloadIcon, ImageIcon, SpinnerIcon } from './ui/icons';

type InputMode = 'webcam' | 'file';

/**
 * Live classification with the trained head: webcam (continuous) or a single
 * uploaded image. Shows a confidence bar per class.
 */
export function PreviewPanel({ onExport }: { onExport(): void }) {
  const trained = useProjectStore((s) => s.trained);
  const classes = useProjectStore((s) => s.classes);
  const stale = useProjectStore((s) => s.stale);
  const busy = useProjectStore((s) => isBusy(s.progress));

  const [mode, setMode] = useState<InputMode>('webcam');
  const [webcamOn, setWebcamOn] = useState(false);
  const [probs, setProbs] = useState<Float32Array | null>(null);
  const [image, setImage] = useState<{ url: string; el: HTMLImageElement } | null>(null);
  const extractor = useRef<FeatureExtractor | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const camera = useCamera(mode === 'webcam' && webcamOn && trained !== null);

  const labels = useMemo(() => (trained ? resolveLabels(trained, classes) : []), [trained, classes]);
  const colorFor = (i: number) => {
    const idx = classes.findIndex((c) => c.id === trained?.classIds[i]);
    return CLASS_COLORS[(idx >= 0 ? idx : i) % CLASS_COLORS.length];
  };

  useEffect(() => {
    loadMobileNet().then((e) => (extractor.current = e), () => {});
  }, []);

  // New model (or none): old probabilities no longer apply.
  useEffect(() => setProbs(null), [trained]);

  // Webcam loop: classify the latest frame, then schedule the next one. Never
  // overlaps predictions, and pauses while training swaps the head.
  useEffect(() => {
    if (mode !== 'webcam' || camera.status !== 'ready') return;
    let stopped = false;
    let raf = 0;
    const tick = async () => {
      if (stopped) return;
      const { trained: model, progress } = useProjectStore.getState();
      const video = camera.videoRef.current;
      if (model && extractor.current && video && video.readyState >= 2 && !isBusy(progress)) {
        try {
          const p = await predictProbabilities(extractor.current, model.head, video, { mirror: true });
          if (!stopped) setProbs(p);
        } catch (err) {
          console.warn('[preview] prediction failed', err);
        }
      }
      if (!stopped) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
    };
  }, [mode, camera.status, camera.videoRef]);

  // File mode: classify the chosen image (again whenever the model changes).
  useEffect(() => {
    if (mode !== 'file' || !image || !trained || busy) return;
    let cancelled = false;
    loadMobileNet()
      .then((ext) => predictProbabilities(ext, trained.head, image.el))
      .then((p) => !cancelled && setProbs(p))
      .catch((err) => toast.error(`Prediction failed: ${err instanceof Error ? err.message : err}`));
    return () => {
      cancelled = true;
    };
  }, [mode, image, trained, busy]);

  useEffect(() => {
    if (!image) return;
    return () => URL.revokeObjectURL(image.url);
  }, [image]);

  const onFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      toast.warning('Please choose an image file.');
      return;
    }
    const url = URL.createObjectURL(file);
    try {
      const el = await loadImage(url);
      setProbs(null); // don't show the previous image's result while this one is classified
      setImage({ url, el });
    } catch {
      URL.revokeObjectURL(url);
      toast.error('That image could not be read.');
    }
  };

  const switchMode = (m: InputMode) => {
    setMode(m);
    setProbs(null);
  };

  if (!trained) {
    return (
      <Card>
        <CardHeader title="Preview" />
        <div className="flex flex-col items-center gap-2 px-6 py-12 text-center text-sm text-slate-500">
          <ImageIcon size={28} className="text-slate-300" />
          You must train a model before you can preview it here.
        </div>
      </Card>
    );
  }

  const top = probs ? Array.from(probs).reduce((best, p, i, arr) => (p > arr[best] ? i : best), 0) : -1;

  return (
    <Card>
      <CardHeader
        title="Preview"
        actions={
          <Button variant="soft" size="sm" onClick={onExport} disabled={busy}>
            <DownloadIcon size={16} /> Export Model
          </Button>
        }
      />
      <div className="flex flex-col gap-4 p-4">
        {stale && (
          <p className="flex items-start gap-2 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">
            <AlertIcon size={16} className="shrink-0" /> Your samples changed after training. This preview uses the previous model.
          </p>
        )}

        <div className="grid grid-cols-2 gap-1 rounded-lg bg-slate-100 p-1 text-sm font-medium" role="radiogroup" aria-label="Input">
          {(['webcam', 'file'] as const).map((m) => (
            <button
              key={m}
              type="button"
              role="radio"
              aria-checked={mode === m}
              onClick={() => switchMode(m)}
              className={`flex items-center justify-center gap-1.5 rounded-md py-1.5 ${mode === m ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
            >
              {m === 'webcam' ? <CameraIcon size={16} /> : <ImageIcon size={16} />}
              {m === 'webcam' ? 'Webcam' : 'File'}
            </button>
          ))}
        </div>

        {mode === 'webcam' ? (
          <div className="relative aspect-square overflow-hidden rounded-xl bg-slate-900">
            {webcamOn && <video ref={camera.videoRef} muted playsInline autoPlay className="h-full w-full -scale-x-100 object-cover" />}
            {(!webcamOn || camera.error || camera.status === 'starting') && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center text-sm text-slate-300">
                {camera.status === 'starting' ? (
                  <span className="flex items-center gap-2">
                    <SpinnerIcon /> Starting camera…
                  </span>
                ) : camera.error ? (
                  <>
                    <p>{camera.error}</p>
                    <div className="flex gap-2">
                      {camera.status !== 'unsupported' && (
                        <Button size="sm" onClick={camera.retry}>
                          Try again
                        </Button>
                      )}
                      <Button size="sm" variant="ghost" className="text-slate-300 hover:bg-slate-800" onClick={() => switchMode('file')}>
                        Use a file
                      </Button>
                    </div>
                  </>
                ) : (
                  <Button variant="primary" onClick={() => setWebcamOn(true)}>
                    <CameraIcon size={16} /> Turn on webcam
                  </Button>
                )}
              </div>
            )}
            {webcamOn && camera.status === 'ready' && (
              <button
                type="button"
                onClick={() => {
                  setWebcamOn(false);
                  setProbs(null);
                }}
                className="absolute right-2 top-2 rounded-full bg-slate-900/60 px-2.5 py-1 text-xs font-medium text-white hover:bg-slate-900"
              >
                Turn off
              </button>
            )}
          </div>
        ) : (
          <button
            type="button"
            onClick={() => fileInput.current?.click()}
            className="group relative flex aspect-square items-center justify-center overflow-hidden rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 text-sm text-slate-500 hover:border-brand-400 hover:text-brand-700"
          >
            {image ? (
              <>
                <img src={image.url} alt="Image to classify" className="h-full w-full object-cover" />
                <span className="absolute bottom-2 right-2 rounded-full bg-slate-900/70 px-2.5 py-1 text-xs font-medium text-white opacity-0 group-hover:opacity-100">
                  Choose another
                </span>
              </>
            ) : (
              <span className="flex flex-col items-center gap-2">
                <ImageIcon size={28} /> Choose an image to classify
              </span>
            )}
          </button>
        )}
        <input ref={fileInput} type="file" accept="image/*" hidden onChange={onFile} />

        <div>
          <div className="mb-2 flex items-baseline justify-between">
            <h3 className="text-sm font-semibold text-slate-800">Output</h3>
            {top >= 0 && probs && (
              <span className="truncate pl-3 text-sm text-slate-500" aria-live="polite">
                <span className="font-semibold text-slate-900">{labels[top]}</span> · {(probs[top] * 100).toFixed(0)}%
              </span>
            )}
          </div>
          <ul className="flex flex-col gap-2">
            {labels.map((label, i) => {
              const p = probs ? probs[i] : 0;
              return (
                <li key={trained.classIds[i]} className="grid grid-cols-[minmax(0,7rem)_1fr] items-center gap-3 text-sm">
                  <span className={`truncate ${i === top ? 'font-semibold text-slate-900' : 'text-slate-600'}`} title={label}>
                    {label}
                  </span>
                  <div className="relative h-6 overflow-hidden rounded-md bg-slate-100">
                    <div
                      className="h-full rounded-md transition-[width] duration-150 ease-out"
                      style={{ width: `${p * 100}%`, background: colorFor(i), opacity: i === top ? 1 : 0.55 }}
                    />
                    <span className="absolute inset-y-0 right-2 flex items-center font-mono text-xs font-medium text-slate-700">
                      {probs ? `${(p * 100).toFixed(0)}%` : '–'}
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    </Card>
  );
}
