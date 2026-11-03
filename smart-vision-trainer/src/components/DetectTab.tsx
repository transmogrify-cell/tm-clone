import { useCallback, useEffect, useRef, useState } from 'react';
import { DETECT_FPS, FACE_COLOR, HAND_COLOR } from '../config';
import { runDetectors } from '../hooks/useDetectors';
import { useFrameLoop, useFrameSource } from '../hooks/useInference';
import { useTraining } from '../hooks/useTraining';
import { predictDualPath } from '../ml/predictor';
import {
  classColor,
  isBusy,
  outputColor,
  resolveLabels,
  sampleQuality,
  useProjectStore,
  type ModelKey,
  type Toggles,
} from '../store/useProjectStore';
import { clearOverlay, cocoColor, drawDetections } from '../utils/canvasOverlay';
import type { DetectionFrame, DualPathPrediction } from '../types';
import { Chip, SourceView } from './SourceView';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { AlertIcon, PlusIcon, SpinnerIcon } from './ui/icons';

const fmt = (v: number) => v.toFixed(2);

/**
 * Mode A, live: every frame (throttled to 10 fps) runs the enabled detectors
 * in parallel and draws their boxes on the canvas overlay. If a custom model
 * is trained, Mode B classifies the same frame and its answer shows alongside.
 */
export function DetectTab() {
  const src = useFrameSource(true);
  const overlay = useRef<HTMLCanvasElement>(null);
  const [frame, setFrame] = useState<DetectionFrame | null>(null);
  const [custom, setCustom] = useState<DualPathPrediction | null>(null);
  const enabled = useProjectStore((s) => s.enabled);
  const models = useProjectStore((s) => s.models);
  const trained = useProjectStore((s) => s.trained);
  const threshold = useProjectStore((s) => s.threshold);
  const { grab, mirror } = src;
  // Each run takes a ticket; a run that finishes after a newer one started
  // (a new photo, a toggled detector) is dropped instead of overwriting it.
  const latestRun = useRef(0);

  const step = useCallback(async () => {
    const run = ++latestRun.current;
    const s = useProjectStore.getState();
    const source = grab();
    if (!source) return;
    const result = await runDetectors(source.canvas, s.enabled, s.models, source);
    if (run !== latestRun.current) return;
    if (overlay.current) drawDetections(overlay.current, result, { mirror });

    // Mode B on the same pixels (reusing BlazeFace's faces when it ran).
    let customResult: DualPathPrediction | null = null;
    if (s.enabled.custom && s.trained && !isBusy(s.progress) && s.models.mobilenet.status === 'ready') {
      customResult = await predictDualPath(source.canvas, {
        head: s.trained.head,
        labels: resolveLabels(s.trained, s.classes),
        kinds: s.trained.kinds,
        threshold: s.threshold,
        faces: s.enabled.face && s.models.face.status === 'ready' ? result.faces : undefined,
      });
      if (run !== latestRun.current) return;
    }
    setFrame(result);
    setCustom(customResult);
  }, [grab, mirror]);

  const live = src.mode === 'webcam' && src.camera.status === 'ready';
  const fps = useFrameLoop(live, step, DETECT_FPS);

  // Still photo: run once, and again whenever a setting that changes the answer changes.
  useEffect(() => {
    if (src.mode !== 'image') return;
    if (!src.image) {
      latestRun.current++;
      clearOverlay(overlay.current);
      setFrame(null);
      return;
    }
    void step().catch((err) => console.warn(err));
  }, [src.mode, src.image, step, enabled, models, trained, threshold]);

  useEffect(() => {
    if (!live && src.mode === 'webcam') clearOverlay(overlay.current);
  }, [live, src.mode]);

  const chips = frame && (
    <>
      {live && <Chip>{fps} fps</Chip>}
      <Chip>{Math.round(frame.ms)} ms</Chip>
      {custom?.prediction && (
        <Chip tone={custom.prediction.isUnknown ? 'dark' : 'brand'}>
          Custom: {custom.prediction.label} {fmt(custom.prediction.confidence)}
        </Chip>
      )}
    </>
  );

  return (
    <main className="mx-auto grid max-w-7xl gap-4 px-4 py-6 lg:grid-cols-[minmax(0,1fr)_380px]">
      <section className="min-w-0">
        <SourceView src={src} overlayRef={overlay} chips={chips} />
      </section>
      <aside className="flex min-w-0 flex-col gap-4">
        <ModeAPanel frame={frame} />
        <ModeBPanel custom={custom} />
      </aside>
    </main>
  );
}

const DETECTORS: { key: keyof Toggles & ModelKey; label: string; hint: string; color: string }[] = [
  { key: 'coco', label: 'COCO-SSD', hint: '80 everyday objects', color: '#ef4444' },
  { key: 'face', label: 'Faces', hint: 'BlazeFace', color: FACE_COLOR },
  { key: 'hands', label: 'Hands', hint: 'MediaPipe, 21 keypoints', color: HAND_COLOR },
];

function ModeAPanel({ frame }: { frame: DetectionFrame | null }) {
  const enabled = useProjectStore((s) => s.enabled);
  const models = useProjectStore((s) => s.models);
  const setEnabled = useProjectStore((s) => s.setEnabled);

  const items = frame
    ? [
        ...frame.coco.map((d) => ({ label: d.label, score: d.score, color: cocoColor(d.label) })),
        ...frame.faces.map(() => ({ label: 'face', score: null, color: FACE_COLOR })),
        ...frame.hands.map((h) => ({ label: `${h.handedness.toLowerCase()} hand`, score: h.score, color: HAND_COLOR })),
      ]
    : [];

  return (
    <Card>
      <div className="border-b border-slate-100 px-4 py-3">
        <h2 className="font-semibold text-slate-900">Mode A · Pretrained detectors</h2>
        <p className="text-xs text-slate-500">Zero-shot: works now, no training.</p>
      </div>
      <div className="flex flex-col gap-1 px-4 py-3">
        {DETECTORS.map((d) => {
          const st = models[d.key];
          return (
            <label key={d.key} className="flex cursor-pointer items-center gap-2.5 rounded-lg px-1 py-1 hover:bg-slate-50">
              <input
                type="checkbox"
                checked={enabled[d.key]}
                onChange={(e) => setEnabled(d.key, e.target.checked)}
                className="h-4 w-4 rounded accent-brand-600"
                data-testid={`toggle-${d.key}`}
              />
              <span className="h-2.5 w-2.5 rounded-sm" style={{ background: d.color }} aria-hidden="true" />
              <span className="text-sm font-medium text-slate-800">{d.label}</span>
              <span className="truncate text-xs text-slate-400">{d.hint}</span>
              <span className="ml-auto shrink-0">
                {st.status === 'loading' && <SpinnerIcon size={14} className="text-slate-400" />}
                {st.status === 'error' && (
                  <span title={st.error}>
                    <AlertIcon size={14} className="text-red-600" />
                  </span>
                )}
              </span>
            </label>
          );
        })}
      </div>
      <div className="border-t border-slate-100 px-4 py-3">
        <p className="mb-2 text-sm font-semibold text-slate-800" data-testid="detected-count">
          Detected ({items.length})
        </p>
        {items.length === 0 ? (
          <p className="text-sm text-slate-400">{frame ? 'Nothing above the 0.50 score threshold.' : 'Waiting for the camera or a photo…'}</p>
        ) : (
          <ul className="flex max-h-64 flex-col gap-1 overflow-y-auto" data-testid="detected-list">
            {items.map((it, i) => (
              <li key={i} className="flex items-center gap-2 text-sm">
                <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: it.color }} aria-hidden="true" />
                <span className="text-slate-800">{it.label}</span>
                {it.score !== null && <span className="ml-auto font-mono tabular-nums text-slate-500">{fmt(it.score)}</span>}
              </li>
            ))}
          </ul>
        )}
        <p className="mt-3 text-xs leading-relaxed text-slate-400">
          COCO-SSD knows only the 80 COCO classes. For your own objects or people, train them in Mode B.
        </p>
      </div>
    </Card>
  );
}

function ModeBPanel({ custom }: { custom: DualPathPrediction | null }) {
  const classes = useProjectStore((s) => s.classes);
  const trained = useProjectStore((s) => s.trained);
  const stale = useProjectStore((s) => s.stale);
  const enabled = useProjectStore((s) => s.enabled.custom);
  const { setEnabled, setTab, setTrainView } = useProjectStore.getState();
  const { train, busy, readiness } = useTraining();
  const labels = trained ? resolveLabels(trained, classes) : [];

  return (
    <Card>
      <div className="flex items-center justify-between gap-2 border-b border-slate-100 px-4 py-3">
        <div>
          <h2 className="font-semibold text-slate-900">Mode B · Custom classes</h2>
          <p className="text-xs text-slate-500">Train your own labels on top.</p>
        </div>
        <Button size="sm" variant="soft" onClick={() => setTab('train')}>
          <PlusIcon size={16} /> Add class
        </Button>
      </div>
      <div className="grid grid-cols-2 gap-2 px-4 py-3">
        {classes.map((c) => (
          <button
            key={c.id}
            type="button"
            onClick={() => setTab('train')}
            className="flex flex-col items-start rounded-xl bg-slate-50 px-3 py-2 text-left ring-1 ring-slate-200 hover:bg-slate-100"
          >
            <span className="flex w-full items-center gap-1.5">
              <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: classColor(classes, c.id) }} />
              <span className="truncate text-sm font-semibold text-slate-900">{c.name}</span>
            </span>
            <span className="text-xs capitalize text-slate-500">{c.kind}</span>
            <span className={`text-xs tabular-nums ${sampleQuality(c).quality === 'too-few' ? 'text-red-600' : 'text-slate-500'}`}>
              {c.samples.length} imgs
            </span>
          </button>
        ))}
      </div>
      <div className="flex flex-col gap-2 border-t border-slate-100 px-4 py-3">
        <Button variant="primary" onClick={train} disabled={busy || !readiness.ok} title={readiness.reason}>
          {busy ? <SpinnerIcon /> : null} {busy ? 'Training…' : trained ? 'Retrain model' : 'Train model'}
        </Button>
        {!readiness.ok && !busy && <p className="text-xs text-slate-500">{readiness.reason}</p>}
        <Button
          disabled={!trained?.evaluation}
          onClick={() => {
            setTab('train');
            setTrainView('report');
          }}
        >
          Accuracy report
        </Button>
      </div>
      {trained && (
        <div className="border-t border-slate-100 px-4 py-3">
          <label className="mb-2 flex cursor-pointer items-center gap-2 text-sm font-medium text-slate-800">
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => setEnabled('custom', e.target.checked)}
              className="h-4 w-4 rounded accent-brand-600"
            />
            Run custom model live
            {stale && <span className="ml-auto text-xs font-normal text-amber-700">data changed, retrain</span>}
          </label>
          {enabled && custom?.prediction && (
            <div className="flex flex-col gap-1">
              <p className="text-sm text-slate-600">
                <b className="text-slate-900">{custom.prediction.label}</b> {fmt(custom.prediction.confidence)} via the{' '}
                {custom.chosen === 'face' ? 'face crop' : 'full frame'}
              </p>
              {labels.map((l, i) => {
                const p = (custom.chosen === 'face' ? custom.face : custom.full)?.probs[i] ?? 0;
                return (
                  <div key={l} className="flex items-center gap-2 text-xs">
                    <span className="w-20 truncate text-slate-600">{l}</span>
                    <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100">
                      <span className="block h-full rounded-full" style={{ width: `${p * 100}%`, background: outputColor(trained, classes, i) }} />
                    </span>
                    <span className="w-8 text-right font-mono tabular-nums text-slate-500">{fmt(p)}</span>
                  </div>
                );
              })}
            </div>
          )}
          {enabled && custom && !custom.prediction && <p className="text-xs text-slate-500">No face in view (this model only knows people).</p>}
        </div>
      )}
    </Card>
  );
}
