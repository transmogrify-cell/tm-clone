import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FACE_COLOR, THRESHOLD_MAX, THRESHOLD_MIN, UNKNOWN_COLOR } from '../config';
import { useFrameLoop, useFrameSource } from '../hooks/useInference';
import { detectCoco } from '../ml/cocoDetector';
import { detectFaces } from '../ml/faceDetector';
import { inferencePaths, predictDualPath, predictHybrid } from '../ml/predictor';
import { isBusy, outputColor, resolveLabels, useProjectStore } from '../store/useProjectStore';
import { clearOverlay, cocoColor, drawBoxes, prepareCanvas, type OverlayBox } from '../utils/canvasOverlay';
import type { DualPathPrediction, HybridBox, PathResult, TrainedModel } from '../types';
import { Chip, SourceView } from './SourceView';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { AlertIcon, InfoIcon } from './ui/icons';

const fmt = (v: number) => v.toFixed(2);

type Result = { kind: 'dual'; value: DualPathPrediction } | { kind: 'hybrid'; value: HybridBox[] };

/**
 * Mode B inference. Default: dual-path classification of the whole frame
 * (face crop vs full frame, higher confidence wins, threshold -> "Unknown").
 * "Detect + Classify": Mode A finds the boxes, Mode B names each one.
 */
export function PreviewTab() {
  const trained = useProjectStore((s) => s.trained);
  const classes = useProjectStore((s) => s.classes);
  const threshold = useProjectStore((s) => s.threshold);
  const hybrid = useProjectStore((s) => s.hybrid);
  const models = useProjectStore((s) => s.models);
  const src = useFrameSource(!!trained);
  const overlay = useRef<HTMLCanvasElement>(null);
  const [result, setResult] = useState<Result | null>(null);
  const labels = useMemo(() => (trained ? resolveLabels(trained, classes) : []), [trained, classes]);
  const { grab, mirror } = src;
  // Same ticketing as the Detect tab: a stale run never overwrites a newer one.
  const latestRun = useRef(0);

  const step = useCallback(async () => {
    const run = ++latestRun.current;
    const s = useProjectStore.getState();
    const source = grab();
    if (!source || !s.trained || isBusy(s.progress) || s.models.mobilenet.status !== 'ready') return;
    const opts = { head: s.trained.head, labels: resolveLabels(s.trained, s.classes), kinds: s.trained.kinds, threshold: s.threshold };
    const colorOf = (i: number) => outputColor(s.trained!, s.classes, i);
    const canvas = overlay.current;

    if (s.hybrid) {
      const [coco, faces] = await Promise.all([
        s.models.coco.status === 'ready' ? detectCoco(source.canvas) : Promise.resolve([]),
        s.models.face.status === 'ready' ? detectFaces(source.canvas) : Promise.resolve([]),
      ]);
      const boxes = await predictHybrid(source.canvas, { ...opts, coco, faces });
      if (run !== latestRun.current) return;
      if (canvas) {
        const ctx = prepareCanvas(canvas, source.width, source.height);
        drawBoxes(ctx, boxes.map((b) => hybridOverlay(b, colorOf)), { mirror });
      }
      setResult({ kind: 'hybrid', value: boxes });
      return;
    }

    const r = await predictDualPath(source.canvas, opts);
    if (run !== latestRun.current) return;
    if (canvas) {
      const ctx = prepareCanvas(canvas, source.width, source.height);
      const p = r.prediction;
      if (r.faceBox) {
        const onFace = r.chosen === 'face' && p;
        drawBoxes(
          ctx,
          [
            {
              box: r.faceBox,
              color: onFace ? (p.isUnknown ? UNKNOWN_COLOR : colorOf(p.topIndex)) : FACE_COLOR,
              label: onFace ? `${p.label} ${fmt(p.confidence)}` : 'face',
              dashed: !onFace || p.isUnknown,
            },
          ],
          { mirror },
        );
      }
    }
    setResult({ kind: 'dual', value: r });
  }, [grab, mirror]);

  const live = !!trained && src.mode === 'webcam' && src.camera.status === 'ready';
  const fps = useFrameLoop(live, step);

  useEffect(() => {
    if (src.mode !== 'image') return;
    if (!src.image) {
      latestRun.current++;
      clearOverlay(overlay.current);
      setResult(null);
      return;
    }
    void step().catch((err) => console.warn(err));
  }, [src.mode, src.image, step, trained, threshold, hybrid, models, labels]);

  useEffect(() => {
    latestRun.current++;
    setResult(null);
    clearOverlay(overlay.current);
  }, [hybrid]);

  if (!trained) {
    return (
      <main className="mx-auto max-w-3xl px-4 py-10">
        <Card className="flex flex-col items-start gap-3 p-6 text-sm text-slate-600">
          <p className="font-semibold text-slate-900">No custom model yet.</p>
          <p>Train at least two classes in the Train tab, then come back to test them live or on photos.</p>
          <Button variant="primary" onClick={() => useProjectStore.getState().setTab('train')}>
            Go to Train
          </Button>
        </Card>
      </main>
    );
  }

  const dual = result?.kind === 'dual' ? result.value : null;
  const chips = (
    <>
      {live && <Chip>{fps} fps</Chip>}
      {dual?.prediction && (
        <Chip tone={dual.prediction.isUnknown ? 'dark' : 'brand'}>
          {dual.prediction.label} {fmt(dual.prediction.confidence)}
        </Chip>
      )}
    </>
  );

  return (
    <main className="mx-auto grid max-w-7xl gap-4 px-4 py-6 lg:grid-cols-[minmax(0,1fr)_380px]">
      <section className="min-w-0">
        <SourceView src={src} overlayRef={overlay} chips={chips} emptyHint="Upload a photo to classify it." />
      </section>
      <aside className="flex min-w-0 flex-col gap-4">
        <Controls trained={trained} />
        {result?.kind === 'dual' && <DualPathPanel r={result.value} labels={labels} trained={trained} />}
        {result?.kind === 'hybrid' && <HybridPanel boxes={result.value} />}
        <Card className="flex gap-2 p-4 text-xs leading-relaxed text-slate-500">
          <InfoIcon size={16} className="mt-0.5 shrink-0" />
          <p>
            Mode B is <b>classification</b>: it names the whole frame (or a crop) but cannot localize a custom class by itself. "Detect +
            Classify" borrows Mode A's boxes, so custom labels only appear on things COCO-SSD or BlazeFace can find. Not for security or
            authentication.
          </p>
        </Card>
      </aside>
    </main>
  );
}

function hybridOverlay(b: HybridBox, colorOf: (i: number) => string): OverlayBox {
  const p = b.prediction;
  if (p && !p.isUnknown) return { box: b.box, color: colorOf(p.topIndex), label: `${p.label} ${fmt(p.confidence)}` };
  const base = b.detector === 'face' ? FACE_COLOR : cocoColor(b.baseLabel);
  const suffix = p ? ' · unknown' : '';
  return { box: b.box, color: base, label: `${b.baseLabel}${b.detector === 'coco' ? ` ${fmt(b.baseScore)}` : ''}${suffix}`, dashed: true };
}

function Controls({ trained }: { trained: TrainedModel }) {
  const threshold = useProjectStore((s) => s.threshold);
  const hybrid = useProjectStore((s) => s.hybrid);
  const stale = useProjectStore((s) => s.stale);
  const { setThreshold, setHybrid } = useProjectStore.getState();
  const paths = inferencePaths(trained.kinds);

  return (
    <Card className="flex flex-col gap-4 p-4">
      <div>
        <div className="mb-1 flex items-center justify-between">
          <label htmlFor="threshold" className="text-sm font-semibold text-slate-900">
            Confidence threshold
          </label>
          <span className="font-mono text-sm tabular-nums text-slate-700">{threshold.toFixed(2)}</span>
        </div>
        <input
          id="threshold"
          type="range"
          min={THRESHOLD_MIN}
          max={THRESHOLD_MAX}
          step={0.01}
          value={threshold}
          onChange={(e) => setThreshold(+e.target.value)}
          className="w-full accent-brand-600"
        />
        <p className="text-xs text-slate-500">Below this, the answer is "Unknown".</p>
      </div>
      <label className="flex cursor-pointer items-start gap-2.5">
        <input
          type="checkbox"
          checked={hybrid}
          onChange={(e) => setHybrid(e.target.checked)}
          className="mt-0.5 h-4 w-4 rounded accent-brand-600"
          data-testid="toggle-hybrid"
        />
        <span>
          <span className="block text-sm font-semibold text-slate-900">Detect + Classify</span>
          <span className="block text-xs text-slate-500">
            Mode A localizes ({paths.full ? 'COCO boxes' : ''}
            {paths.full && paths.face ? ' and ' : ''}
            {paths.face ? 'faces' : ''}), each crop gets your custom label.
          </span>
        </span>
      </label>
      <p className="text-xs text-slate-500">
        Paths:{' '}
        {paths.face && paths.full
          ? 'dual (face crop + full frame, higher confidence wins)'
          : paths.face
            ? 'face crop only (the model has no Object classes)'
            : 'full frame only (the model has no Person classes)'}
      </p>
      {stale && (
        <p className="flex items-center gap-1.5 text-xs text-amber-700">
          <AlertIcon size={14} /> Data changed since training; this is the previous model.
        </p>
      )}
    </Card>
  );
}

function ProbBars({ r, labels, trained, chosen }: { r: PathResult; labels: string[]; trained: TrainedModel; chosen: boolean }) {
  const classes = useProjectStore((s) => s.classes);
  return (
    <div className={`rounded-xl p-3 ring-1 ${chosen ? 'bg-brand-50/60 ring-brand-200' : 'ring-slate-200'}`}>
      <p className="mb-2 flex items-center justify-between text-xs font-semibold text-slate-700">
        {r.path === 'face' ? 'a) Face crop' : 'b) Full frame'}
        <span className="font-mono tabular-nums text-slate-500">top {fmt(r.confidence)}</span>
      </p>
      {labels.map((l, i) => (
        <div key={l} className="flex items-center gap-2 py-0.5 text-xs">
          <span className={`w-20 truncate ${i === r.topIndex ? 'font-semibold text-slate-900' : 'text-slate-600'}`}>{l}</span>
          <span className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
            <span className="block h-full rounded-full" style={{ width: `${r.probs[i] * 100}%`, background: outputColor(trained, classes, i) }} />
          </span>
          <span className="w-9 text-right font-mono tabular-nums text-slate-500">{fmt(r.probs[i])}</span>
        </div>
      ))}
    </div>
  );
}

function DualPathPanel({ r, labels, trained }: { r: DualPathPrediction; labels: string[]; trained: TrainedModel }) {
  const p = r.prediction;
  return (
    <Card className="flex flex-col gap-3 p-4" data-testid="dual-path-result">
      {p ? (
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Prediction</p>
          <p className={`text-3xl font-bold tracking-tight ${p.isUnknown ? 'text-slate-500' : 'text-slate-900'}`}>{p.label}</p>
          <p className="text-sm text-slate-600">
            {fmt(p.confidence)} via the {r.chosen === 'face' ? 'face crop' : 'full frame'}
            {p.reason === 'below-threshold' && ` (top guess ${labels[p.topIndex]} is below the threshold)`}
            {p.reason === 'unknown-class' && ' (matched the Unknown class)'}
          </p>
          {r.faceCount > 1 && <p className="text-xs text-amber-700">{r.faceCount} faces in view; the largest is classified.</p>}
        </div>
      ) : (
        <p className="text-sm text-slate-500">No face in view. This model only has Person classes, so it needs a face to classify.</p>
      )}
      {r.face && <ProbBars r={r.face} labels={labels} trained={trained} chosen={r.chosen === 'face'} />}
      {r.full && <ProbBars r={r.full} labels={labels} trained={trained} chosen={r.chosen === 'full'} />}
    </Card>
  );
}

function HybridPanel({ boxes }: { boxes: HybridBox[] }) {
  return (
    <Card className="p-4" data-testid="hybrid-result">
      <p className="mb-2 text-sm font-semibold text-slate-900">Boxes ({boxes.length})</p>
      {boxes.length === 0 ? (
        <p className="text-sm text-slate-500">Mode A found nothing to classify in this frame.</p>
      ) : (
        <ul className="flex flex-col gap-1 text-sm">
          {boxes.map((b, i) => (
            <li key={i} className="flex items-center gap-2">
              <span className="text-slate-500">
                {b.baseLabel}
                {b.detector === 'coco' && ` ${fmt(b.baseScore)}`}
              </span>
              <span className="text-slate-300">→</span>
              {b.prediction ? (
                <span className={b.prediction.isUnknown ? 'text-slate-500' : 'font-semibold text-slate-900'}>
                  {b.prediction.label} {fmt(b.prediction.confidence)}
                </span>
              ) : (
                <span className="text-xs text-slate-400">not classified</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
