import { useMemo, useState } from 'react';
import { THRESHOLD_MAX, THRESHOLD_MIN } from '../config';
import { thresholdStats, weakClasses } from '../ml/evaluator';
import { classColor, resolveLabels, useProjectStore } from '../store/useProjectStore';
import type { Evaluation } from '../types';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { AlertIcon, CheckIcon, ScanIcon } from './ui/icons';

const fmt = (v: number) => (Number.isNaN(v) ? '–' : `${(v * 100).toFixed(1)}%`);

// Sequential blue ramp (light -> dark) for the heatmap; index 0 is "zero".
const RAMP = ['#f3f5f8', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
const rampIndex = (f: number) => (f <= 0 ? 0 : Math.min(7, Math.max(1, Math.ceil(f * 7))));

/** STEP 5 report: overall accuracy, per-class precision/recall/support, confusion matrix heatmap, threshold impact. */
export function AccuracyReport() {
  const trained = useProjectStore((s) => s.trained);
  const classes = useProjectStore((s) => s.classes);
  const progress = useProjectStore((s) => s.progress);
  const threshold = useProjectStore((s) => s.threshold);
  const setThreshold = useProjectStore((s) => s.setThreshold);
  const { setTrainView, setTab } = useProjectStore.getState();
  const ev = trained?.evaluation ?? null;
  const labels = useMemo(() => (trained ? resolveLabels(trained, classes) : []), [trained, classes]);

  if (!trained || !ev) {
    return (
      <Card className="p-6 text-sm text-slate-600">No accuracy report yet. Train a model first; every class needs at least 2 samples so one can be held out.</Card>
    );
  }

  const weak = weakClasses({ ...ev, perClass: ev.perClass.map((m, i) => ({ ...m, label: labels[i] })) });
  const minSupport = Math.min(...ev.perClass.map((m) => m.support));
  const stats = thresholdStats(ev, threshold, trained.kinds);
  const n = ev.valTrue.length;
  const good = ev.overallAccuracy >= 0.85;

  return (
    <div className="flex flex-col gap-4">
      <Card className="flex flex-col gap-4 p-6 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-medium text-slate-500">Overall Accuracy</p>
          <p className="text-5xl font-bold tracking-tight text-slate-900 tabular-nums" data-testid="overall-accuracy">
            {fmt(ev.overallAccuracy)}
          </p>
          <p className="mt-1 text-sm text-slate-500">
            On {n} held-out samples (20% of each class, never used for training, un-augmented). Weights from epoch {trained.bestEpoch} of{' '}
            {trained.epochsRun}
            {progress.stoppedEarly ? ' (early stopping)' : ''}.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button onClick={() => setTrainView('classes')}>Add more samples</Button>
          <Button variant="primary" size="lg" onClick={() => setTab('preview')}>
            <ScanIcon /> Preview
          </Button>
        </div>
      </Card>

      {(weak.length > 0 || minSupport < 10) && (
        <div className="flex flex-col gap-2">
          {weak.map((w) => (
            <p key={w.label} role="alert" className="flex items-start gap-2 rounded-xl bg-amber-50 px-4 py-3 text-sm text-amber-900 ring-1 ring-amber-200">
              <AlertIcon size={16} className="mt-0.5 shrink-0" /> <span>{w.message}</span>
            </p>
          ))}
          {minSupport < 10 && (
            <p className="flex items-start gap-2 rounded-xl bg-slate-100 px-4 py-3 text-sm text-slate-700">
              <AlertIcon size={16} className="mt-0.5 shrink-0" />
              <span>
                Small test set: the smallest class has only {minSupport} held-out sample{minSupport === 1 ? '' : 's'}, so each one moves its
                recall by {Math.round(100 / Math.max(1, minSupport))} points. More samples give steadier numbers.
              </span>
            </p>
          )}
        </div>
      )}
      {weak.length === 0 && (
        <p className={`flex items-center gap-2 rounded-xl px-4 py-3 text-sm ring-1 ${good ? 'bg-emerald-50 text-emerald-800 ring-emerald-200' : 'bg-slate-100 text-slate-700 ring-slate-200'}`}>
          <CheckIcon size={16} /> Every class is at or above 70% recall and precision.
        </p>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="overflow-hidden">
          <h2 className="border-b border-slate-100 px-5 py-3 font-semibold text-slate-900">Per-class accuracy</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm" data-testid="per-class-table">
              <thead className="bg-slate-50 text-xs text-slate-500">
                <tr>
                  <th className="px-5 py-2 text-left font-medium">Class</th>
                  <th className="px-3 py-2 text-right font-medium" title="When it predicts this class, how often is it right?">
                    Precision
                  </th>
                  <th className="px-3 py-2 text-right font-medium" title="Of this class's test samples, how many were recognised?">
                    Recall
                  </th>
                  <th className="px-3 py-2 text-right font-medium">F1</th>
                  <th className="px-3 py-2 text-right font-medium" title="Held-out test samples behind this row">
                    Support
                  </th>
                  <th className="px-5 py-2 text-right font-medium">Samples</th>
                </tr>
              </thead>
              <tbody>
                {ev.perClass.map((m, i) => (
                  <tr key={trained.classIds[i]} className="border-t border-slate-100">
                    <td className="px-5 py-2">
                      <span className="flex items-center gap-2 font-medium text-slate-900">
                        <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: classColor(classes, trained.classIds[i]) }} />
                        {labels[i]}
                        <span className="text-xs font-normal capitalize text-slate-400">{m.kind}</span>
                      </span>
                    </td>
                    <Cell v={m.precision} />
                    <Cell v={m.recall} />
                    <Cell v={m.f1} />
                    <td className="px-3 py-2 text-right tabular-nums text-slate-500">{m.support}</td>
                    <td className="px-5 py-2 text-right tabular-nums text-slate-500">{m.samples}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="px-5 py-3 text-xs leading-relaxed text-slate-500">
            <b>Precision</b>: when the model says "{labels[0]}", how often it really is. <b>Recall</b>: of the {labels[0]} test samples, how many it
            recognised. <b>Support</b>: held-out samples behind each row. Red = below 70%.
          </p>
        </Card>

        <ConfusionMatrix ev={ev} labels={labels} />
      </div>

      <Card className="flex flex-col gap-3 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="font-semibold text-slate-900">Confidence threshold</h2>
          <span className="font-mono text-sm tabular-nums text-slate-700">{threshold.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={THRESHOLD_MIN}
          max={THRESHOLD_MAX}
          step={0.01}
          value={threshold}
          onChange={(e) => setThreshold(+e.target.value)}
          aria-label="Confidence threshold"
          className="w-full accent-brand-600"
        />
        <p className="text-sm text-slate-600">
          At this threshold, <b>{fmt(stats.coverage)}</b> of the test samples get a confident answer and <b>{fmt(stats.confidentAccuracy)}</b> of
          those are right. {stats.rejected > 0 && `${stats.rejected} sample(s) of your classes would show "Unknown".`} Raise it to reject more
          unfamiliar inputs; lower it if your own classes often come out as "Unknown".
        </p>
      </Card>
    </div>
  );
}

function Cell({ v }: { v: number }) {
  const weak = !Number.isNaN(v) && v < 0.7;
  return <td className={`px-3 py-2 text-right tabular-nums ${weak ? 'font-semibold text-red-700' : 'text-slate-800'}`}>{fmt(v)}</td>;
}

/** Rows = actual class, columns = prediction; color = share of the row. */
function ConfusionMatrix({ ev, labels }: { ev: Evaluation; labels: string[] }) {
  const [hover, setHover] = useState<{ a: number; p: number } | null>(null);
  const rowSums = ev.confusion.map((r) => r.reduce((x, y) => x + y, 0));
  const k = labels.length;

  return (
    <Card className="flex flex-col">
      <h2 className="border-b border-slate-100 px-5 py-3 font-semibold text-slate-900">Confusion matrix</h2>
      <div className="overflow-x-auto p-5">
        <div className="mb-2 pl-24 text-center text-xs font-medium text-slate-500">Predicted</div>
        <div className="flex">
          <div className="flex w-6 items-center justify-center">
            <span className="-rotate-90 whitespace-nowrap text-xs font-medium text-slate-500">Actual</span>
          </div>
          <table className="border-separate" style={{ borderSpacing: 2 }} data-testid="confusion-matrix">
            <thead>
              <tr>
                <th className="w-20" />
                {labels.map((l) => (
                  <th key={l} className="max-w-[5rem] truncate px-1 pb-1 text-xs font-medium text-slate-600" title={l}>
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ev.confusion.map((row, a) => (
                <tr key={a}>
                  <th className="max-w-[5rem] truncate pr-2 text-right text-xs font-medium text-slate-600" title={labels[a]}>
                    {labels[a]}
                  </th>
                  {row.map((count, p) => {
                    const f = rowSums[a] ? count / rowSums[a] : 0;
                    const idx = rampIndex(f);
                    const dark = idx >= 5;
                    return (
                      <td
                        key={p}
                        onMouseEnter={() => setHover({ a, p })}
                        onMouseLeave={() => setHover(null)}
                        className={`relative h-14 min-w-[3.5rem] rounded text-center align-middle tabular-nums ${a === p ? 'ring-2 ring-inset ring-slate-900/15' : ''}`}
                        style={{ background: RAMP[idx], color: dark ? '#fff' : '#0b0b0b' }}
                        aria-label={`Actual ${labels[a]}, predicted ${labels[p]}: ${count} of ${rowSums[a]}`}
                      >
                        <div className="text-sm font-semibold">{count}</div>
                        <div className={`text-[10px] ${dark ? 'text-white/80' : 'text-slate-500'}`}>{rowSums[a] ? `${Math.round(f * 100)}%` : ''}</div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 min-h-[1.25rem] text-xs text-slate-600" aria-live="polite">
          {hover
            ? `${labels[hover.a]} samples predicted as ${labels[hover.p]}: ${ev.confusion[hover.a][hover.p]} of ${rowSums[hover.a]}${
                hover.a === hover.p ? ' (correct)' : ''
              }`
            : `Diagonal = correct. Off-diagonal cells show which class gets mistaken for which (${k}×${k}). Hover a cell for details.`}
        </p>
        <div className="mt-2 flex items-center gap-2 text-[10px] text-slate-500">
          0%
          <span className="flex">
            {RAMP.slice(1).map((c) => (
              <span key={c} className="h-2 w-5" style={{ background: c }} />
            ))}
          </span>
          100% of the row
        </div>
      </div>
    </Card>
  );
}
