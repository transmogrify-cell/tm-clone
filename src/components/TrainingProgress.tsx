import { useProjectStore } from '../store/useProjectStore';
import type { EpochLog } from '../types';
import { ProgressBar } from './ui/ProgressBar';
import { AlertIcon, CheckIcon } from './ui/icons';

const pct = (v?: number) => (v === undefined || Number.isNaN(v) ? '–' : `${(v * 100).toFixed(1)}%`);
const num = (v?: number) => (v === undefined || Number.isNaN(v) ? '–' : v.toFixed(4));

function Metrics({ log }: { log?: EpochLog }) {
  if (!log) return null;
  const items = [
    ['loss', num(log.loss)],
    ['acc', pct(log.acc)],
    ['val loss', num(log.valLoss)],
    ['val acc', pct(log.valAcc)],
  ];
  return (
    <dl className="flex flex-wrap gap-x-4 gap-y-0.5 font-mono text-xs text-slate-600">
      {items.map(([k, v]) => (
        <div key={k} className="flex gap-1">
          <dt className="text-slate-400">{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Live status of a training run: feature extraction (per image), then epochs. */
export function TrainingProgress() {
  const progress = useProjectStore((s) => s.progress);
  const trained = useProjectStore((s) => s.trained);
  const stale = useProjectStore((s) => s.stale);
  const last = progress.history[progress.history.length - 1];

  switch (progress.phase) {
    case 'extracting':
      return (
        <div className="flex min-w-0 flex-col gap-1.5">
          <div className="flex items-baseline justify-between gap-3 text-sm">
            <span className="font-medium text-slate-800">Extracting features with MobileNet</span>
            <span className="tabular-nums text-slate-500">
              {progress.extracted} / {progress.extractTotal} images
            </span>
          </div>
          <ProgressBar value={progress.extractTotal ? progress.extracted / progress.extractTotal : 0} label="Feature extraction" tone="amber" />
        </div>
      );
    case 'training': {
      const done = Math.min(progress.totalEpochs, progress.epoch + progress.epochFraction);
      return (
        <div className="flex min-w-0 flex-col gap-1.5">
          <div className="flex items-baseline justify-between gap-3 text-sm">
            <span className="font-medium text-slate-800">Training the classifier head</span>
            <span className="tabular-nums text-slate-500">
              Epoch {Math.min(progress.totalEpochs, progress.epoch + 1)} / {progress.totalEpochs}
            </span>
          </div>
          <ProgressBar value={done / progress.totalEpochs} label="Training" />
          <Metrics log={last} />
        </div>
      );
    }
    case 'error':
      return (
        <p className="flex items-start gap-2 text-sm text-red-600">
          <AlertIcon className="mt-0.5 shrink-0" /> {progress.error}
        </p>
      );
    default:
      if (!trained) {
        return <p className="text-sm text-slate-500">Add at least 5 samples to each class, then train. Training runs entirely in this tab.</p>;
      }
      return (
        <div className="flex min-w-0 flex-col gap-1">
          <p className={`flex items-center gap-1.5 text-sm font-medium ${stale ? 'text-amber-700' : 'text-emerald-700'}`}>
            {stale ? <AlertIcon size={16} /> : <CheckIcon size={16} />}
            {stale ? 'Samples changed since training. Retrain to include them.' : `Model trained on ${trained.sampleCount || '–'} samples`}
          </p>
          <Metrics log={trained.finalMetrics ?? last} />
        </div>
      );
  }
}

/** Per-epoch metrics table ("under the hood"). */
export function TrainingHistory() {
  const history = useProjectStore((s) => s.progress.history);
  if (history.length === 0) return <p className="text-sm text-slate-500">Metrics for each epoch appear here once you train.</p>;
  return (
    <div className="max-h-48 overflow-auto rounded-lg ring-1 ring-slate-200">
      <table className="w-full text-right font-mono text-xs">
        <thead className="sticky top-0 bg-slate-50 text-slate-500">
          <tr>
            {['epoch', 'loss', 'acc', 'val loss', 'val acc'].map((h) => (
              <th key={h} className="px-3 py-1.5 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="text-slate-700">
          {history.map((h) => (
            <tr key={h.epoch} className="border-t border-slate-100">
              <td className="px-3 py-1">{h.epoch}</td>
              <td className="px-3 py-1">{num(h.loss)}</td>
              <td className="px-3 py-1">{pct(h.acc)}</td>
              <td className="px-3 py-1">{num(h.valLoss)}</td>
              <td className="px-3 py-1">{pct(h.valAcc)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
