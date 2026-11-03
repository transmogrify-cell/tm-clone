import { requestStop } from '../hooks/useTraining';
import { VARIANTS_PER_IMAGE } from '../ml/augmentation';
import { useProjectStore } from '../store/useProjectStore';
import type { EpochLog } from '../types';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { LineChart } from './ui/LineChart';
import { ProgressBar } from './ui/ProgressBar';
import { AlertIcon, CheckIcon, SpinnerIcon, StopIcon } from './ui/icons';

const TRAIN_COLOR = '#2a78d6';
const VAL_COLOR = '#eb6834';
const pct = (v?: number) => (v === undefined || Number.isNaN(v) ? '–' : `${(v * 100).toFixed(1)}%`);
const num = (v?: number) => (v === undefined || Number.isNaN(v) ? '–' : v.toFixed(3));

function bestSoFar(history: EpochLog[]) {
  let best: EpochLog | undefined;
  for (const h of history) if (h.valLoss !== undefined && (!best || h.valLoss < best.valLoss!)) best = h;
  return best;
}

/** Training view: feature-extraction progress, then live loss/accuracy curves. */
export function TrainingProgress() {
  const progress = useProjectStore((s) => s.progress);
  const params = useProjectStore((s) => s.params);
  const setTrainView = useProjectStore((s) => s.setTrainView);
  const { phase, history } = progress;
  const best = bestSoFar(history);
  const sinceBest = best ? history.length - best.epoch : 0;

  let headline: JSX.Element;
  let bar: JSX.Element | null = null;
  if (phase === 'extracting') {
    headline = (
      <>
        <SpinnerIcon /> Extracting features: {progress.extracted}/{progress.extractTotal} images
      </>
    );
    bar = <ProgressBar value={progress.extractTotal ? progress.extracted / progress.extractTotal : 0} tone="amber" label="Feature extraction" />;
  } else if (phase === 'training') {
    headline = (
      <>
        <SpinnerIcon /> Training: epoch {Math.min(progress.totalEpochs, progress.epoch + 1)}/{progress.totalEpochs}
      </>
    );
    bar = <ProgressBar value={(progress.epoch + progress.epochFraction) / progress.totalEpochs} label="Training" />;
  } else if (phase === 'evaluating') {
    headline = (
      <>
        <SpinnerIcon /> Evaluating on held-out samples…
      </>
    );
  } else if (phase === 'done') {
    headline = (
      <>
        <CheckIcon className="text-emerald-600" /> Training complete
      </>
    );
  } else if (phase === 'error') {
    headline = (
      <>
        <AlertIcon className="text-red-600" /> Training failed
      </>
    );
  } else {
    headline = <>No training run yet</>;
  }

  return (
    <div className="flex flex-col gap-4">
      <Card className="flex flex-col gap-3 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="flex items-center gap-2 text-lg font-semibold text-slate-900 tabular-nums">{headline}</h1>
          {phase === 'training' && (
            <Button variant="danger" size="sm" onClick={requestStop}>
              <StopIcon size={14} /> Stop and keep best
            </Button>
          )}
          {phase === 'done' && (
            <Button variant="primary" onClick={() => setTrainView('report')}>
              View Accuracy Report
            </Button>
          )}
          {phase === 'error' && <Button onClick={() => setTrainView('classes')}>Back to classes</Button>}
        </div>
        {bar}
        <p className="text-sm text-slate-500">
          {phase === 'extracting' &&
            `${progress.samples} samples × ${VARIANTS_PER_IMAGE} augmented variants (original, flip, brightness ±15%, rotation ±10°, zoom 0.9x, zoom 1.1x) through the frozen MobileNet V2. Samples cached from an earlier run are skipped.`}
          {phase === 'training' &&
            (best
              ? `Lowest validation loss so far: ${num(best.valLoss)} at epoch ${best.epoch}. Early stopping after ${params.patience} epochs without improvement (${sinceBest}/${params.patience}).`
              : 'Training the small head on cached features…')}
          {phase === 'done' &&
            `${progress.stoppedEarly ? `Stopped early after ${history.length} epochs` : `Ran all ${history.length} epochs`}; kept the weights from epoch ${progress.bestEpoch} (lowest validation loss).`}
          {phase === 'error' && progress.error}
        </p>
      </Card>

      <Card className="grid gap-6 p-5 lg:grid-cols-2">
        <LineChart
          title="Loss (lower is better)"
          series={[
            { name: 'Train', color: TRAIN_COLOR, values: history.map((h) => h.loss) },
            { name: 'Validation', color: VAL_COLOR, values: history.map((h) => h.valLoss) },
          ]}
          xMax={progress.totalEpochs}
          format={(v) => v.toFixed(2)}
          marker={best ? { x: best.epoch, label: 'best' } : undefined}
        />
        <LineChart
          title="Accuracy"
          series={[
            { name: 'Train', color: TRAIN_COLOR, values: history.map((h) => h.acc) },
            { name: 'Validation', color: VAL_COLOR, values: history.map((h) => h.valAcc) },
          ]}
          xMax={progress.totalEpochs}
          yMax={1}
          format={(v) => `${Math.round(v * 100)}%`}
          marker={best ? { x: best.epoch, label: 'best' } : undefined}
        />
        <details className="text-sm lg:col-span-2">
          <summary className="cursor-pointer font-medium text-slate-600">Show as table</summary>
          <div className="mt-2 max-h-56 overflow-auto rounded-lg ring-1 ring-slate-200">
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
                  <tr key={h.epoch} className={`border-t border-slate-100 ${h.epoch === best?.epoch ? 'bg-brand-50' : ''}`}>
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
        </details>
      </Card>
    </div>
  );
}
