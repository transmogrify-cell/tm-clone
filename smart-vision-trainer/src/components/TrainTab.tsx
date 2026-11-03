import { MAX_CLASSES, MIN_SAMPLES, MIN_SAMPLES_UNKNOWN } from '../config';
import { useTraining } from '../hooks/useTraining';
import { useProjectStore, type TrainView } from '../store/useProjectStore';
import type { TrainingParams } from '../types';
import { AccuracyReport } from './AccuracyReport';
import { ClassCard } from './ClassCard';
import { TrainingProgress } from './TrainingProgress';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { AlertIcon, BoxIcon, HelpIcon, SpinnerIcon, SparkIcon, UserIcon } from './ui/icons';

const VIEWS: { id: TrainView; label: string }[] = [
  { id: 'classes', label: 'Classes' },
  { id: 'progress', label: 'Training' },
  { id: 'report', label: 'Accuracy report' },
];

/** Mode B: class cards (collect samples), training settings, live training, accuracy report. */
export function TrainTab() {
  const view = useProjectStore((s) => s.trainView);
  const setTrainView = useProjectStore((s) => s.setTrainView);
  const phase = useProjectStore((s) => s.progress.phase);
  const hasReport = useProjectStore((s) => !!s.trained?.evaluation);

  return (
    <main className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-6">
      <nav className="inline-flex self-start rounded-lg bg-slate-200/70 p-0.5" aria-label="Training steps">
        {VIEWS.map((v) => {
          const disabled = (v.id === 'progress' && phase === 'idle') || (v.id === 'report' && !hasReport);
          return (
            <button
              key={v.id}
              type="button"
              disabled={disabled}
              aria-current={view === v.id ? 'page' : undefined}
              onClick={() => setTrainView(v.id)}
              className={`h-8 rounded-md px-3 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                view === v.id ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {v.label}
            </button>
          );
        })}
      </nav>
      {view === 'classes' && <ClassesView />}
      {view === 'progress' && <TrainingProgress />}
      {view === 'report' && <AccuracyReport />}
    </main>
  );
}

function ClassesView() {
  const classes = useProjectStore((s) => s.classes);
  const stale = useProjectStore((s) => s.stale);
  const trained = useProjectStore((s) => s.trained);
  const { addClass, addUnknownClass } = useProjectStore.getState();
  const { train, busy, readiness } = useTraining();
  const regular = classes.filter((c) => c.kind !== 'unknown');
  const hasUnknown = classes.some((c) => c.kind === 'unknown');
  const full = regular.length >= MAX_CLASSES;

  return (
    <>
      <Card className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="text-sm text-slate-600">
          <p className="font-semibold text-slate-900">Add classes, collect samples, train.</p>
          <p>
            <b>Person</b> classes learn a face (crop via BlazeFace). <b>Object</b> classes learn the whole image. Min {MIN_SAMPLES} samples per
            class{hasUnknown ? `, ${MIN_SAMPLES_UNKNOWN} for Unknown` : ''}.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button onClick={() => addClass('person')} disabled={busy || full}>
            <UserIcon size={16} /> Person class
          </Button>
          <Button onClick={() => addClass('object')} disabled={busy || full}>
            <BoxIcon size={16} /> Object class
          </Button>
          {!hasUnknown && (
            <Button variant="ghost" onClick={addUnknownClass} disabled={busy} title="Lets the model answer 'Unknown'">
              <HelpIcon size={16} /> Unknown class
            </Button>
          )}
        </div>
      </Card>

      {stale && trained && (
        <p className="flex items-center gap-2 rounded-xl bg-amber-50 px-4 py-3 text-sm text-amber-900 ring-1 ring-amber-200">
          <AlertIcon size={16} /> Classes or samples changed since the last training. Retrain to include them.
        </p>
      )}

      <div className="grid gap-4 xl:grid-cols-2">
        {classes.map((c) => (
          <ClassCard key={c.id} cls={c} canDelete={c.kind === 'unknown' || regular.length > 1} />
        ))}
      </div>

      <Card className="sticky bottom-3 z-10 flex flex-col gap-3 p-4 shadow-lg lg:flex-row lg:items-center">
        <TrainingSettings disabled={busy} />
        <div className="flex flex-1 flex-col items-stretch gap-1 lg:items-end">
          <Button variant="primary" size="lg" onClick={train} disabled={busy || !readiness.ok} data-testid="train-button">
            {busy ? <SpinnerIcon /> : <SparkIcon />} {busy ? 'Training…' : trained ? 'Retrain model' : 'Train model'}
          </Button>
          {!readiness.ok && <p className="text-xs text-slate-500 lg:text-right">{readiness.reason}</p>}
        </div>
      </Card>
    </>
  );
}

const FIELDS: { key: keyof TrainingParams; label: string; min: number; max: number; step: number }[] = [
  { key: 'epochs', label: 'Epochs', min: 5, max: 200, step: 1 },
  { key: 'batchSize', label: 'Batch', min: 4, max: 128, step: 4 },
  { key: 'learningRate', label: 'Learning rate', min: 0.00005, max: 0.01, step: 0.00005 },
  { key: 'patience', label: 'Patience', min: 1, max: 20, step: 1 },
];

function TrainingSettings({ disabled }: { disabled: boolean }) {
  const params = useProjectStore((s) => s.params);
  const setParams = useProjectStore((s) => s.setParams);
  return (
    <details className="text-sm">
      <summary className="cursor-pointer select-none font-medium text-slate-600">
        Settings: {params.epochs} epochs · batch {params.batchSize} · adam {params.learningRate} · early stop {params.patience}
      </summary>
      <div className="mt-3 flex flex-wrap gap-3">
        {FIELDS.map((f) => (
          <label key={f.key} className="flex flex-col gap-1 text-xs text-slate-500">
            {f.label}
            <input
              type="number"
              min={f.min}
              max={f.max}
              step={f.step}
              value={params[f.key]}
              disabled={disabled}
              onChange={(e) => {
                const v = +e.target.value;
                if (Number.isFinite(v) && v >= f.min && v <= f.max) setParams({ [f.key]: v });
              }}
              className="w-28 rounded-md px-2 py-1 text-sm text-slate-900 ring-1 ring-slate-300 focus:outline-none focus:ring-2 focus:ring-brand-500"
            />
          </label>
        ))}
        <p className="basis-full text-xs text-slate-400">
          Fixed: 20% stratified validation split, L2 0.001, BatchNorm, dropout 0.4 / 0.3, restore best weights.
        </p>
      </div>
    </details>
  );
}
