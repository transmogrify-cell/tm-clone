import { DEFAULT_TRAINING } from '../config';
import { useProjectStore } from '../store/useProjectStore';

const BATCH_SIZES = [8, 16, 32, 64, 128, 256];

/** Advanced hyper-parameters: epochs, batch size, learning rate. */
export function TrainingSettings({ disabled }: { disabled?: boolean }) {
  const params = useProjectStore((s) => s.params);
  const setParams = useProjectStore((s) => s.setParams);

  const field = 'mt-1 h-9 w-full rounded-lg border-0 bg-white px-2.5 text-sm text-slate-900 ring-1 ring-inset ring-slate-300 focus:ring-2 focus:ring-brand-500 disabled:bg-slate-50 disabled:text-slate-400';

  return (
    <fieldset disabled={disabled} className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      <label className="text-xs font-medium text-slate-600">
        Epochs
        <input
          type="number"
          min={1}
          max={500}
          value={params.epochs}
          onChange={(e) => setParams({ epochs: clamp(Math.round(+e.target.value), 1, 500) })}
          className={field}
        />
        <span className="mt-1 block font-normal text-slate-400">Passes over the whole dataset.</span>
      </label>
      <label className="text-xs font-medium text-slate-600">
        Batch size
        <select value={params.batchSize} onChange={(e) => setParams({ batchSize: +e.target.value })} className={field}>
          {BATCH_SIZES.map((b) => (
            <option key={b} value={b}>
              {b}
            </option>
          ))}
        </select>
        <span className="mt-1 block font-normal text-slate-400">Samples per weight update.</span>
      </label>
      <label className="text-xs font-medium text-slate-600">
        Learning rate
        <input
          type="number"
          min={0.00001}
          max={1}
          step={0.0001}
          value={params.learningRate}
          onChange={(e) => {
            const v = +e.target.value;
            if (v > 0) setParams({ learningRate: clamp(v, 0.00001, 1) });
          }}
          className={field}
        />
        <span className="mt-1 block font-normal text-slate-400">Adam step size.</span>
      </label>
      <div className="sm:col-span-3">
        <button
          type="button"
          onClick={() => setParams({ ...DEFAULT_TRAINING })}
          className="text-xs font-medium text-brand-700 hover:underline disabled:text-slate-400"
        >
          Reset to defaults
        </button>
      </div>
    </fieldset>
  );
}

function clamp(v: number, lo: number, hi: number) {
  return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : lo;
}
