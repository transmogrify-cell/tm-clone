import { useTraining } from '../hooks/useTraining';
import { useProjectStore } from '../store/useProjectStore';
import { Button } from './ui/Button';
import { SpinnerIcon } from './ui/icons';

/** Starts training. Disabled (with the reason shown) until every class has enough samples. */
export function TrainButton({ className = '' }: { className?: string }) {
  const { train, busy, readiness } = useTraining();
  const trained = useProjectStore((s) => s.trained);
  const stale = useProjectStore((s) => s.stale);

  const label = busy ? 'Training…' : trained ? (stale ? 'Retrain Model' : 'Train Again') : 'Train Model';

  return (
    <div className={`flex flex-col items-stretch gap-1 ${className}`}>
      <Button
        variant="primary"
        size="lg"
        onClick={() => void train()}
        disabled={busy || !readiness.ok}
        title={readiness.reason}
        aria-describedby={readiness.ok ? undefined : 'train-hint'}
      >
        {busy && <SpinnerIcon />}
        {label}
      </Button>
      {!readiness.ok && !busy && (
        <p id="train-hint" className="max-w-xs text-xs text-slate-500 sm:text-right">
          {readiness.reason}
        </p>
      )}
    </div>
  );
}
