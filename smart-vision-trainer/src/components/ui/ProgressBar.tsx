interface ProgressBarProps {
  /** 0..1 */
  value: number;
  label?: string;
  className?: string;
  tone?: 'brand' | 'emerald' | 'amber';
}

const tones = {
  brand: 'bg-brand-600',
  emerald: 'bg-emerald-500',
  amber: 'bg-amber-500',
};

export function ProgressBar({ value, label, className = '', tone = 'brand' }: ProgressBarProps) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      aria-label={label}
      className={`h-2 w-full overflow-hidden rounded-full bg-slate-200 ${className}`}
    >
      <div className={`h-full rounded-full transition-[width] duration-200 ease-out ${tones[tone]}`} style={{ width: `${pct}%` }} />
    </div>
  );
}
