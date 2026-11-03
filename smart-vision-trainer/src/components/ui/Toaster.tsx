import { useToastStore, type ToastKind } from '../../store/useToastStore';
import { AlertIcon, CheckIcon, InfoIcon, XIcon } from './icons';

const styles: Record<ToastKind, string> = {
  info: 'bg-slate-900 text-white',
  success: 'bg-emerald-600 text-white',
  warning: 'bg-amber-500 text-slate-950',
  error: 'bg-red-600 text-white',
};

const icons: Record<ToastKind, JSX.Element> = {
  info: <InfoIcon />,
  success: <CheckIcon />,
  warning: <AlertIcon />,
  error: <AlertIcon />,
};

export function Toaster() {
  const toasts = useToastStore((s) => s.toasts);
  const dismiss = useToastStore((s) => s.dismiss);
  return (
    <div
      aria-live="polite"
      className="pointer-events-none fixed inset-x-0 bottom-24 z-[60] flex flex-col items-center gap-2 px-4 sm:bottom-auto sm:left-auto sm:right-4 sm:top-4 sm:items-end"
    >
      {toasts.map((t) => (
        <div
          key={t.id}
          role={t.kind === 'error' ? 'alert' : 'status'}
          className={`pointer-events-auto flex w-full max-w-sm animate-toast-in items-start gap-2.5 rounded-xl px-4 py-3 text-sm shadow-lg ${styles[t.kind]}`}
        >
          <span className="mt-0.5 shrink-0">{icons[t.kind]}</span>
          <p className="flex-1 leading-snug">{t.message}</p>
          <button type="button" onClick={() => dismiss(t.id)} className="shrink-0 opacity-70 hover:opacity-100" aria-label="Dismiss">
            <XIcon size={16} />
          </button>
        </div>
      ))}
    </div>
  );
}
