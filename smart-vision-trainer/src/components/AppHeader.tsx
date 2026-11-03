import { useDetectors } from '../hooks/useDetectors';
import { useProjectStore, type ModelKey, type Tab } from '../store/useProjectStore';
import { AlertIcon, CheckIcon, SpinnerIcon } from './ui/icons';

const TABS: { id: Tab; label: string }[] = [
  { id: 'detect', label: 'Detect' },
  { id: 'train', label: 'Train' },
  { id: 'preview', label: 'Preview' },
  { id: 'export', label: 'Export' },
];

const MODEL_NAMES: Record<ModelKey, string> = {
  coco: 'COCO-SSD',
  face: 'BlazeFace',
  hands: 'Hands',
  mobilenet: 'MobileNet',
};

export function AppHeader() {
  const tab = useProjectStore((s) => s.tab);
  const setTab = useProjectStore((s) => s.setTab);
  const { models, backend, retry } = useDetectors();
  const failed = Object.values(models).some((m) => m.status === 'error');

  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 pt-3">
        <button type="button" onClick={() => useProjectStore.setState({ screen: 'landing' })} className="flex items-center gap-2 font-semibold text-slate-900">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-600 text-white" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
              <path d="M3 8V5a2 2 0 0 1 2-2h3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M8 21H5a2 2 0 0 1-2-2v-3" />
              <circle cx="12" cy="12" r="3" fill="currentColor" />
            </svg>
          </span>
          Smart Vision Trainer
        </button>
        <div className="ml-auto flex flex-wrap items-center gap-1.5 text-xs" aria-label="Model status">
          {(Object.keys(MODEL_NAMES) as ModelKey[]).map((k) => {
            const st = models[k];
            return (
              <span
                key={k}
                title={st.error ?? `${MODEL_NAMES[k]}: ${st.status}`}
                className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-medium ring-1 ring-inset ${
                  st.status === 'ready'
                    ? 'bg-emerald-50 text-emerald-700 ring-emerald-200'
                    : st.status === 'error'
                      ? 'bg-red-50 text-red-700 ring-red-200'
                      : 'bg-slate-50 text-slate-500 ring-slate-200'
                }`}
                data-testid={`model-${k}`}
                data-status={st.status}
              >
                {st.status === 'ready' ? <CheckIcon size={11} /> : st.status === 'error' ? <AlertIcon size={11} /> : <SpinnerIcon size={11} />}
                {MODEL_NAMES[k]}
              </span>
            );
          })}
          {backend && <span className="rounded-full bg-slate-100 px-2 py-0.5 font-mono text-slate-500">{backend}</span>}
          {failed && (
            <button type="button" onClick={() => void retry()} className="font-medium text-brand-700 hover:underline">
              Retry
            </button>
          )}
        </div>
      </div>
      <nav className="mx-auto flex max-w-7xl gap-1 overflow-x-auto px-4" role="tablist" aria-label="Sections">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            data-tab={t.id}
            className={`-mb-px whitespace-nowrap border-b-2 px-3 py-2.5 text-sm font-medium transition-colors ${
              tab === t.id ? 'border-brand-600 text-brand-700' : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>
    </header>
  );
}
