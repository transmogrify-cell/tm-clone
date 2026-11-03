import { useState } from 'react';
import { MAX_CLASSES, MIN_CLASSES } from '../config';
import { isBusy, useProjectStore } from '../store/useProjectStore';
import { ClassCard } from './ClassCard';
import { ExportModal } from './ExportModal';
import { HowItWorks } from './HowItWorks';
import { Logo, ModelStatusLine } from './Landing';
import { PreviewPanel } from './PreviewPanel';
import { TrainButton } from './TrainButton';
import { TrainingHistory, TrainingProgress } from './TrainingProgress';
import { TrainingSettings } from './TrainingSettings';
import { Button } from './ui/Button';
import { ArrowLeftIcon, ChevronIcon, DownloadIcon, PlusIcon, SettingsIcon } from './ui/icons';

export function ProjectWorkspace() {
  const classes = useProjectStore((s) => s.classes);
  const trained = useProjectStore((s) => s.trained);
  const busy = useProjectStore((s) => isBusy(s.progress));
  const { addClass, goTo, newProject } = useProjectStore.getState();
  const [exportOpen, setExportOpen] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);

  const startOver = () => {
    const hasWork = classes.some((c) => c.samples.length > 0) || trained;
    if (!hasWork || window.confirm('Start a new project? All samples and the trained model will be discarded.')) newProject();
  };

  return (
    <div className="min-h-screen pb-44 sm:pb-32">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-3 px-4">
          <button
            type="button"
            onClick={() => goTo('landing')}
            className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 hover:text-slate-800"
            aria-label="Back to start"
          >
            <ArrowLeftIcon />
          </button>
          <Logo />
          <h1 className="truncate font-semibold text-slate-900">Image Project</h1>
          <div className="ml-auto hidden text-xs md:block">
            <ModelStatusLine />
          </div>
          <Button size="sm" variant="ghost" onClick={startOver} disabled={busy}>
            New project
          </Button>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-4 py-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <section className="flex min-w-0 flex-col gap-4" aria-label="Classes">
          <HowItWorks />
          <div className="grid gap-4 xl:grid-cols-2">
            {classes.map((cls, i) => (
              <ClassCard key={cls.id} cls={cls} index={i} canDelete={classes.length > MIN_CLASSES} />
            ))}
          </div>
          <button
            type="button"
            onClick={addClass}
            disabled={busy || classes.length >= MAX_CLASSES}
            className="flex h-16 items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-slate-300 text-sm font-semibold text-slate-500 transition-colors hover:border-brand-400 hover:bg-white hover:text-brand-700 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-300 disabled:hover:bg-transparent"
          >
            <PlusIcon />
            {classes.length >= MAX_CLASSES ? `Maximum of ${MAX_CLASSES} classes` : 'Add a class'}
          </button>
        </section>

        <aside className="min-w-0 lg:sticky lg:top-20 lg:self-start" aria-label="Preview">
          <PreviewPanel onExport={() => setExportOpen(true)} />
        </aside>
      </main>

      {/* Bottom bar: training status, advanced settings, Train and Export. */}
      <div className="fixed inset-x-0 bottom-0 z-30 border-t border-slate-200 bg-white/95 shadow-[0_-4px_16px_rgba(15,23,42,0.06)] backdrop-blur">
        {panelOpen && (
          <div className="mx-auto grid max-w-7xl gap-6 border-b border-slate-100 px-4 py-4 md:grid-cols-2">
            <div>
              <h2 className="mb-3 text-sm font-semibold text-slate-800">Advanced</h2>
              <TrainingSettings disabled={busy} />
            </div>
            <div>
              <h2 className="mb-3 text-sm font-semibold text-slate-800">Under the hood</h2>
              <TrainingHistory />
            </div>
          </div>
        )}
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:gap-4">
          <div className="min-w-0 flex-1">
            <TrainingProgress />
          </div>
          <div className="flex items-start gap-2">
            <Button variant="ghost" size="lg" onClick={() => setPanelOpen((v) => !v)} aria-expanded={panelOpen} title="Advanced settings and metrics">
              <SettingsIcon />
              <span className="hidden sm:inline">Advanced</span>
              <ChevronIcon size={16} className={`transition-transform ${panelOpen ? '' : 'rotate-180'}`} />
            </Button>
            <Button variant="secondary" size="lg" onClick={() => setExportOpen(true)} disabled={!trained || busy}>
              <DownloadIcon />
              <span className="hidden sm:inline">Export Model</span>
            </Button>
            <TrainButton className="flex-1 sm:flex-none" />
          </div>
        </div>
      </div>

      <ExportModal open={exportOpen} onClose={() => setExportOpen(false)} />
    </div>
  );
}
