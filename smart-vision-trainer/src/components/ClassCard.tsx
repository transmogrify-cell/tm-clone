import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { MAX_SAMPLES_PER_CLASS } from '../config';
import { preprocessForClass } from '../ml/preprocessing';
import { classColor, isBusy, sampleQuality, useProjectStore, type Quality } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';
import { canvasToSample, fileToCanvas } from '../utils/image';
import type { ClassType, CustomClass, Sample } from '../types';
import { SampleGallery } from './SampleGallery';
import { WebcamCapture } from './WebcamCapture';
import { Card } from './ui/Card';
import { AlertIcon, BoxIcon, CameraIcon, CheckIcon, HelpIcon, PencilIcon, SpinnerIcon, TrashIcon, UploadIcon, UserIcon } from './ui/icons';

/**
 * Uploads go through the same preprocessing as the webcam (STEP 1): Person ->
 * exactly one face -> 20%-margin crop; Object -> letterbox; Unknown -> both.
 * Images that don't fit are skipped and summarised in one toast.
 */
async function importImages(cls: CustomClass, files: File[], onProgress: (done: number) => void): Promise<void> {
  const { addSamples } = useProjectStore.getState();
  const images = files.filter((f) => f.type.startsWith('image/'));
  const notImages = files.length - images.length;
  const room = MAX_SAMPLES_PER_CLASS - cls.samples.length;
  const queue = images.slice(0, Math.max(0, room));
  const tally = { added: 0, none: 0, multiple: 0, unreadable: 0 };
  let pending: Omit<Sample, 'id'>[] = [];

  for (let i = 0; i < queue.length; i++) {
    try {
      const image = await fileToCanvas(queue[i]);
      const result = await preprocessForClass(image, cls.kind);
      if (result.status === 'ok') {
        pending.push(...result.samples.map((s) => ({ dataUrl: canvasToSample(s.canvas), view: s.view, source: 'upload' as const })));
      } else tally[result.status]++;
    } catch {
      tally.unreadable++;
    }
    // Add in small chunks so the gallery fills while the rest are processed.
    if (pending.length >= 6 || i === queue.length - 1) {
      tally.added += addSamples(cls.id, pending);
      pending = [];
    }
    onProgress(i + 1);
  }

  const skipped = [
    tally.none && `${tally.none} no face found`,
    tally.multiple && `${tally.multiple} multiple faces`,
    tally.unreadable && `${tally.unreadable} unreadable`,
    notImages && `${notImages} not images`,
    images.length > queue.length && `${images.length - queue.length} over the ${MAX_SAMPLES_PER_CLASS} limit`,
  ].filter(Boolean);
  if (skipped.length) toast.warning(`Added ${tally.added} to ${cls.name}. Skipped: ${skipped.join(', ')}.`);
  else if (tally.added) toast.success(`Added ${tally.added} sample${tally.added === 1 ? '' : 's'} to ${cls.name}.`);
}

const BADGES: Record<Quality, { label: string; cls: string; icon: JSX.Element }> = {
  good: { label: 'Good', cls: 'bg-emerald-50 text-emerald-700 ring-emerald-200', icon: <CheckIcon size={12} /> },
  'needs-more': { label: 'Needs more', cls: 'bg-amber-50 text-amber-800 ring-amber-200', icon: <AlertIcon size={12} /> },
  'too-few': { label: 'Too few', cls: 'bg-red-50 text-red-700 ring-red-200', icon: <AlertIcon size={12} /> },
};

export function QualityBadge({ cls }: { cls: CustomClass }) {
  const { quality, min, good } = sampleQuality(cls);
  const b = BADGES[quality];
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${b.cls}`}
      title={`Minimum ${min} to train, ${good}+ recommended`}
    >
      {b.icon}
      {b.label}
    </span>
  );
}

const EMPTY_TEXT: Record<CustomClass['kind'], string> = {
  person: 'Hold to capture or drop photos. Only the face is kept (one face per photo).',
  object: 'Hold to capture or drop photos. The whole image is kept, letterboxed to a square.',
  unknown: 'Other people, other objects, empty scenes: anything that is none of your classes.',
};

export function ClassCard({ cls, canDelete }: { cls: CustomClass; canDelete: boolean }) {
  const busy = useProjectStore((s) => isBusy(s.progress));
  const webcamOpen = useProjectStore((s) => s.activeWebcamClassId === cls.id);
  const faceReady = useProjectStore((s) => s.models.face.status === 'ready');
  const color = useProjectStore((s) => classColor(s.classes, cls.id));
  const { setActiveWebcam, renameClass, removeClass, removeSample, clearSamples, setClassType } = useProjectStore.getState();
  const [draft, setDraft] = useState(cls.name);
  const [importing, setImporting] = useState<{ done: number; total: number } | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const nameInput = useRef<HTMLInputElement>(null);
  const isUnknown = cls.kind === 'unknown';
  const locked = busy || importing !== null;
  // Person (and Unknown, for its face crops) need BlazeFace; Object classes don't.
  const sourceReady = cls.kind === 'object' || faceReady;
  const { min } = sampleQuality(cls);
  const n = cls.samples.length;

  useEffect(() => setDraft(cls.name), [cls.name]);

  const commitName = () => {
    const name = draft.trim();
    if (!name) setDraft(cls.name);
    else if (name !== cls.name) renameClass(cls.id, name);
  };

  const runImport = async (list: FileList | File[]) => {
    if (!sourceReady) {
      toast.warning('The face detector is still loading.');
      return;
    }
    const files = Array.from(list);
    setImporting({ done: 0, total: files.length });
    try {
      await importImages(useProjectStore.getState().classes.find((c) => c.id === cls.id) ?? cls, files, (done) =>
        setImporting({ done, total: files.length }),
      );
    } finally {
      setImporting(null);
    }
  };

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) void runImport(e.target.files);
    e.target.value = '';
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (!locked && e.dataTransfer.files.length) void runImport(e.dataTransfer.files);
  };
  const onDelete = () => {
    if (n > 0 && !window.confirm(`Delete "${cls.name}" and its ${n} samples?`)) return;
    removeClass(cls.id);
  };
  const onType = (type: ClassType) => {
    if (type === cls.kind) return;
    if (n > 0 && !window.confirm(`Switching "${cls.name}" to ${type} changes its preprocessing, so its ${n} samples will be removed. Continue?`)) return;
    setClassType(cls.id, type);
  };

  return (
    <Card
      className={`flex flex-col ${isUnknown ? 'bg-slate-50' : ''} ${dragOver ? 'ring-2 ring-brand-500' : ''}`}
      onDragOver={(e) => {
        if (locked) return;
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      data-class-card={cls.name}
    >
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-100 px-4 py-3">
        <span className="h-3 w-3 shrink-0 rounded-full" style={{ background: color }} aria-hidden="true" />
        {isUnknown ? (
          <div className="flex min-w-0 flex-1 items-center gap-1.5">
            <span className="font-semibold text-slate-900">{cls.name}</span>
            <span className="text-slate-400" title="Teaches the model to answer 'Unknown' instead of force-matching." aria-label="About the Unknown class">
              <HelpIcon size={15} />
            </span>
          </div>
        ) : (
          <div className="flex min-w-0 flex-1 items-center gap-1.5">
            <input
              ref={nameInput}
              value={draft}
              disabled={busy}
              maxLength={40}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commitName}
              onKeyDown={(e) => {
                if (e.key === 'Enter') e.currentTarget.blur();
                if (e.key === 'Escape') {
                  setDraft(cls.name);
                  requestAnimationFrame(() => nameInput.current?.blur());
                }
              }}
              aria-label="Class name"
              className="min-w-0 flex-1 rounded-md bg-transparent px-1 py-0.5 font-semibold text-slate-900 outline-none ring-brand-500 hover:bg-slate-50 focus:bg-white focus:ring-2"
            />
            <button
              type="button"
              onClick={() => nameInput.current?.select()}
              disabled={busy}
              className="rounded p-1 text-slate-400 hover:text-slate-700 disabled:hidden"
              aria-label="Rename"
            >
              <PencilIcon size={14} />
            </button>
          </div>
        )}
        <QualityBadge cls={cls} />
        <span className="shrink-0 text-xs font-medium tabular-nums text-slate-500" title={`Minimum ${min} to train`}>
          {n}
        </span>
        <div className="flex shrink-0">
          {n > 0 && (
            <button
              type="button"
              onClick={() => window.confirm(`Remove all ${n} samples from "${cls.name}"?`) && clearSamples(cls.id)}
              disabled={locked}
              className="rounded px-1.5 py-1 text-xs font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-800 disabled:opacity-40"
            >
              Clear
            </button>
          )}
          {canDelete && (
            <button
              type="button"
              onClick={onDelete}
              disabled={locked}
              className="rounded p-1.5 text-slate-400 hover:bg-red-50 hover:text-red-600 disabled:opacity-40"
              aria-label={`Delete ${cls.name}`}
            >
              <TrashIcon size={16} />
            </button>
          )}
        </div>
      </div>

      {isUnknown ? (
        <p className="px-4 pt-3 text-xs leading-relaxed text-slate-500">
          Each image is kept as a full frame, plus a face crop when it shows one face, so both inference paths learn to say "Unknown". Aim
          for about 20% of your samples.
        </p>
      ) : (
        <div className="flex items-center gap-2 px-4 pt-3">
          <div className="inline-flex rounded-lg bg-slate-100 p-0.5" role="radiogroup" aria-label="Class type">
            <TypeButton active={cls.kind === 'person'} disabled={locked} onClick={() => onType('person')} testId="type-person">
              <UserIcon size={14} /> Person
            </TypeButton>
            <TypeButton active={cls.kind === 'object'} disabled={locked} onClick={() => onType('object')} testId="type-object">
              <BoxIcon size={14} /> Object
            </TypeButton>
          </div>
          <span className="text-xs text-slate-500">{cls.kind === 'person' ? 'face crop · 20% margin' : 'full image · letterbox'}</span>
        </div>
      )}

      <div className="flex flex-1 flex-col gap-3 p-4 sm:flex-row">
        {webcamOpen ? (
          <div className="w-full shrink-0 sm:w-56">
            <WebcamCapture classId={cls.id} kind={cls.kind} disabled={busy} onClose={() => setActiveWebcam(null)} />
          </div>
        ) : (
          <div className="flex shrink-0 gap-2 sm:flex-col">
            <SourceButton icon={<CameraIcon />} label="Webcam" disabled={locked || !sourceReady} onClick={() => setActiveWebcam(cls.id)} />
            <SourceButton
              icon={importing ? <SpinnerIcon /> : <UploadIcon />}
              label={importing ? `${importing.done}/${importing.total}` : 'Upload'}
              disabled={locked || !sourceReady}
              onClick={() => fileInput.current?.click()}
            />
            <input ref={fileInput} type="file" accept="image/*" multiple hidden onChange={onFileChange} data-upload-for={cls.name} />
          </div>
        )}
        <div className="min-w-0 flex-1">
          <p className="mb-2 text-xs font-medium text-slate-500">
            {n === 0 ? 'Samples appear here' : `${n} sample${n === 1 ? '' : 's'}`}
            {n > 0 && n < min && <span className="text-red-600"> · {min - n} more to train</span>}
          </p>
          <SampleGallery
            samples={cls.samples}
            onDelete={(id) => removeSample(cls.id, id)}
            disabled={locked}
            emptyText={EMPTY_TEXT[cls.kind]}
            showView={isUnknown}
          />
        </div>
      </div>
    </Card>
  );
}

function TypeButton({
  active,
  disabled,
  onClick,
  children,
  testId,
}: {
  active: boolean;
  disabled?: boolean;
  onClick(): void;
  children: React.ReactNode;
  testId: string;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      disabled={disabled}
      onClick={onClick}
      data-testid={testId}
      className={`inline-flex h-7 items-center gap-1 rounded-md px-2.5 text-xs font-semibold transition-colors disabled:cursor-not-allowed ${
        active ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-800'
      }`}
    >
      {children}
    </button>
  );
}

function SourceButton({ icon, label, onClick, disabled }: { icon: JSX.Element; label: string; onClick(): void; disabled?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="flex h-16 w-24 flex-col items-center justify-center gap-1 rounded-xl bg-brand-50 text-xs font-medium text-brand-700 transition-colors hover:bg-brand-100 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
    >
      {icon}
      {label}
    </button>
  );
}
