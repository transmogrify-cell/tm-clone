import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { CLASS_COLORS, MAX_SAMPLES_PER_CLASS, MIN_SAMPLES_PER_CLASS } from '../config';
import { isBusy, useProjectStore } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';
import { fileToSample } from '../utils/image';
import type { ImageClass } from '../types';
import { SampleGallery } from './SampleGallery';
import { WebcamCapture } from './WebcamCapture';
import { Card } from './ui/Card';
import { CameraIcon, PencilIcon, SpinnerIcon, TrashIcon, UploadIcon } from './ui/icons';

interface ClassCardProps {
  cls: ImageClass;
  index: number;
  canDelete: boolean;
}

export function ClassCard({ cls, index, canDelete }: ClassCardProps) {
  const busy = useProjectStore((s) => isBusy(s.progress));
  const webcamOpen = useProjectStore((s) => s.activeWebcamClassId === cls.id);
  const { setActiveWebcam, renameClass, removeClass, removeSample, clearSamples, addSamples } = useProjectStore.getState();

  const [draft, setDraft] = useState(cls.name);
  const [importing, setImporting] = useState<{ done: number; total: number } | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const nameInput = useRef<HTMLInputElement>(null);
  const color = CLASS_COLORS[index % CLASS_COLORS.length];
  const locked = busy || importing !== null;

  useEffect(() => setDraft(cls.name), [cls.name]);

  const commitName = () => {
    const name = draft.trim();
    if (!name) setDraft(cls.name);
    else if (name !== cls.name) renameClass(cls.id, name);
  };

  /** Decode, crop and add uploaded/dropped images, in small chunks so the gallery fills progressively. */
  const importFiles = async (fileList: FileList | File[]) => {
    const all = Array.from(fileList);
    const images = all.filter((f) => f.type.startsWith('image/'));
    if (images.length < all.length) toast.warning(`Skipped ${all.length - images.length} file(s) that are not images.`);
    const room = MAX_SAMPLES_PER_CLASS - cls.samples.length;
    if (room <= 0) {
      toast.info(`"${cls.name}" already has ${MAX_SAMPLES_PER_CLASS} samples, the maximum.`);
      return;
    }
    if (images.length > room) toast.warning(`Only the first ${room} images were added (limit ${MAX_SAMPLES_PER_CLASS} per class).`);
    const queue = images.slice(0, room);
    if (queue.length === 0) return;

    setImporting({ done: 0, total: queue.length });
    let failed = 0;
    for (let i = 0; i < queue.length; i += 8) {
      const chunk = queue.slice(i, i + 8);
      const results = await Promise.allSettled(chunk.map(fileToSample));
      const ok = results.flatMap((r) => (r.status === 'fulfilled' ? [{ dataUrl: r.value, source: 'upload' as const }] : []));
      failed += results.length - ok.length;
      addSamples(cls.id, ok);
      setImporting({ done: Math.min(i + chunk.length, queue.length), total: queue.length });
    }
    setImporting(null);
    if (failed) toast.error(`${failed} image(s) could not be read.`);
  };

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) void importFiles(e.target.files);
    e.target.value = ''; // allow re-selecting the same files
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (!locked && e.dataTransfer.files.length) void importFiles(e.dataTransfer.files);
  };

  const onDelete = () => {
    if (cls.samples.length > 0 && !window.confirm(`Delete "${cls.name}" and its ${cls.samples.length} samples?`)) return;
    removeClass(cls.id);
  };

  const n = cls.samples.length;
  const enough = n >= MIN_SAMPLES_PER_CLASS;

  return (
    <Card
      className={`flex flex-col transition-shadow ${dragOver ? 'ring-2 ring-brand-500' : ''}`}
      onDragOver={(e) => {
        if (locked) return;
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      <div className="flex items-center gap-2 border-b border-slate-100 px-4 py-3">
        <span className="h-3 w-3 shrink-0 rounded-full" style={{ background: color }} aria-hidden="true" />
        <div className="group flex min-w-0 flex-1 items-center gap-1.5">
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
            className="min-w-0 flex-1 rounded-md bg-transparent px-1 py-0.5 font-semibold text-slate-900 outline-none ring-brand-500 hover:bg-slate-50 focus:bg-white focus:ring-2 disabled:hover:bg-transparent"
          />
          <button
            type="button"
            onClick={() => nameInput.current?.select()}
            disabled={busy}
            className="rounded p-1 text-slate-400 hover:text-slate-700 disabled:hidden"
            aria-label="Rename class"
          >
            <PencilIcon size={14} />
          </button>
        </div>
        <span className={`shrink-0 text-xs font-medium ${enough ? 'text-slate-500' : 'text-amber-600'}`} title={`Minimum ${MIN_SAMPLES_PER_CLASS} to train`}>
          {n} / {MAX_SAMPLES_PER_CLASS}
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
              aria-label={`Delete class ${cls.name}`}
            >
              <TrashIcon size={16} />
            </button>
          )}
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-3 p-4 sm:flex-row">
        {webcamOpen ? (
          <div className="w-full shrink-0 sm:w-48">
            <WebcamCapture classId={cls.id} disabled={busy} onClose={() => setActiveWebcam(null)} />
          </div>
        ) : (
          <div className="flex shrink-0 gap-2 sm:flex-col">
            <SourceButton icon={<CameraIcon />} label="Webcam" disabled={locked} onClick={() => setActiveWebcam(cls.id)} />
            <SourceButton
              icon={importing ? <SpinnerIcon /> : <UploadIcon />}
              label={importing ? `${importing.done}/${importing.total}` : 'Upload'}
              disabled={locked}
              onClick={() => fileInput.current?.click()}
            />
            <input ref={fileInput} type="file" accept="image/*" multiple hidden onChange={onFileChange} />
          </div>
        )}
        <div className="min-w-0 flex-1">
          <p className="mb-2 text-xs font-medium text-slate-500">
            {n === 0 ? 'Add image samples' : `${n} image sample${n === 1 ? '' : 's'}`}
            {!enough && n > 0 && <span className="text-amber-600"> · {MIN_SAMPLES_PER_CLASS - n} more to train</span>}
          </p>
          <SampleGallery samples={cls.samples} onDelete={(id) => removeSample(cls.id, id)} disabled={locked} />
        </div>
      </div>
    </Card>
  );
}

function SourceButton({ icon, label, onClick, disabled }: { icon: JSX.Element; label: string; onClick(): void; disabled?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="flex h-16 w-20 flex-col items-center justify-center gap-1 rounded-xl bg-brand-50 text-xs font-medium text-brand-700 transition-colors hover:bg-brand-100 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
    >
      {icon}
      {label}
    </button>
  );
}
