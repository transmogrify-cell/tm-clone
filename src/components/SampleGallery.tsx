import { memo } from 'react';
import { XIcon } from './ui/icons';
import type { Sample } from '../types';

interface SampleGalleryProps {
  samples: Sample[];
  onDelete(sampleId: string): void;
  disabled?: boolean;
  className?: string;
}

const Thumb = memo(function Thumb({
  sample,
  index,
  onDelete,
  disabled,
}: {
  sample: Sample;
  index: number;
  onDelete(id: string): void;
  disabled?: boolean;
}) {
  return (
    <div className="group relative aspect-square overflow-hidden rounded-md bg-slate-100 ring-1 ring-slate-200">
      <img src={sample.dataUrl} alt={`Sample ${index + 1}`} loading="lazy" decoding="async" className="h-full w-full object-cover" />
      {!disabled && (
        <button
          type="button"
          onClick={() => onDelete(sample.id)}
          aria-label={`Delete sample ${index + 1}`}
          className="absolute right-0.5 top-0.5 hidden rounded-full bg-slate-900/70 p-0.5 text-white hover:bg-red-600 focus:flex group-hover:flex"
        >
          <XIcon size={12} />
        </button>
      )}
    </div>
  );
});

/** Thumbnails of a class's samples, newest first, with delete on hover. */
export function SampleGallery({ samples, onDelete, disabled, className = '' }: SampleGalleryProps) {
  if (samples.length === 0) {
    return (
      <div className={`flex min-h-[5rem] items-center justify-center rounded-lg border border-dashed border-slate-200 px-3 text-center text-sm text-slate-400 ${className}`}>
        No samples yet. Record from the webcam, upload, or drop images here.
      </div>
    );
  }
  const newestFirst = [...samples].reverse();
  return (
    <div className={`grid max-h-56 grid-cols-[repeat(auto-fill,minmax(52px,1fr))] content-start gap-1.5 overflow-y-auto pr-1 ${className}`}>
      {newestFirst.map((s, i) => (
        <Thumb key={s.id} sample={s} index={samples.length - 1 - i} onDelete={onDelete} disabled={disabled} />
      ))}
    </div>
  );
}
