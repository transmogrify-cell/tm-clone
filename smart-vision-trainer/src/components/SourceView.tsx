import { useRef, useState, type ChangeEvent, type DragEvent, type ReactNode, type RefObject } from 'react';
import type { useFrameSource } from '../hooks/useInference';
import { Button } from './ui/Button';
import { CameraIcon, ImageIcon, SpinnerIcon, UploadIcon } from './ui/icons';

type FrameSourceState = ReturnType<typeof useFrameSource>;

/**
 * The live camera (mirrored, selfie view) or an uploaded photo, with the
 * detection canvas stretched exactly over it. The container takes the
 * source's aspect ratio so canvas pixels map 1:1 onto the media.
 */
export function SourceView({
  src,
  overlayRef,
  chips,
  emptyHint = 'Upload a photo to run on a still image.',
}: {
  src: FrameSourceState;
  overlayRef: RefObject<HTMLCanvasElement>;
  chips?: ReactNode;
  emptyHint?: string;
}) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [videoSize, setVideoSize] = useState<{ w: number; h: number } | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const { mode, camera, image } = src;

  const aspect = mode === 'image' && image ? `${image.canvas.width} / ${image.canvas.height}` : videoSize ? `${videoSize.w} / ${videoSize.h}` : '4 / 3';

  const onFile = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) void src.loadFile(file);
    e.target.value = '';
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) void src.loadFile(file);
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="inline-flex rounded-lg bg-slate-200/70 p-0.5" role="group" aria-label="Input source">
          <SegButton active={mode === 'webcam'} onClick={() => void src.setMode('webcam')}>
            <CameraIcon size={16} /> Webcam
          </SegButton>
          <SegButton active={mode === 'image'} onClick={() => (image ? void src.setMode('image') : fileInput.current?.click())}>
            <ImageIcon size={16} /> Photo
          </SegButton>
        </div>
        <Button size="sm" onClick={() => fileInput.current?.click()}>
          <UploadIcon size={16} /> Upload image
        </Button>
        <input ref={fileInput} type="file" accept="image/*" hidden onChange={onFile} data-testid="source-upload" />
        {mode === 'image' && image && <span className="truncate text-xs text-slate-500">{image.name}</span>}
      </div>

      <div
        className={`relative w-full overflow-hidden rounded-2xl bg-slate-900 ${dragOver ? 'ring-4 ring-brand-500' : ''}`}
        style={{ aspectRatio: aspect }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        {mode === 'webcam' ? (
          <video
            ref={camera.videoRef}
            muted
            playsInline
            autoPlay
            onLoadedMetadata={(e) => setVideoSize({ w: e.currentTarget.videoWidth, h: e.currentTarget.videoHeight })}
            className="absolute inset-0 h-full w-full -scale-x-100 object-fill"
          />
        ) : image ? (
          <img src={image.url} alt={image.name} className="absolute inset-0 h-full w-full object-fill" />
        ) : null}
        <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true" />

        {mode === 'webcam' && camera.status === 'starting' && (
          <div className="absolute inset-0 flex items-center justify-center gap-2 text-sm text-slate-300">
            <SpinnerIcon /> Starting camera…
          </div>
        )}
        {mode === 'webcam' && camera.error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center text-sm text-slate-200">
            <p className="max-w-sm">{camera.error}</p>
            <div className="flex gap-2">
              {camera.status !== 'unsupported' && (
                <Button size="sm" onClick={camera.retry}>
                  Try again
                </Button>
              )}
              <Button size="sm" variant="primary" onClick={() => fileInput.current?.click()}>
                Use a photo instead
              </Button>
            </div>
          </div>
        )}
        {mode === 'image' && !image && (
          <div className="absolute inset-0 flex items-center justify-center p-6 text-center text-sm text-slate-400">{emptyHint}</div>
        )}
        {chips && <div className="absolute left-2 top-2 flex flex-wrap gap-1.5">{chips}</div>}
      </div>
    </div>
  );
}

function SegButton({ active, onClick, children }: { active: boolean; onClick(): void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`inline-flex h-8 items-center gap-1.5 rounded-md px-3 text-sm font-medium transition-colors ${
        active ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
      }`}
    >
      {children}
    </button>
  );
}

export function Chip({ children, tone = 'dark' }: { children: ReactNode; tone?: 'dark' | 'ok' | 'warn' | 'brand' }) {
  const tones = {
    dark: 'bg-slate-900/75 text-white',
    ok: 'bg-emerald-600 text-white',
    warn: 'bg-amber-500 text-slate-950',
    brand: 'bg-brand-600 text-white',
  };
  return <span className={`rounded-full px-2 py-0.5 text-xs font-semibold tabular-nums backdrop-blur ${tones[tone]}`}>{children}</span>;
}
