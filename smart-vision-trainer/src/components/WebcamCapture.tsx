import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { CAPTURE_FPS, FACE_COLOR, MAX_SAMPLES_PER_CLASS } from '../config';
import { detectFaces, significantFaces } from '../ml/faceDetector';
import { prepareSamples } from '../ml/preprocessing';
import { useProjectStore } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';
import { useCamera } from '../utils/camera';
import { clearOverlay, drawBoxes, prepareCanvas } from '../utils/canvasOverlay';
import { canvasToSample, isVideoReady, snapshotVideo } from '../utils/image';
import type { ClassKind } from '../types';
import { Button } from './ui/Button';
import { SpinnerIcon, XIcon } from './ui/icons';

/**
 * Live webcam with "Hold to capture". One loop at CAPTURE_FPS: snapshot the
 * frame and, while the button is held, preprocess it for this class's type:
 *  - Person: BlazeFace runs every frame (boxes drawn); exactly one face -> crop.
 *  - Object: the whole frame is letterboxed (no detector needed).
 *  - Unknown: letterboxed frame, plus the face crop when there is one face.
 */
export function WebcamCapture({ classId, kind, disabled, onClose }: { classId: string; kind: ClassKind; disabled?: boolean; onClose(): void }) {
  const { videoRef, status, error, retry } = useCamera(true);
  const addSamples = useProjectStore((s) => s.addSamples);
  const faceReady = useProjectStore((s) => s.models.face.status === 'ready');
  const count = useProjectStore((s) => s.classes.find((c) => c.id === classId)?.samples.length ?? 0);
  const overlay = useRef<HTMLCanvasElement>(null);
  const [recording, setRecording] = useState(false);
  const [faceCount, setFaceCount] = useState<number | null>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const recordingRef = useRef(false);
  const tally = useRef({ none: 0, multiple: 0, added: 0 });
  const needsFaces = kind !== 'object';

  const stop = useCallback(() => {
    if (!recordingRef.current) return;
    recordingRef.current = false;
    setRecording(false);
    const { none, multiple, added } = tally.current;
    const parts = [none && `${none} with no face found`, multiple && `${multiple} with multiple faces`].filter(Boolean);
    if (parts.length) toast.warning(`Captured ${added}. Skipped ${parts.join(' and ')}.`);
  }, []);

  const start = useCallback(() => {
    if (status !== 'ready' || disabled || recordingRef.current) return;
    tally.current = { none: 0, multiple: 0, added: 0 };
    recordingRef.current = true;
    setRecording(true);
  }, [disabled, status]);

  useEffect(() => {
    if (status !== 'ready') return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const frameCanvas = document.createElement('canvas');
    const loop = async () => {
      if (stopped) return;
      const t0 = performance.now();
      const video = videoRef.current;
      if (isVideoReady(video)) {
        try {
          const frame = snapshotVideo(video, frameCanvas);
          const faces = needsFaces && faceReady ? await detectFaces(frame) : [];
          if (stopped) return;
          if (needsFaces && overlay.current) {
            const main = significantFaces(faces);
            const ctx = prepareCanvas(overlay.current, frame.width, frame.height);
            drawBoxes(ctx, main.map((f) => ({ box: f.box, color: main.length === 1 ? FACE_COLOR : '#f59e0b' })), { mirror: true });
            setFaceCount(main.length);
          }
          if (recordingRef.current) {
            const result = prepareSamples(frame, kind, faces);
            if (result.status === 'ok') {
              const added = addSamples(
                classId,
                result.samples.map((s) => ({ dataUrl: canvasToSample(s.canvas), view: s.view, source: 'webcam' as const })),
              );
              tally.current.added += added;
              if (added === 0) {
                toast.info(`This class has reached the ${MAX_SAMPLES_PER_CLASS}-sample limit.`);
                stop();
              }
            } else tally.current[result.status]++;
          }
        } catch (err) {
          console.warn('[webcam] capture failed', err);
        }
      }
      timer = setTimeout(loop, Math.max(0, 1000 / CAPTURE_FPS - (performance.now() - t0)));
    };
    void loop();
    return () => {
      stopped = true;
      clearTimeout(timer);
      clearOverlay(overlay.current);
    };
  }, [status, videoRef, addSamples, classId, kind, needsFaces, faceReady, stop]);

  useEffect(() => stop, [stop]);
  useEffect(() => {
    if (disabled) stop();
  }, [disabled, stop]);
  useEffect(() => {
    window.addEventListener('blur', stop);
    return () => window.removeEventListener('blur', stop);
  }, [stop]);

  const onPointerDown = (e: PointerEvent<HTMLButtonElement>) => {
    if (e.button !== 0) return;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* synthetic pointer: pointerup/leave still stop recording */
    }
    start();
  };
  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) {
      e.preventDefault();
      start();
    }
  };

  const full = count >= MAX_SAMPLES_PER_CLASS;
  const chip =
    !needsFaces || status !== 'ready' || faceCount === null
      ? null
      : faceCount === 1
        ? { text: 'Face found', cls: 'bg-emerald-600' }
        : faceCount === 0
          ? { text: 'No face', cls: 'bg-slate-900/70' }
          : { text: `${faceCount} faces`, cls: 'bg-amber-500 text-slate-950' };

  return (
    <div className="flex flex-col gap-2">
      <div className="relative overflow-hidden rounded-xl bg-slate-900" style={{ aspectRatio: size ? `${size.w} / ${size.h}` : '4 / 3' }}>
        <video
          ref={videoRef}
          muted
          playsInline
          autoPlay
          onLoadedMetadata={(e) => setSize({ w: e.currentTarget.videoWidth, h: e.currentTarget.videoHeight })}
          className="h-full w-full -scale-x-100 object-fill"
        />
        <canvas ref={overlay} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true" />
        {status === 'starting' && (
          <div className="absolute inset-0 flex items-center justify-center gap-2 text-sm text-slate-300">
            <SpinnerIcon /> Starting camera…
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-4 text-center text-sm text-slate-200">
            <p>{error}</p>
            {status !== 'unsupported' && (
              <Button size="sm" onClick={retry}>
                Try again
              </Button>
            )}
          </div>
        )}
        {chip && <span className={`absolute bottom-2 left-2 rounded-full px-2 py-0.5 text-xs font-semibold text-white ${chip.cls}`}>{chip.text}</span>}
        {recording && (
          <span className="absolute left-2 top-2 flex items-center gap-1.5 rounded-full bg-red-600 px-2 py-0.5 text-xs font-semibold text-white">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white" /> REC
          </span>
        )}
        <button
          type="button"
          onClick={onClose}
          aria-label="Close webcam"
          className="absolute right-2 top-2 rounded-full bg-slate-900/60 p-1 text-white hover:bg-slate-900"
        >
          <XIcon size={16} />
        </button>
      </div>
      <button
        type="button"
        disabled={disabled || status !== 'ready' || full || (needsFaces && !faceReady)}
        onPointerDown={onPointerDown}
        onPointerUp={stop}
        onPointerCancel={stop}
        onPointerLeave={stop}
        onLostPointerCapture={stop}
        onKeyDown={onKeyDown}
        onKeyUp={(e) => (e.key === ' ' || e.key === 'Enter') && stop()}
        onBlur={stop}
        onContextMenu={(e) => e.preventDefault()}
        style={{ touchAction: 'none' }}
        className={`h-10 w-full select-none rounded-lg text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 ${
          recording ? 'bg-red-600 text-white' : 'bg-brand-600 text-white hover:bg-brand-700'
        }`}
      >
        {full ? 'Sample limit reached' : recording ? 'Capturing…' : 'Hold to capture'}
      </button>
    </div>
  );
}
