import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { CAPTURE_FPS, MAX_SAMPLES_PER_CLASS } from '../config';
import { useProjectStore } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';
import { useCamera } from '../utils/camera';
import { videoFrameToSample } from '../utils/image';
import { Button } from './ui/Button';
import { SpinnerIcon, XIcon } from './ui/icons';

interface WebcamCaptureProps {
  classId: string;
  disabled?: boolean;
  onClose(): void;
}

/**
 * Live webcam preview with a "Hold to Record" button that captures frames at
 * CAPTURE_FPS while pressed (mouse, touch or Space/Enter), up to the per-class cap.
 */
export function WebcamCapture({ classId, disabled, onClose }: WebcamCaptureProps) {
  const { videoRef, status, error, retry } = useCamera(true);
  const addSamples = useProjectStore((s) => s.addSamples);
  const count = useProjectStore((s) => s.classes.find((c) => c.id === classId)?.samples.length ?? 0);
  const [recording, setRecording] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const stop = useCallback(() => {
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
    setRecording(false);
  }, []);

  const captureOnce = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !video.videoWidth) return;
    const added = addSamples(classId, [{ dataUrl: videoFrameToSample(video), source: 'webcam' }]);
    if (added === 0) {
      stop();
      toast.info(`This class has reached the ${MAX_SAMPLES_PER_CLASS}-sample limit.`);
    }
  }, [addSamples, classId, stop, videoRef]);

  const start = useCallback(() => {
    if (status !== 'ready' || disabled || timer.current) return;
    setRecording(true);
    captureOnce();
    timer.current = setInterval(captureOnce, 1000 / CAPTURE_FPS);
  }, [captureOnce, disabled, status]);

  // Stop on unmount, when the UI locks, or when the window loses focus mid-hold.
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
      // Keep receiving pointerup even if the finger/mouse slides off the button.
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* not capturable (e.g. synthetic pointer): recording still stops on pointerup/leave */
    }
    start();
  };
  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) {
      e.preventDefault();
      start();
    }
  };
  const onKeyUp = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key === ' ' || e.key === 'Enter') stop();
  };

  const full = count >= MAX_SAMPLES_PER_CLASS;

  return (
    <div className="flex flex-col gap-2">
      <div className="relative aspect-square overflow-hidden rounded-xl bg-slate-900">
        <video ref={videoRef} muted playsInline autoPlay className="h-full w-full -scale-x-100 object-cover" />
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
        disabled={disabled || status !== 'ready' || full}
        onPointerDown={onPointerDown}
        onPointerUp={stop}
        onPointerCancel={stop}
        onPointerLeave={stop}
        onLostPointerCapture={stop}
        onKeyDown={onKeyDown}
        onKeyUp={onKeyUp}
        onBlur={stop}
        onContextMenu={(e) => e.preventDefault()}
        style={{ touchAction: 'none' }}
        className={`h-10 w-full select-none rounded-lg text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 ${
          recording ? 'bg-red-600 text-white' : 'bg-brand-600 text-white hover:bg-brand-700'
        }`}
      >
        {full ? 'Sample limit reached' : recording ? 'Recording…' : 'Hold to Record'}
      </button>
    </div>
  );
}
