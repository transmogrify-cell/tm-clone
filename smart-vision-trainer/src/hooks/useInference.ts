import { useCallback, useEffect, useRef, useState } from 'react';
import { DETECT_FPS } from '../config';
import { resetHandTracking } from '../ml/handDetector';
import { toast } from '../store/useToastStore';
import { useCamera } from '../utils/camera';
import { fileToCanvas, isVideoReady, snapshotVideo } from '../utils/image';

export type SourceMode = 'webcam' | 'image';

export interface FrameSource {
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
  /** Still image (hand tracking off, no mirroring) vs webcam frame. */
  isStatic: boolean;
}

/**
 * Where frames come from: the shared webcam, or one uploaded photo. Webcam
 * frames are snapshotted into a canvas first, so every detector and the
 * classifier see the exact same pixels.
 */
export function useFrameSource(active: boolean) {
  const [mode, setMode] = useState<SourceMode>('webcam');
  const camera = useCamera(active && mode === 'webcam');
  const [image, setImage] = useState<{ canvas: HTMLCanvasElement; name: string; url: string } | null>(null);
  const snapshot = useRef<HTMLCanvasElement | null>(null);

  const loadFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      toast.warning(`"${file.name}" is not an image.`);
      return;
    }
    try {
      const canvas = await fileToCanvas(file);
      await resetHandTracking();
      setImage({ canvas, name: file.name, url: canvas.toDataURL('image/jpeg', 0.9) });
      setMode('image');
    } catch {
      toast.error(`Could not read "${file.name}".`);
    }
  }, []);

  const switchMode = useCallback(async (next: SourceMode) => {
    await resetHandTracking();
    setMode(next);
  }, []);

  /** The current frame, or null when nothing is ready yet. */
  const grab = useCallback((): FrameSource | null => {
    if (mode === 'image') return image ? { canvas: image.canvas, width: image.canvas.width, height: image.canvas.height, isStatic: true } : null;
    const video = camera.videoRef.current;
    if (!isVideoReady(video)) return null;
    const canvas = snapshotVideo(video, (snapshot.current ??= document.createElement('canvas')));
    return { canvas, width: canvas.width, height: canvas.height, isStatic: false };
  }, [mode, image, camera.videoRef]);

  return { mode, setMode: switchMode, camera, image, loadFile, grab, mirror: mode === 'webcam' };
}

/**
 * Throttled frame loop: requestAnimationFrame + a timestamp check caps the
 * work at `fps`, and a frame is never started while the previous one is still
 * running (slow devices just get fewer frames). Returns the measured rate.
 */
export function useFrameLoop(active: boolean, step: () => Promise<void>, fps = DETECT_FPS): number {
  const [measured, setMeasured] = useState(0);
  const stepRef = useRef(step);
  stepRef.current = step;

  useEffect(() => {
    if (!active) {
      setMeasured(0);
      return;
    }
    let stopped = false;
    let raf = 0;
    let busy = false;
    let last = 0;
    let frames = 0;
    let since = performance.now();
    const interval = 1000 / fps;

    const tick = (now: number) => {
      if (stopped) return;
      raf = requestAnimationFrame(tick);
      if (busy || now - last < interval) return;
      last = now;
      busy = true;
      stepRef
        .current()
        .catch((err) => console.warn('[frame] step failed', err))
        .finally(() => {
          busy = false;
          frames++;
          const t = performance.now();
          if (t - since >= 1000) {
            if (!stopped) setMeasured(Math.round((frames * 1000) / (t - since)));
            frames = 0;
            since = t;
          }
        });
    };
    raf = requestAnimationFrame(tick);
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
    };
  }, [active, fps]);

  return measured;
}
