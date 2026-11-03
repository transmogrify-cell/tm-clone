/**
 * Webcam access via WebRTC getUserMedia.
 *
 * Several components can show the camera at once (a class card and the
 * preview panel), so a single MediaStream is shared and reference-counted.
 * The camera is switched off shortly after the last viewer goes away.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { toast } from '../store/useToastStore';

export type CameraStatus = 'idle' | 'starting' | 'ready' | 'denied' | 'not-found' | 'in-use' | 'unsupported' | 'error';

let stream: MediaStream | null = null;
let pending: Promise<MediaStream> | null = null;
let refs = 0;
let stopTimer: ReturnType<typeof setTimeout> | undefined;

function isLive(s: MediaStream | null): s is MediaStream {
  return !!s && s.getVideoTracks().some((t) => t.readyState === 'live');
}

function acquireStream(): Promise<MediaStream> {
  refs++;
  clearTimeout(stopTimer);
  if (isLive(stream)) return Promise.resolve(stream);
  pending ??= navigator.mediaDevices
    .getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    })
    .then((s) => (stream = s))
    .finally(() => {
      pending = null;
    });
  return pending.catch((err) => {
    refs = Math.max(0, refs - 1);
    throw err;
  });
}

function releaseStream(): void {
  refs = Math.max(0, refs - 1);
  if (refs > 0) return;
  // Short grace period so switching between cards doesn't flicker the camera light.
  stopTimer = setTimeout(() => {
    if (refs === 0 && stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
  }, 1500);
}

/** Returns a reason the camera can't work here at all, or null. */
export function cameraUnsupportedReason(): string | null {
  if (typeof window !== 'undefined' && !window.isSecureContext) {
    return 'The camera needs a secure page: open the app on http://localhost or over HTTPS.';
  }
  if (!navigator.mediaDevices?.getUserMedia) return 'This browser does not support webcam access.';
  return null;
}

export function describeCameraError(err: unknown): { status: CameraStatus; message: string } {
  const name = err instanceof DOMException || err instanceof Error ? err.name : '';
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
    case 'SecurityError':
      return {
        status: 'denied',
        message: 'Camera permission was denied. Allow camera access in the address bar, then try again.',
      };
    case 'NotFoundError':
    case 'DevicesNotFoundError':
    case 'OverconstrainedError':
      return { status: 'not-found', message: 'No camera was found on this device.' };
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return { status: 'in-use', message: 'The camera is in use by another app or could not be started.' };
    default:
      return { status: 'error', message: `Could not start the camera${err instanceof Error ? `: ${err.message}` : '.'}` };
  }
}

/**
 * Attach the shared webcam stream to a <video> while `active` is true.
 *
 *   const { videoRef, status, error, retry } = useCamera(open);
 *   <video ref={videoRef} muted playsInline />
 */
export function useCamera(active: boolean) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [status, setStatus] = useState<CameraStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!active) {
      setStatus('idle');
      setError(null);
      return;
    }
    const unsupported = cameraUnsupportedReason();
    if (unsupported) {
      setStatus('unsupported');
      setError(unsupported);
      return;
    }

    let cancelled = false;
    let acquired = false;
    let track: MediaStreamTrack | undefined;
    const video = videoRef.current;
    const onEnded = () => {
      setStatus('error');
      setError('The camera was disconnected.');
    };
    setStatus('starting');
    setError(null);

    // acquireStream() takes a reference immediately (and gives it back itself
    // on failure). Exactly one releaseStream() must follow a success: here if
    // we were cancelled while waiting, otherwise in the cleanup below.
    acquireStream()
      .then(async (s) => {
        if (cancelled) {
          releaseStream();
          return;
        }
        acquired = true;
        if (!video) return;
        track = s.getVideoTracks()[0];
        track?.addEventListener('ended', onEnded);
        video.srcObject = s;
        try {
          await video.play();
        } catch {
          /* autoplay of a muted video is allowed; ignore spurious aborts */
        }
        if (!cancelled) setStatus('ready');
      })
      .catch((err) => {
        if (cancelled) return;
        const { status: st, message } = describeCameraError(err);
        setStatus(st);
        setError(message);
        toast.error(message);
      });

    return () => {
      cancelled = true;
      if (video) video.srcObject = null;
      track?.removeEventListener('ended', onEnded);
      if (acquired) releaseStream();
    };
  }, [active, attempt]);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  return { videoRef, status, error, retry };
}
