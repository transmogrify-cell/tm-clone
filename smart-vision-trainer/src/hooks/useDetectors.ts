import { useCallback, useEffect } from 'react';
import { detectCoco, loadCocoSsd } from '../ml/cocoDetector';
import { detectFaces, loadFaceDetector, type Detectable } from '../ml/faceDetector';
import { detectHands, loadHandDetector } from '../ml/handDetector';
import { initBackend, loadMobilenet } from '../ml/mobilenet';
import { useProjectStore, type ModelKey, type ModelStatus, type Toggles } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';
import type { DetectionFrame } from '../types';

/**
 * Loads every pretrained model on app start. The three Mode A detectors load
 * in parallel (each is independent and small); MobileNet (Mode B features,
 * the largest download) follows so it doesn't delay the first detections.
 * A model that fails is reported on its own; the others keep working.
 */
export function useDetectors() {
  const models = useProjectStore((s) => s.models);
  const backend = useProjectStore((s) => s.backend);

  const load = useCallback(async () => {
    const { setModelStatus, setBackend } = useProjectStore.getState();
    const b = await initBackend();
    setBackend(b);
    if (b !== 'webgl') toast.warning('WebGL is not available, so the slow CPU backend is used. Detection will be well below 10 fps.');

    const track = async (key: ModelKey, run: () => Promise<unknown>) => {
      if (useProjectStore.getState().models[key].status === 'ready') return;
      setModelStatus(key, { status: 'loading' });
      try {
        await run();
        setModelStatus(key, { status: 'ready' });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setModelStatus(key, { status: 'error', error: message });
        toast.error(message);
      }
    };
    await Promise.all([track('coco', loadCocoSsd), track('face', loadFaceDetector), track('hands', loadHandDetector)]);
    await track('mobilenet', loadMobilenet);
  }, []);

  useEffect(() => {
    const { models } = useProjectStore.getState();
    if (Object.values(models).every((m) => m.status === 'idle')) void load();
  }, [load]);

  return { models, backend, retry: load };
}

/**
 * Mode A, one frame: every enabled + loaded detector runs in parallel
 * (Promise.all). A disabled or failed detector contributes an empty list.
 */
export async function runDetectors(
  source: Detectable,
  enabled: Toggles,
  models: Record<ModelKey, ModelStatus>,
  { width, height, isStatic }: { width: number; height: number; isStatic: boolean },
): Promise<DetectionFrame> {
  const on = (k: 'coco' | 'face' | 'hands') => enabled[k] && models[k].status === 'ready';
  const t0 = performance.now();
  const [coco, faces, hands] = await Promise.all([
    on('coco') ? detectCoco(source) : Promise.resolve([]),
    on('face') ? detectFaces(source) : Promise.resolve([]),
    on('hands') ? detectHands(source, isStatic) : Promise.resolve([]),
  ]);
  return { width, height, coco, faces, hands, ms: performance.now() - t0 };
}
