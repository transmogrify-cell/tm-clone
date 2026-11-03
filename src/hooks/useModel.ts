import { useCallback, useEffect } from 'react';
import { initBackend, loadMobileNet } from '../ml/mobilenet';
import { useProjectStore } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';

/**
 * Loads MobileNet once when the app mounts (so it is ready by the time the
 * user clicks Train) and mirrors its status into the store for the UI.
 */
export function useModel() {
  const model = useProjectStore((s) => s.model);
  const setModelStatus = useProjectStore((s) => s.setModelStatus);

  const load = useCallback(async () => {
    setModelStatus({ status: 'loading', progress: 0, error: undefined });
    try {
      const backend = await initBackend();
      setModelStatus({ backend });
      if (backend !== 'webgl') {
        toast.warning('WebGL is not available, so the CPU backend is used. Training and preview will be slower.');
      }
      const extractor = await loadMobileNet((progress) => setModelStatus({ progress }));
      setModelStatus({ status: 'ready', progress: 1, featureDim: extractor.featureDim });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setModelStatus({ status: 'error', error: message });
      toast.error(`Model failed to load. ${message}`);
    }
  }, [setModelStatus]);

  useEffect(() => {
    // StrictMode runs effects twice in dev; loadMobileNet() is memoized, so
    // only the status check is needed to avoid duplicate toasts.
    if (useProjectStore.getState().model.status === 'idle') void load();
  }, [load]);

  return { ...model, retry: load };
}
