import { useCallback } from 'react';
import { loadMobileNet } from '../ml/mobilenet';
import { trainClassifier } from '../ml/trainer';
import { isBusy, trainingReadiness, useProjectStore } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';

/** Turns a tfjs failure into something a user can act on. */
export function describeTrainingError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  if (/out of memory|\boom\b|failed to allocate|allocation failed|context lost|exceeds the size limit/i.test(message)) {
    return 'Ran out of memory while training. Remove some samples or close other tabs, then try again.';
  }
  return `Training failed: ${message}`;
}

/**
 * Orchestrates a training run: feature extraction (with per-image progress),
 * then fitting the head (with per-epoch progress), writing everything to the
 * store as it goes. The UI is locked while `busy` is true.
 */
export function useTraining() {
  const classes = useProjectStore((s) => s.classes);
  const progress = useProjectStore((s) => s.progress);
  const modelReady = useProjectStore((s) => s.model.status === 'ready');
  const busy = isBusy(progress);
  const readiness = trainingReadiness(classes, modelReady);

  const train = useCallback(async () => {
    const state = useProjectStore.getState();
    if (isBusy(state.progress)) return;
    const check = trainingReadiness(state.classes, state.model.status === 'ready');
    if (!check.ok) {
      toast.warning(check.reason!);
      return;
    }

    const { setProgress, setTrained, setActiveWebcam } = state;
    // Snapshot: the UI is locked, but train on exactly what was there at the click.
    const classes = state.classes.map((c) => ({ ...c, samples: [...c.samples] }));
    const params = { ...state.params };
    const total = classes.reduce((n, c) => n + c.samples.length, 0);

    setActiveWebcam(null); // close card webcams; recording during training would race
    setProgress({
      phase: 'extracting',
      extracted: 0,
      extractTotal: total,
      epoch: 0,
      epochFraction: 0,
      totalEpochs: params.epochs,
      history: [],
      error: undefined,
    });

    try {
      const extractor = await loadMobileNet();
      const trained = await trainClassifier(classes, params, extractor, {
        onExtractProgress: (extracted, extractTotal) => setProgress({ extracted, extractTotal }),
        onTrainStart: () => setProgress({ phase: 'training' }),
        onBatchEnd: (epoch, epochFraction) => setProgress({ epoch, epochFraction }),
        onEpochEnd: (log) =>
          setProgress({
            epoch: log.epoch,
            epochFraction: 0,
            history: [...useProjectStore.getState().progress.history, log],
          }),
      });
      setTrained(trained);
      setProgress({ phase: 'done' });
      const acc = trained.finalMetrics?.valAcc ?? trained.finalMetrics?.acc;
      toast.success(`Model trained${acc !== undefined ? ` · ${(acc * 100).toFixed(0)}% accuracy` : ''}.`);
    } catch (err) {
      console.error(err);
      const message = describeTrainingError(err);
      setProgress({ phase: 'error', error: message });
      toast.error(message);
    }
  }, []);

  return { train, busy, progress, readiness };
}
