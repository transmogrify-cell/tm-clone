import { useCallback } from 'react';
import { VARIANTS_PER_IMAGE } from '../ml/augmentation';
import { evaluate } from '../ml/evaluator';
import { loadMobilenet } from '../ml/mobilenet';
import { predictFeatures, trainHead } from '../ml/trainer';
import { isBusy, trainingReadiness, useProjectStore } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';

export function describeTrainingError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  if (/out of memory|\boom\b|failed to allocate|allocation failed|context lost|exceeds the size limit/i.test(message)) {
    return 'Ran out of memory while training. Remove some samples or close other tabs, then try again.';
  }
  return `Training failed: ${message}`;
}

// Set by "Stop" on the training view; read by the trainer after each batch.
let stopRequested = false;
export const requestStop = () => {
  stopRequested = true;
};

/**
 * Orchestrates a run: augment + extract features (per-image progress), train
 * the head with early stopping (per-epoch progress), then evaluate it on the
 * held-out samples and open the Accuracy Report.
 */
export function useTraining() {
  const classes = useProjectStore((s) => s.classes);
  const progress = useProjectStore((s) => s.progress);
  const mobilenetReady = useProjectStore((s) => s.models.mobilenet.status === 'ready');
  const busy = isBusy(progress);
  const readiness = trainingReadiness(classes, mobilenetReady);

  const train = useCallback(async () => {
    const state = useProjectStore.getState();
    if (isBusy(state.progress)) return;
    const check = trainingReadiness(state.classes, state.models.mobilenet.status === 'ready');
    if (!check.ok) {
      toast.warning(check.reason!);
      return;
    }
    const { setProgress, setTrained, setActiveWebcam, setTrainView, setTab } = state;
    // Snapshot exactly what was there at the click (the UI is locked anyway).
    const classes = state.classes.map((c) => ({ ...c, samples: [...c.samples] }));
    const params = { ...state.params };
    const samples = classes.reduce((n, c) => n + c.samples.length, 0);

    stopRequested = false;
    setActiveWebcam(null);
    setProgress({
      phase: 'extracting',
      extracted: 0,
      extractTotal: samples * VARIANTS_PER_IMAGE,
      samples,
      epoch: 0,
      epochFraction: 0,
      totalEpochs: params.epochs,
      history: [],
      bestEpoch: undefined,
      stoppedEarly: undefined,
      error: undefined,
    });
    setTab('train');
    setTrainView('progress');

    try {
      const extractor = await loadMobilenet();
      const result = await trainHead(classes, params, extractor, {
        onExtractProgress: (extracted, extractTotal) => setProgress({ extracted, extractTotal }),
        onTrainStart: () => setProgress({ phase: 'training' }),
        onBatchEnd: (epoch, epochFraction) => setProgress({ epoch, epochFraction }),
        onEpochEnd: (log) =>
          setProgress({ epoch: log.epoch, epochFraction: 0, history: [...useProjectStore.getState().progress.history, log] }),
        shouldStop: () => stopRequested,
      });

      // STEP 5: evaluate the restored best head on the held-out original samples.
      setProgress({ phase: 'evaluating' });
      const labels = classes.map((c) => c.name.trim());
      const kinds = classes.map((c) => c.kind);
      const probs = await predictFeatures(result.head, result.valFeatures, extractor.featureDim);
      const evaluation = result.valLabels.length
        ? evaluate(probs, result.valLabels, labels, kinds, classes.map((c) => c.samples.length))
        : null;

      setTrained({
        head: result.head,
        classIds: classes.map((c) => c.id),
        labels,
        kinds,
        featureDim: extractor.featureDim,
        trainedAt: Date.now(),
        sampleCount: result.sampleCount,
        evaluation,
        bestEpoch: result.bestEpoch,
        epochsRun: result.epochsRun,
      });
      setProgress({ phase: 'done', bestEpoch: result.bestEpoch, stoppedEarly: result.stoppedEarly || stopRequested });
      setTrainView('report');
    } catch (err) {
      console.error(err);
      const message = describeTrainingError(err);
      setProgress({ phase: 'error', error: message });
      toast.error(message);
    }
  }, []);

  return { train, busy, progress, readiness };
}
