import { create } from 'zustand';
import { DEFAULT_TRAINING, MAX_CLASSES, MAX_SAMPLES_PER_CLASS, MIN_CLASSES, MIN_SAMPLES_PER_CLASS } from '../config';
import { clearFeatureCache, forgetFeatures } from '../ml/trainer';
import type { ImageClass, Sample, TrainedModel, TrainingParams, TrainingProgress } from '../types';

export type Screen = 'landing' | 'workspace';

export interface ModelStatus {
  status: 'idle' | 'loading' | 'ready' | 'error';
  progress: number;
  backend?: string;
  featureDim?: number;
  error?: string;
}

interface ProjectStore {
  screen: Screen;
  classes: ImageClass[];
  params: TrainingParams;
  progress: TrainingProgress;
  trained: TrainedModel | null;
  /** True when samples or classes changed after the last training run. */
  stale: boolean;
  /** Which class card currently has its webcam open (one at a time). */
  activeWebcamClassId: string | null;
  model: ModelStatus;

  goTo(screen: Screen): void;
  newProject(): void;
  addClass(): void;
  removeClass(id: string): void;
  renameClass(id: string, name: string): void;
  /** Returns how many samples were accepted (stops at MAX_SAMPLES_PER_CLASS). */
  addSamples(classId: string, samples: Omit<Sample, 'id'>[]): number;
  removeSample(classId: string, sampleId: string): void;
  clearSamples(classId: string): void;
  setParams(params: Partial<TrainingParams>): void;
  setProgress(progress: Partial<TrainingProgress>): void;
  setTrained(trained: TrainedModel | null): void;
  /** Starts a project around an imported model (classes with no samples). */
  loadTrained(trained: Omit<TrainedModel, 'classIds'>): void;
  setActiveWebcam(classId: string | null): void;
  setModelStatus(status: Partial<ModelStatus>): void;
}

export const uid = (): string =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;

const makeClass = (n: number): ImageClass => ({ id: uid(), name: `Class ${n}`, samples: [] });

const initialProgress = (): TrainingProgress => ({
  phase: 'idle',
  extracted: 0,
  extractTotal: 0,
  epoch: 0,
  epochFraction: 0,
  totalEpochs: DEFAULT_TRAINING.epochs,
  history: [],
});

export const useProjectStore = create<ProjectStore>((set, get) => ({
  screen: 'landing',
  classes: [makeClass(1), makeClass(2)],
  params: { ...DEFAULT_TRAINING },
  progress: initialProgress(),
  trained: null,
  stale: false,
  activeWebcamClassId: null,
  model: { status: 'idle', progress: 0 },

  goTo: (screen) => set({ screen }),

  newProject() {
    get().trained?.head.dispose();
    clearFeatureCache();
    set({
      classes: [makeClass(1), makeClass(2)],
      progress: initialProgress(),
      trained: null,
      stale: false,
      activeWebcamClassId: null,
    });
  },

  addClass() {
    const { classes } = get();
    if (classes.length >= MAX_CLASSES) return;
    // Pick the next "Class N" that isn't taken.
    let n = classes.length + 1;
    while (classes.some((c) => c.name === `Class ${n}`)) n++;
    set({ classes: [...classes, makeClass(n)], stale: get().trained !== null });
  },

  removeClass(id) {
    const cls = get().classes.find((c) => c.id === id);
    if (!cls) return;
    forgetFeatures(cls.samples.map((s) => s.id));
    set((s) => ({
      classes: s.classes.filter((c) => c.id !== id),
      activeWebcamClassId: s.activeWebcamClassId === id ? null : s.activeWebcamClassId,
      stale: s.trained !== null,
    }));
  },

  renameClass(id, name) {
    // Renaming does not invalidate the model: outputs are mapped by class id.
    set((s) => ({ classes: s.classes.map((c) => (c.id === id ? { ...c, name } : c)) }));
  },

  addSamples(classId, samples) {
    const cls = get().classes.find((c) => c.id === classId);
    if (!cls) return 0;
    const room = Math.max(0, MAX_SAMPLES_PER_CLASS - cls.samples.length);
    const accepted = samples.slice(0, room).map((s) => ({ ...s, id: uid() }));
    if (accepted.length === 0) return 0;
    set((s) => ({
      classes: s.classes.map((c) => (c.id === classId ? { ...c, samples: [...c.samples, ...accepted] } : c)),
      stale: s.trained !== null,
    }));
    return accepted.length;
  },

  removeSample(classId, sampleId) {
    forgetFeatures([sampleId]);
    set((s) => ({
      classes: s.classes.map((c) =>
        c.id === classId ? { ...c, samples: c.samples.filter((x) => x.id !== sampleId) } : c,
      ),
      stale: s.trained !== null,
    }));
  },

  clearSamples(classId) {
    const cls = get().classes.find((c) => c.id === classId);
    if (!cls || cls.samples.length === 0) return;
    forgetFeatures(cls.samples.map((s) => s.id));
    set((s) => ({
      classes: s.classes.map((c) => (c.id === classId ? { ...c, samples: [] } : c)),
      stale: s.trained !== null,
    }));
  },

  setParams: (params) => set((s) => ({ params: { ...s.params, ...params } })),

  setProgress: (progress) => set((s) => ({ progress: { ...s.progress, ...progress } })),

  setTrained(trained) {
    const previous = get().trained;
    set({ trained, stale: false });
    // Safe to dispose now: predictions run synchronously inside tf.tidy, so
    // nothing can be mid-way through using the old head at this point.
    if (previous && previous.head !== trained?.head) previous.head.dispose();
  },

  loadTrained(model) {
    get().trained?.head.dispose();
    clearFeatureCache();
    const classes = model.labels.map((name) => ({ id: uid(), name, samples: [] as Sample[] }));
    set({
      classes,
      trained: { ...model, classIds: classes.map((c) => c.id) },
      stale: false,
      progress: { ...initialProgress(), phase: 'done' },
      activeWebcamClassId: null,
      screen: 'workspace',
    });
  },

  setActiveWebcam: (activeWebcamClassId) => set({ activeWebcamClassId }),

  setModelStatus: (status) => set((s) => ({ model: { ...s.model, ...status } })),
}));

// ───────────────────────────── Selectors ─────────────────────────────

export const isBusy = (p: TrainingProgress): boolean => p.phase === 'extracting' || p.phase === 'training';

export interface Readiness {
  ok: boolean;
  reason?: string;
}

/** Why the Train button is disabled, if it is. */
export function trainingReadiness(classes: ImageClass[], modelReady: boolean): Readiness {
  if (classes.length < MIN_CLASSES) return { ok: false, reason: `Add at least ${MIN_CLASSES} classes.` };
  const unnamed = classes.find((c) => !c.name.trim());
  if (unnamed) return { ok: false, reason: 'Every class needs a name.' };
  const seen = new Set<string>();
  for (const c of classes) {
    const key = c.name.trim().toLowerCase();
    if (seen.has(key)) return { ok: false, reason: `Two classes are named "${c.name.trim()}".` };
    seen.add(key);
  }
  const short = classes.filter((c) => c.samples.length < MIN_SAMPLES_PER_CLASS);
  if (short.length > 0) {
    const names = short.map((c) => `"${c.name}" (${c.samples.length})`).join(', ');
    return { ok: false, reason: `Each class needs at least ${MIN_SAMPLES_PER_CLASS} samples: ${names}.` };
  }
  if (!modelReady) return { ok: false, reason: 'MobileNet is still loading.' };
  return { ok: true };
}

/**
 * Labels for the trained model's outputs, using current class names so a
 * rename after training shows up in the preview and in exports.
 */
export function resolveLabels(trained: TrainedModel, classes: ImageClass[]): string[] {
  return trained.classIds.map((id, i) => classes.find((c) => c.id === id)?.name.trim() || trained.labels[i]);
}
