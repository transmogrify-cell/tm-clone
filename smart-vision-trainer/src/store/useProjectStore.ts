import { create } from 'zustand';
import {
  CLASS_COLORS,
  DEFAULT_THRESHOLD,
  DEFAULT_TRAINING,
  GOOD_SAMPLES,
  GOOD_SAMPLES_UNKNOWN,
  MAX_CLASSES,
  MAX_SAMPLES_PER_CLASS,
  MIN_SAMPLES,
  MIN_SAMPLES_UNKNOWN,
  UNKNOWN_COLOR,
  UNKNOWN_LABEL,
} from '../config';
import { clearFeatureCache, forgetFeatures } from '../ml/trainer';
import type { ClassType, CustomClass, DetectorId, Sample, TrainedModel, TrainingParams, TrainingProgress } from '../types';

export type Screen = 'landing' | 'app';
export type Tab = 'detect' | 'train' | 'preview' | 'export';
export type TrainView = 'classes' | 'progress' | 'report';
export type ModelKey = DetectorId | 'mobilenet';

export interface ModelStatus {
  status: 'idle' | 'loading' | 'ready' | 'error';
  error?: string;
}

/** Which Mode A detectors run, plus whether the Detect tab also shows the custom model's answer. */
export type Toggles = Record<DetectorId | 'custom', boolean>;

interface ProjectStore {
  screen: Screen;
  tab: Tab;
  trainView: TrainView;
  models: Record<ModelKey, ModelStatus>;
  backend?: string;
  enabled: Toggles;

  classes: CustomClass[];
  params: TrainingParams;
  progress: TrainingProgress;
  trained: TrainedModel | null;
  /** Classes or samples changed since the last training run. */
  stale: boolean;
  threshold: number;
  activeWebcamClassId: string | null;
  /** Preview tab: Mode A localizes, Mode B labels each box. */
  hybrid: boolean;

  start(): void;
  setTab(tab: Tab): void;
  setTrainView(view: TrainView): void;
  setModelStatus(key: ModelKey, status: ModelStatus): void;
  setBackend(backend: string): void;
  setEnabled(key: keyof Toggles, on: boolean): void;

  addClass(type: ClassType): void;
  addUnknownClass(): void;
  removeClass(id: string): void;
  renameClass(id: string, name: string): void;
  /** Switching type changes the preprocessing, so existing samples are dropped. */
  setClassType(id: string, type: ClassType): void;
  addSamples(classId: string, samples: Omit<Sample, 'id'>[]): number;
  removeSample(classId: string, sampleId: string): void;
  clearSamples(classId: string): void;
  resetProject(): void;

  setParams(params: Partial<TrainingParams>): void;
  setProgress(progress: Partial<TrainingProgress>): void;
  setTrained(trained: TrainedModel | null): void;
  setThreshold(threshold: number): void;
  setActiveWebcam(classId: string | null): void;
  setHybrid(on: boolean): void;
}

export const uid = (): string =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;

const makeClass = (name: string, kind: ClassType): CustomClass => ({ id: uid(), name, kind, samples: [] });
const defaultClasses = () => [makeClass('Class 1', 'object'), makeClass('Class 2', 'object')];

const initialProgress = (): TrainingProgress => ({
  phase: 'idle',
  extracted: 0,
  extractTotal: 0,
  samples: 0,
  epoch: 0,
  epochFraction: 0,
  totalEpochs: DEFAULT_TRAINING.epochs,
  history: [],
});

const idle: ModelStatus = { status: 'idle' };

export const useProjectStore = create<ProjectStore>((set, get) => ({
  screen: 'landing',
  tab: 'detect',
  trainView: 'classes',
  models: { coco: idle, face: idle, hands: idle, mobilenet: idle },
  enabled: { coco: true, face: true, hands: true, custom: true },

  classes: defaultClasses(),
  params: { ...DEFAULT_TRAINING },
  progress: initialProgress(),
  trained: null,
  stale: false,
  threshold: DEFAULT_THRESHOLD,
  activeWebcamClassId: null,
  hybrid: false,

  start: () => set({ screen: 'app' }),
  setTab: (tab) => set((s) => ({ tab, activeWebcamClassId: tab === 'train' ? s.activeWebcamClassId : null })),
  setTrainView: (trainView) => set({ trainView }),
  setModelStatus: (key, status) => set((s) => ({ models: { ...s.models, [key]: status } })),
  setBackend: (backend) => set({ backend }),
  setEnabled: (key, on) => set((s) => ({ enabled: { ...s.enabled, [key]: on } })),

  addClass(type) {
    const { classes, trained } = get();
    const regular = classes.filter((c) => c.kind !== 'unknown');
    if (regular.length >= MAX_CLASSES) return;
    let n = regular.length + 1;
    while (classes.some((c) => c.name === `Class ${n}`)) n++;
    // Keep "Unknown" last.
    const unknown = classes.filter((c) => c.kind === 'unknown');
    set({ classes: [...regular, makeClass(`Class ${n}`, type), ...unknown], stale: trained !== null });
  },

  addUnknownClass() {
    const { classes, trained } = get();
    if (classes.some((c) => c.kind === 'unknown')) return;
    set({ classes: [...classes, { id: uid(), name: UNKNOWN_LABEL, kind: 'unknown', samples: [] }], stale: trained !== null });
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
    set((s) => ({ classes: s.classes.map((c) => (c.id === id && c.kind !== 'unknown' ? { ...c, name } : c)) }));
  },

  setClassType(id, type) {
    const cls = get().classes.find((c) => c.id === id);
    if (!cls || cls.kind === 'unknown' || cls.kind === type) return;
    forgetFeatures(cls.samples.map((s) => s.id));
    set((s) => ({
      classes: s.classes.map((c) => (c.id === id ? { ...c, kind: type, samples: [] } : c)),
      stale: s.trained !== null,
    }));
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
      classes: s.classes.map((c) => (c.id === classId ? { ...c, samples: c.samples.filter((x) => x.id !== sampleId) } : c)),
      stale: s.trained !== null,
    }));
  },

  clearSamples(classId) {
    const cls = get().classes.find((c) => c.id === classId);
    if (!cls?.samples.length) return;
    forgetFeatures(cls.samples.map((s) => s.id));
    set((s) => ({
      classes: s.classes.map((c) => (c.id === classId ? { ...c, samples: [] } : c)),
      stale: s.trained !== null,
    }));
  },

  resetProject() {
    get().trained?.head.dispose();
    clearFeatureCache();
    set({
      classes: defaultClasses(),
      progress: initialProgress(),
      trained: null,
      stale: false,
      activeWebcamClassId: null,
      trainView: 'classes',
      hybrid: false,
    });
  },

  setParams: (params) => set((s) => ({ params: { ...s.params, ...params } })),
  setProgress: (progress) => set((s) => ({ progress: { ...s.progress, ...progress } })),

  setTrained(trained) {
    const previous = get().trained;
    set({ trained, stale: false });
    // Predictions run inside tf.tidy and read the head synchronously, so nothing is mid-way through the old one.
    if (previous && previous.head !== trained?.head) previous.head.dispose();
  },

  setThreshold: (threshold) => set({ threshold }),
  setActiveWebcam: (activeWebcamClassId) => set({ activeWebcamClassId }),
  setHybrid: (hybrid) => set({ hybrid }),
}));

// ───────────────────────────── Selectors ─────────────────────────────

export const isBusy = (p: TrainingProgress): boolean =>
  p.phase === 'extracting' || p.phase === 'training' || p.phase === 'evaluating';

export type Quality = 'good' | 'needs-more' | 'too-few';

export function sampleQuality(cls: CustomClass): { quality: Quality; min: number; good: number } {
  const min = cls.kind === 'unknown' ? MIN_SAMPLES_UNKNOWN : MIN_SAMPLES;
  const good = cls.kind === 'unknown' ? GOOD_SAMPLES_UNKNOWN : GOOD_SAMPLES;
  const n = cls.samples.length;
  return { quality: n >= good ? 'good' : n >= min ? 'needs-more' : 'too-few', min, good };
}

export interface Readiness {
  ok: boolean;
  reason?: string;
}

/** STEP 7: Train stays disabled until every class has its minimum (20, or 10 for Unknown). */
export function trainingReadiness(classes: CustomClass[], mobilenetReady: boolean): Readiness {
  const regular = classes.filter((c) => c.kind !== 'unknown');
  if (regular.length === 0) return { ok: false, reason: 'Add at least one class.' };
  if (classes.length < 2) return { ok: false, reason: 'Add a second class or the Unknown class.' };
  const seen = new Set<string>();
  for (const c of regular) {
    const key = c.name.trim().toLowerCase();
    if (!key) return { ok: false, reason: 'Every class needs a name.' };
    if (key === UNKNOWN_LABEL.toLowerCase()) return { ok: false, reason: `"${UNKNOWN_LABEL}" is reserved; use the Unknown card.` };
    if (seen.has(key)) return { ok: false, reason: `Two classes are named "${c.name.trim()}".` };
    seen.add(key);
  }
  const short = classes.filter((c) => sampleQuality(c).quality === 'too-few');
  if (short.length) {
    const list = short.map((c) => `${c.name} ${c.samples.length}/${sampleQuality(c).min}`).join(', ');
    return { ok: false, reason: `Need ${MIN_SAMPLES}+ samples per class and ${MIN_SAMPLES_UNKNOWN}+ for Unknown (${list}).` };
  }
  if (!mobilenetReady) return { ok: false, reason: 'MobileNet is still loading.' };
  return { ok: true };
}

/** Current names for the trained outputs (renames after training show up). */
export function resolveLabels(trained: TrainedModel, classes: CustomClass[]): string[] {
  return trained.classIds.map((id, i) => classes.find((c) => c.id === id)?.name.trim() || trained.labels[i]);
}

/** Identity color: classes take palette slots in order; Unknown is neutral gray. */
export function classColor(classes: CustomClass[], id: string): string {
  const cls = classes.find((c) => c.id === id);
  if (!cls || cls.kind === 'unknown') return UNKNOWN_COLOR;
  const slot = classes.filter((c) => c.kind !== 'unknown').findIndex((c) => c.id === id);
  return CLASS_COLORS[slot % CLASS_COLORS.length];
}

/** Color for a trained output index (works after classes were deleted, too). */
export function outputColor(trained: TrainedModel, classes: CustomClass[], index: number): string {
  if (trained.kinds[index] === 'unknown') return UNKNOWN_COLOR;
  if (classes.some((c) => c.id === trained.classIds[index])) return classColor(classes, trained.classIds[index]);
  return CLASS_COLORS[index % CLASS_COLORS.length];
}
