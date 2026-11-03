/**
 * Mode B STEP 5: post-training evaluation on the held-out validation samples.
 *
 * Confusion matrix: rows are the true class, columns the prediction.
 *   precision(c) = matrix[c][c] / column sum   "when it says Bottle, how often is it a bottle?"
 *   recall(c)    = matrix[c][c] / row sum      "of the bottle samples, how many did it get?"
 * Overall accuracy = diagonal / total.
 */
import { WEAK_CLASS_THRESHOLD } from '../config';
import type { ClassKind, ClassMetrics, Evaluation } from '../types';

export function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/** confusion[actual][predicted] = count. */
export function confusionMatrix(actual: number[], predicted: number[], numClasses: number): number[][] {
  const m = Array.from({ length: numClasses }, () => new Array<number>(numClasses).fill(0));
  actual.forEach((a, i) => m[a][predicted[i]]++);
  return m;
}

export function computePerClassMetrics(
  confusion: number[][],
  labels: string[],
  kinds: ClassKind[],
  totals: number[],
): ClassMetrics[] {
  return labels.map((label, c) => {
    const tp = confusion[c][c];
    const rowSum = confusion[c].reduce((a, b) => a + b, 0);
    const colSum = confusion.reduce((a, row) => a + row[c], 0);
    const precision = colSum > 0 ? tp / colSum : NaN;
    const recall = rowSum > 0 ? tp / rowSum : NaN;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : NaN;
    return { label, kind: kinds[c], precision, recall, f1, support: rowSum, samples: totals[c] };
  });
}

export function evaluate(
  probs: number[][],
  actual: number[],
  labels: string[],
  kinds: ClassKind[],
  totals: number[],
): Evaluation {
  const predicted = probs.map(argmax);
  const confusion = confusionMatrix(actual, predicted, labels.length);
  const correct = predicted.filter((p, i) => p === actual[i]).length;
  return {
    labels,
    overallAccuracy: actual.length ? correct / actual.length : NaN,
    perClass: computePerClassMetrics(confusion, labels, kinds, totals),
    confusion,
    valProbs: probs,
    valTrue: actual,
  };
}

export interface ThresholdStats {
  /** Fraction of validation samples that clear the threshold (get a confident answer). */
  coverage: number;
  /** Accuracy among those confident answers. */
  confidentAccuracy: number;
  /** Validation samples of real (non-Unknown) classes that the gate turned into "Unknown". */
  rejected: number;
}

/** What the confidence gate would do to the validation samples at a given threshold. */
export function thresholdStats(evaluation: Evaluation, threshold: number, kinds: ClassKind[]): ThresholdStats {
  let confident = 0;
  let confidentCorrect = 0;
  let rejected = 0;
  evaluation.valProbs.forEach((p, i) => {
    const top = argmax(p);
    if (p[top] >= threshold) {
      confident++;
      if (top === evaluation.valTrue[i]) confidentCorrect++;
    } else if (kinds[evaluation.valTrue[i]] !== 'unknown') {
      rejected++;
    }
  });
  const n = evaluation.valProbs.length;
  return {
    coverage: n ? confident / n : NaN,
    confidentAccuracy: confident ? confidentCorrect / confident : NaN,
    rejected,
  };
}

export interface WeakClass {
  label: string;
  message: string;
}

/** Classes whose recall (or precision) fell below 70%, with a concrete suggestion. */
export function weakClasses(evaluation: Evaluation): WeakClass[] {
  const out: WeakClass[] = [];
  for (const m of evaluation.perClass) {
    if (m.support === 0) continue;
    const lowRecall = m.recall < WEAK_CLASS_THRESHOLD;
    const lowPrecision = !Number.isNaN(m.precision) && m.precision < WEAK_CLASS_THRESHOLD;
    if (!lowRecall && !lowPrecision) continue;
    const what =
      m.kind === 'unknown'
        ? 'more "everything else" samples (other people, other objects, empty scenes)'
        : m.kind === 'person'
          ? `more photos of ${m.label}, from different angles and lighting`
          : `more photos of ${m.label}, from different angles, distances and backgrounds`;
    const why = lowRecall
      ? `recall is ${(m.recall * 100).toFixed(0)}% (below 70%): only ${Math.round(m.recall * m.support)} of ${m.support} test samples were recognised`
      : `precision is ${(m.precision * 100).toFixed(0)}%: ${(100 - m.precision * 100).toFixed(0)}% of "${m.label}" predictions were another class`;
    out.push({ label: m.label, message: `${m.label}: ${why}. Add ${what}.` });
  }
  return out;
}
