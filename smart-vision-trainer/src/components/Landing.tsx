import { useProjectStore, type Tab } from '../store/useProjectStore';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { AlertIcon, ScanIcon, ShieldIcon, SparkIcon } from './ui/icons';

const LIMITS = [
  'Mode A (COCO-SSD) only detects the 80 COCO classes. A custom object ("my red mug") is not detected without Mode B.',
  'Mode B is classification, not detection: it says "this image is mostly X" but draws no box of its own. "Detect + Classify" in Preview borrows Mode A\'s boxes.',
  'Not for security or authentication. An educational / demo tool.',
];

export function Landing() {
  const models = useProjectStore((s) => s.models);
  const ready = Object.values(models).filter((m) => m.status === 'ready').length;
  const open = (tab: Tab) => {
    const s = useProjectStore.getState();
    s.setTab(tab);
    s.start();
  };

  return (
    <main className="mx-auto flex min-h-screen max-w-5xl flex-col justify-center gap-8 px-4 py-12">
      <div className="flex flex-col gap-3">
        <p className="inline-flex items-center gap-1.5 self-start rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700 ring-1 ring-emerald-200">
          <ShieldIcon size={14} /> 100% in your browser. No server, no cloud, images never leave this device.
        </p>
        <h1 className="text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl">Smart Vision Trainer</h1>
        <p className="max-w-2xl text-lg text-slate-600">
          Detect everyday objects, faces and hands instantly, then teach it your own classes with transfer learning, and see exactly how
          accurate it is.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card className="flex flex-col gap-3 p-6">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-red-50 text-red-600">
            <ScanIcon />
          </span>
          <h2 className="text-lg font-semibold text-slate-900">Mode A · Zero-shot detection</h2>
          <p className="text-sm text-slate-600">
            Works immediately. COCO-SSD boxes 80 object types (person, car, bicycle, motorcycle, bottle, cup, cell phone, chair, book, laptop…),
            BlazeFace finds faces, MediaPipe Hands tracks 21 keypoints per hand.
          </p>
          <Button variant="primary" className="mt-auto self-start" onClick={() => open('detect')} data-testid="start-detect">
            Start detecting
          </Button>
        </Card>
        <Card className="flex flex-col gap-3 p-6">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-50 text-brand-700">
            <SparkIcon />
          </span>
          <h2 className="text-lg font-semibold text-slate-900">Mode B · Custom training</h2>
          <p className="text-sm text-slate-600">
            Add classes like "Alice" (Person: face crops) or "MyBottle" (Object: whole image). A small head trains on frozen MobileNet V2
            features with 6× augmentation, then reports per-class precision, recall and a confusion matrix.
          </p>
          <Button className="mt-auto self-start" onClick={() => open('train')}>
            Train custom classes
          </Button>
        </Card>
      </div>

      <Card className="p-5">
        <h2 className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-900">
          <AlertIcon size={16} className="text-amber-600" /> Realistic limitations
        </h2>
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-600">
          {LIMITS.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </Card>

      <p className="text-xs text-slate-400" aria-live="polite">
        Pretrained models: {ready}/4 loaded from this site (public/models).
      </p>
    </main>
  );
}
