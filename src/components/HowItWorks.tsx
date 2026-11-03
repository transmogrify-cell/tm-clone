import { ChevronIcon } from './ui/icons';

/** Collapsible explainer: this is transfer learning, not training from scratch. */
export function HowItWorks({ defaultOpen = false }: { defaultOpen?: boolean }) {
  return (
    <details open={defaultOpen} className="group rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 [&_summary::-webkit-details-marker]:hidden">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 text-sm font-semibold text-slate-800">
        How it works: transfer learning in your browser
        <ChevronIcon size={16} className="text-slate-400 transition-transform group-open:rotate-180" />
      </summary>
      <div className="grid gap-4 border-t border-slate-100 px-4 py-4 text-sm leading-relaxed text-slate-600 md:grid-cols-3">
        <Step n={1} title="A pre-trained eye">
          <b>MobileNet V2</b> already learned to recognise visual patterns from 1.2 million ImageNet photos. We remove its final 1000-way
          classifier and keep the layer before it, which turns any 224×224 image into <b>1,280 numbers</b> describing what's in it. These
          weights are <b>frozen</b>: they never change.
        </Step>
        <Step n={2} title="A small, new brain">
          On top we train a tiny network (Dense 128 → Dropout → Softmax) that learns to map those 1,280 numbers to <b>your</b> classes. Each
          image goes through MobileNet once; then the head trains for a few epochs on the cached numbers, which takes seconds.
        </Step>
        <Step n={3} title="Private by design">
          Everything, including the camera, training and prediction, runs locally with TensorFlow.js on your GPU (WebGL). No image is uploaded. Exports contain only
          the small head plus labels; MobileNet is re-attached when the model is loaded.
        </Step>
        <p className="rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500 md:col-span-3">
          Why not train from scratch? A network that learns to see from zero needs thousands of images per class and hours of compute.
          Re-using MobileNet's features gets good results from 10–50 examples per class.
        </p>
      </div>
    </details>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1 flex items-center gap-2 font-semibold text-slate-800">
        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-brand-600 text-[11px] text-white">{n}</span>
        {title}
      </div>
      <p>{children}</p>
    </div>
  );
}
