import { useState } from 'react';
import { CheckIcon, CopyIcon } from './icons';

/** Monospace code with a Copy button. */
export function CodeBlock({ code, label, maxHeight = 320 }: { code: string; label?: string; maxHeight?: number }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
    } catch {
      // Clipboard API can be blocked (permissions, insecure context): fall back to a hidden textarea.
      const ta = document.createElement('textarea');
      ta.value = code;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="overflow-hidden rounded-xl bg-slate-900 ring-1 ring-slate-800">
      <div className="flex items-center justify-between border-b border-slate-800 px-3 py-1.5">
        <span className="font-mono text-xs text-slate-400">{label}</span>
        <button
          type="button"
          onClick={copy}
          className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium text-slate-300 hover:bg-slate-800 hover:text-white"
        >
          {copied ? <CheckIcon size={14} /> : <CopyIcon size={14} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="overflow-auto p-3 font-mono text-[12.5px] leading-relaxed text-slate-100" style={{ maxHeight }}>
        <code>{code}</code>
      </pre>
    </div>
  );
}
