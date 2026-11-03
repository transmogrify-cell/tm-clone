import type { HTMLAttributes, ReactNode } from 'react';

export function Card({ className = '', ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div className={`rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 ${className}`} {...rest} />;
}

export function CardHeader({ title, actions, className = '' }: { title: ReactNode; actions?: ReactNode; className?: string }) {
  return (
    <div className={`flex items-center justify-between gap-3 border-b border-slate-100 px-4 py-3 ${className}`}>
      <div className="min-w-0 flex-1 font-semibold text-slate-900">{title}</div>
      {actions && <div className="flex shrink-0 items-center gap-1">{actions}</div>}
    </div>
  );
}
