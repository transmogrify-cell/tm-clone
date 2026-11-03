import { useId, useRef, type KeyboardEvent, type ReactNode } from 'react';

export interface TabItem<T extends string> {
  id: T;
  label: ReactNode;
}

interface TabsProps<T extends string> {
  tabs: TabItem<T>[];
  value: T;
  onChange(id: T): void;
  children: ReactNode;
}

/** WAI-ARIA tabs: arrow keys move between tabs, the active panel is labelled by its tab. */
export function Tabs<T extends string>({ tabs, value, onChange, children }: TabsProps<T>) {
  const base = useId();
  const listRef = useRef<HTMLDivElement>(null);

  const onKeyDown = (e: KeyboardEvent) => {
    const i = tabs.findIndex((t) => t.id === value);
    const next = e.key === 'ArrowRight' ? i + 1 : e.key === 'ArrowLeft' ? i - 1 : null;
    if (next === null) return;
    e.preventDefault();
    const tab = tabs[(next + tabs.length) % tabs.length];
    onChange(tab.id);
    listRef.current?.querySelector<HTMLButtonElement>(`[data-tab="${tab.id}"]`)?.focus();
  };

  return (
    <div>
      <div ref={listRef} role="tablist" className="flex gap-1 overflow-x-auto border-b border-slate-200 px-5" onKeyDown={onKeyDown}>
        {tabs.map((t) => {
          const selected = t.id === value;
          return (
            <button
              key={t.id}
              type="button"
              role="tab"
              data-tab={t.id}
              id={`${base}-tab-${t.id}`}
              aria-selected={selected}
              aria-controls={`${base}-panel`}
              tabIndex={selected ? 0 : -1}
              onClick={() => onChange(t.id)}
              className={`-mb-px whitespace-nowrap border-b-2 px-3 py-3 text-sm font-medium transition-colors ${
                selected ? 'border-brand-600 text-brand-700' : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              {t.label}
            </button>
          );
        })}
      </div>
      <div role="tabpanel" id={`${base}-panel`} aria-labelledby={`${base}-tab-${value}`}>
        {children}
      </div>
    </div>
  );
}
