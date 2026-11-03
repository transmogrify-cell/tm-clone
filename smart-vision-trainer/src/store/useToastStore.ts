import { create } from 'zustand';

export type ToastKind = 'info' | 'success' | 'warning' | 'error';

export interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
}

interface ToastStore {
  toasts: Toast[];
  push(kind: ToastKind, message: string, durationMs?: number): void;
  dismiss(id: number): void;
}

let nextId = 1;

export const useToastStore = create<ToastStore>((set, get) => ({
  toasts: [],
  push(kind, message, durationMs = kind === 'error' ? 8000 : 4500) {
    // Collapse exact duplicates (e.g. the same camera error from two cards).
    if (get().toasts.some((t) => t.message === message)) return;
    const id = nextId++;
    set((s) => ({ toasts: [...s.toasts.slice(-3), { id, kind, message }] }));
    setTimeout(() => get().dismiss(id), durationMs);
  },
  dismiss(id) {
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
  },
}));

/** Callable from anywhere, including non-React code. */
export const toast = {
  info: (m: string) => useToastStore.getState().push('info', m),
  success: (m: string) => useToastStore.getState().push('success', m),
  warning: (m: string) => useToastStore.getState().push('warning', m),
  error: (m: string) => useToastStore.getState().push('error', m),
};
