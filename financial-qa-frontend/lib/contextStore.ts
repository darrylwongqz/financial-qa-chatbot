import { create } from 'zustand';
import { ContextItem } from '@/components/ui/ContextAccordion';

interface ContextState {
  contexts: ContextItem[] | null;
  setContexts: (contexts: ContextItem[] | null) => void;
  clearContexts: () => void;
}

export const useContextStore = create<ContextState>((set) => ({
  contexts: null,
  setContexts: (contexts: ContextItem[] | null) => set({ contexts }),
  clearContexts: () => set({ contexts: null }),
}));
