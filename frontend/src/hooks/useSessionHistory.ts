import { useCallback, useState } from "react";
import type { HistoryEntry } from "../types";

const STORAGE_KEY = "spectraguard.history";
const MAX_ENTRIES = 20;

function readStoredHistory(): HistoryEntry[] {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

export function useSessionHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>(readStoredHistory);

  const addEntry = useCallback((entry: HistoryEntry) => {
    setEntries((prev) => {
      const next = [entry, ...prev].slice(0, MAX_ENTRIES);
      try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        // sessionStorage lleno o no disponible (modo privado): el historial
        // sigue funcionando en memoria para esta sesión, solo no persiste.
      }
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setEntries([]);
    sessionStorage.removeItem(STORAGE_KEY);
  }, []);

  return { entries, addEntry, clear };
}
