import type { HistoryEntry } from "../types";

interface HistoryListProps {
  entries: HistoryEntry[];
  onClear: () => void;
}

export function HistoryList({ entries, onClear }: HistoryListProps) {
  return (
    <aside className="history-panel">
      <div className="history-panel__header">
        <h2>Historial de esta sesión</h2>
        {entries.length > 0 && (
          <button type="button" className="history-panel__clear" onClick={onClear}>
            Limpiar
          </button>
        )}
      </div>

      {entries.length === 0 ? (
        <p className="history-panel__empty">Los análisis que hagas van a aparecer acá.</p>
      ) : (
        <ul className="history-panel__list">
          {entries.map((entry) => (
            <li key={entry.id} className="history-item">
              <img src={entry.previewDataUrl} alt="" className="history-item__thumb" />
              <div className="history-item__body">
                <span className={`history-item__label ${entry.label === "real" ? "is-good" : "is-critical"}`}>
                  {entry.label === "real" ? "Real" : "Generado por IA"}
                </span>
                <span className="history-item__meta">
                  {(entry.confidence * 100).toFixed(0)}% confianza ·{" "}
                  {new Date(entry.created_at).toLocaleTimeString()}
                </span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
