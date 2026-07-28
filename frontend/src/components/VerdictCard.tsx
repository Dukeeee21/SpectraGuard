import type { AnalysisResult } from "../types";
import { ConfidenceDial } from "./ConfidenceDial";

interface VerdictCardProps {
  result: AnalysisResult;
}

const VERDICT_COPY: Record<AnalysisResult["label"], { title: string; icon: string }> = {
  real: { title: "Rostro real", icon: "✓" },
  ai_generated: { title: "Generado por IA", icon: "⚠" },
};

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span className="metric__label">{label}</span>
      <span className="metric__value">{value}</span>
    </div>
  );
}

export function VerdictCard({ result }: VerdictCardProps) {
  const copy = VERDICT_COPY[result.label];
  const statusClass = result.label === "real" ? "is-good" : "is-critical";

  return (
    <section className={`verdict-card ${statusClass}`} aria-live="polite">
      <div className="verdict-card__headline">
        <span className={`verdict-badge ${statusClass}`}>
          <span aria-hidden="true">{copy.icon}</span>
          {copy.title}
        </span>
        <ConfidenceDial confidence={result.confidence} verdict={result.label} />
      </div>

      <dl className="verdict-card__metrics">
        <Metric label="P(generado por IA)" value={`${(result.ai_probability * 100).toFixed(1)}%`} />
        <Metric label="Confianza de detección de rostro" value={`${(result.face_confidence * 100).toFixed(1)}%`} />
        <Metric label="Tiempo de procesamiento" value={`${result.processing_time_ms.toFixed(0)} ms`} />
        <Metric label="Versión del modelo" value={result.model_version} />
      </dl>
    </section>
  );
}
