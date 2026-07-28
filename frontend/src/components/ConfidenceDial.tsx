import type { Verdict } from "../types";

interface ConfidenceDialProps {
  confidence: number; // [0, 1]
  verdict: Verdict;
}

const RADIUS = 54;
const STROKE = 10;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export function ConfidenceDial({ confidence, verdict }: ConfidenceDialProps) {
  const pct = Math.round(confidence * 100);
  const offset = CIRCUMFERENCE * (1 - confidence);
  const colorVar = verdict === "real" ? "var(--status-good)" : "var(--status-critical)";

  return (
    <div className="confidence-dial" role="img" aria-label={`Confianza ${pct}%`}>
      <svg viewBox="0 0 128 128" className="confidence-dial__svg">
        <circle
          cx="64"
          cy="64"
          r={RADIUS}
          fill="none"
          stroke="var(--border-hairline)"
          strokeWidth={STROKE}
        />
        <circle
          cx="64"
          cy="64"
          r={RADIUS}
          fill="none"
          stroke={colorVar}
          strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
          transform="rotate(-90 64 64)"
          className="confidence-dial__arc"
        />
      </svg>
      <div className="confidence-dial__readout">
        <span className="confidence-dial__value">{pct}</span>
        <span className="confidence-dial__unit">%</span>
      </div>
    </div>
  );
}
