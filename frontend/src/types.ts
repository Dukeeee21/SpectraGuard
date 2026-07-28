export type Verdict = "real" | "ai_generated";

export interface AnalysisResult {
  id: number;
  label: Verdict;
  confidence: number;
  ai_probability: number;
  face_confidence: number;
  processing_time_ms: number;
  model_version: string;
  created_at: string;
}

export type AnalysisErrorKind = "invalid_upload" | "no_face" | "model_unavailable" | "network";

export interface AnalysisError {
  kind: AnalysisErrorKind;
  detail: string;
}

export interface HistoryEntry extends AnalysisResult {
  previewDataUrl: string;
}
