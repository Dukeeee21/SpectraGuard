import type { AnalysisError, AnalysisResult } from "../types";

// Vacío por defecto: rutas relativas ("/api/v1/analyze/"), resueltas por el
// proxy del servidor de dev de Vite en local y por nginx dentro de Docker
// (ver vite.config.ts y frontend/nginx.conf) — mismo origen, sin CORS.
// Solo hace falta setear VITE_API_BASE_URL si la API vive en otro dominio.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

function errorKindForStatus(status: number): AnalysisError["kind"] {
  if (status === 422) return "no_face";
  if (status === 503) return "model_unavailable";
  return "invalid_upload";
}

export async function analyzeImage(file: File): Promise<AnalysisResult> {
  const body = new FormData();
  body.append("image", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/analyze/`, { method: "POST", body });
  } catch {
    throw { kind: "network", detail: "No se pudo contactar al servidor de análisis." } satisfies AnalysisError;
  }

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail: string =
      typeof payload.detail === "string" ? payload.detail : "El servidor rechazó la solicitud.";
    throw { kind: errorKindForStatus(response.status), detail } satisfies AnalysisError;
  }

  return payload as AnalysisResult;
}
