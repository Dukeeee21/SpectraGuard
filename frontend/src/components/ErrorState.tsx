import type { AnalysisError } from "../types";

const COPY: Record<AnalysisError["kind"], { title: string; hint: string }> = {
  invalid_upload: {
    title: "No se pudo procesar el archivo",
    hint: "Usá una imagen JPEG, PNG o WEBP de hasta 10 MB.",
  },
  no_face: {
    title: "No se detectó ningún rostro",
    hint: "Probá con una foto donde el rostro se vea de frente y sin obstrucciones.",
  },
  model_unavailable: {
    title: "El modelo todavía no está disponible",
    hint: "El sistema de detección aún no tiene un checkpoint entrenado cargado.",
  },
  network: {
    title: "No se pudo contactar al servidor",
    hint: "Revisá tu conexión o que la API esté corriendo.",
  },
};

export function ErrorState({ error }: { error: AnalysisError }) {
  const copy = COPY[error.kind];
  return (
    <div className="error-state" role="alert">
      <span className="error-state__icon" aria-hidden="true">
        ⚠
      </span>
      <div>
        <p className="error-state__title">{copy.title}</p>
        <p className="error-state__hint">{copy.hint}</p>
        <p className="error-state__detail">{error.detail}</p>
      </div>
    </div>
  );
}
