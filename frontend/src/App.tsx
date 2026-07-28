import { useCallback, useState } from "react";
import { Header } from "./components/Header";
import { Dropzone } from "./components/Dropzone";
import { SpectrumScanner } from "./components/SpectrumScanner";
import { VerdictCard } from "./components/VerdictCard";
import { ErrorState } from "./components/ErrorState";
import { HistoryList } from "./components/HistoryList";
import { useSessionHistory } from "./hooks/useSessionHistory";
import { analyzeImage } from "./lib/api";
import type { AnalysisError, AnalysisResult } from "./types";
import "./App.css";

type Status = "idle" | "analyzing" | "done" | "error";

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function App() {
  const [status, setStatus] = useState<Status>("idle");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<AnalysisError | null>(null);
  const { entries, addEntry, clear } = useSessionHistory();

  const handleFileSelected = useCallback(
    async (file: File) => {
      setStatus("analyzing");
      setResult(null);
      setError(null);

      const dataUrl = await fileToDataUrl(file);
      setPreviewUrl(dataUrl);

      try {
        const analysis = await analyzeImage(file);
        setResult(analysis);
        setStatus("done");
        addEntry({ ...analysis, previewDataUrl: dataUrl });
      } catch (err) {
        setError(err as AnalysisError);
        setStatus("error");
      }
    },
    [addEntry],
  );

  return (
    <div className="app-shell">
      <Header />

      <main className="app-main">
        <section className="analyzer-panel">
          <Dropzone
            previewUrl={previewUrl}
            onFileSelected={handleFileSelected}
            disabled={status === "analyzing"}
          />

          <div className="analyzer-panel__result">
            {status === "analyzing" && <SpectrumScanner />}
            {status === "done" && result && <VerdictCard result={result} />}
            {status === "error" && error && <ErrorState error={error} />}
            {status === "idle" && (
              <p className="analyzer-panel__hint">
                El resultado del análisis va a aparecer acá: veredicto, confianza y desglose
                por señal (espacial + frecuencia).
              </p>
            )}
          </div>
        </section>

        <HistoryList entries={entries} onClear={clear} />
      </main>
    </div>
  );
}

export default App;
