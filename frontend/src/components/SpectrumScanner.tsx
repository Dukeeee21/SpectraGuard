import { useEffect, useRef } from "react";

const BAR_COUNT = 48;

/**
 * Visualización animada de barras estilo analizador de espectro, usada como
 * estado de carga durante el análisis. No es decoración genérica: referencia
 * directamente lo que el backend calcula de verdad (magnitud del espectro
 * DFT), así el estado de "procesando" comunica qué está pasando.
 */
export function SpectrumScanner() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    let frame = 0;
    let animationId: number;
    const heights = new Array(BAR_COUNT).fill(0);
    const targets = new Array(BAR_COUNT).fill(0).map(() => Math.random());

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    };
    resize();
    window.addEventListener("resize", resize);

    const brand = getComputedStyle(canvas).getPropertyValue("--brand-500").trim() || "#2dd4bf";
    const brandDim = getComputedStyle(canvas).getPropertyValue("--brand-700").trim() || "#0f7d70";

    const draw = () => {
      frame += 1;
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      ctx.clearRect(0, 0, width, height);

      if (frame % 6 === 0) {
        for (let i = 0; i < BAR_COUNT; i++) {
          targets[i] = Math.random() * (0.35 + 0.65 * Math.abs(Math.sin(i / 5 + frame / 30)));
        }
      }

      const barWidth = width / BAR_COUNT;
      for (let i = 0; i < BAR_COUNT; i++) {
        heights[i] += (targets[i] - heights[i]) * 0.18;
        const barHeight = Math.max(2, heights[i] * height);
        const x = i * barWidth;
        const y = height - barHeight;

        const gradient = ctx.createLinearGradient(0, y, 0, height);
        gradient.addColorStop(0, brand);
        gradient.addColorStop(1, brandDim);
        ctx.fillStyle = gradient;
        ctx.fillRect(x + barWidth * 0.15, y, barWidth * 0.7, barHeight);
      }

      animationId = requestAnimationFrame(draw);
    };
    animationId = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <div className="spectrum-scanner" role="status" aria-live="polite">
      <canvas ref={canvasRef} className="spectrum-scanner__canvas" aria-hidden="true" />
      <p className="spectrum-scanner__label">
        <span className="spectrum-scanner__dot" />
        Analizando espectro de frecuencia&hellip;
      </p>
    </div>
  );
}
