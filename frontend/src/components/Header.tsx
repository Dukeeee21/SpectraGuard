const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export function Header() {
  return (
    <header className="app-header">
      <div className="app-header__brand">
        <svg viewBox="0 0 32 32" className="app-header__mark" aria-hidden="true">
          <circle cx="16" cy="16" r="14" fill="none" stroke="var(--brand-500)" strokeWidth="2" />
          <path
            d="M6 20 L11 12 L14 17 L18 9 L22 16 L26 11"
            fill="none"
            stroke="var(--brand-500)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <div>
          <p className="app-header__title">SPECTRAGUARD</p>
          <p className="app-header__subtitle">Detección forense de imágenes generadas por IA</p>
        </div>
      </div>
      <a className="app-header__docs" href={`${API_BASE_URL}/api/docs/`} target="_blank" rel="noreferrer">
        API Docs ↗
      </a>
    </header>
  );
}
