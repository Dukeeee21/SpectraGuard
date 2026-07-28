import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Mismo tratamiento que nginx en Docker: /api/* va al backend Django,
    // así el cliente siempre usa rutas relativas y no hay que lidiar con
    // CORS en ningún entorno.
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
})
