import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // maplibre-gl loads its own worker script at runtime; Vite's esbuild-
  // based dependency pre-bundler doesn't resolve that correctly and
  // fails with "file does not exist at .../maplibre-gl-worker.mjs" --
  // excluding it here lets the browser load MapLibre's own ES module
  // graph directly instead of through the (broken, for this package)
  // pre-bundled version.
  optimizeDeps: {
    exclude: ['maplibre-gl'],
  },
})
