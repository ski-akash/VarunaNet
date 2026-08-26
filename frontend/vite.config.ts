import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

const require = createRequire(import.meta.url)

// maplibre-gl locates its web worker at runtime with
//   new URL('./maplibre-gl-worker.mjs', import.meta.url)
// which the bundler cannot statically resolve, so the worker file is
// never emitted. In the production bundle import.meta.url is the hashed
// entry chunk, so MapLibre ends up requesting
// /assets/maplibre-gl-worker.mjs, gets a 404, the worker never starts,
// the style never finishes loading, and the map renders as an empty
// rectangle -- with no console error, which is what made this hard to
// spot. Dev is unaffected because the worker is served straight out of
// node_modules there, so this only ever broke real deployments.
//
// Emitting the file next to the entry chunk makes that runtime URL
// resolve. Kept as a local plugin rather than adding a copy-plugin
// dependency for one file.
function emitMaplibreWorker(): Plugin {
  return {
    name: 'emit-maplibre-worker',
    apply: 'build',
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'assets/maplibre-gl-worker.mjs',
        source: readFileSync(require.resolve('maplibre-gl/dist/maplibre-gl-worker.mjs')),
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), emitMaplibreWorker()],
  // Dev only: Vite's esbuild dependency pre-bundler fails to resolve
  // maplibre-gl's worker ("file does not exist at .../maplibre-gl-worker.mjs"),
  // so the browser loads MapLibre's own module graph directly instead.
  // This has no effect on the production build -- the worker problem the
  // plugin above fixes is a separate, build-time one.
  optimizeDeps: {
    exclude: ['maplibre-gl'],
  },
})
