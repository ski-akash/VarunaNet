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
  // Both files, not just the worker: maplibre-gl-worker.mjs does
  // `from "./maplibre-gl-shared.mjs"`, so shipping the worker alone gets
  // it a 404 on its own import and it still never starts. (Shipping only
  // the worker is exactly how this was first "fixed" -- the worker then
  // loaded, which looked like progress, while the map stayed just as
  // blank.) maplibre-gl-shared.mjs imports nothing further, so these two
  // close the chain.
  const WORKER_FILES = ['maplibre-gl-worker.mjs', 'maplibre-gl-shared.mjs']

  return {
    name: 'emit-maplibre-worker',
    apply: 'build',
    generateBundle() {
      for (const file of WORKER_FILES) {
        this.emitFile({
          type: 'asset',
          fileName: `assets/${file}`,
          source: readFileSync(require.resolve(`maplibre-gl/dist/${file}`)),
        })
      }
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
