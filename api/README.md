# api/

The Node.js (Fastify) gateway that sits between the frontend and everything else.

This folder holds:
- REST endpoints the frontend calls (flood stats, scene metadata, report generation, etc).
- The LLM tool-calling agent loop — the piece that lets the chat panel answer questions by calling real database queries instead of guessing numbers.
- The shared tool registry (e.g. `query_flood_stats`, `get_worst_affected`, `set_map_view`) that all four AI features use, so there's one integration instead of four separate ones.
- SSE (Server-Sent Events) streaming for the chat panel, so responses appear token-by-token.
- Auth (a simple API key) and rate limiting.

This is orchestration only — it doesn't run the model itself (that's `inference/`) and doesn't render anything (that's `frontend/`).

## Built so far

A minimal Fastify skeleton with the first real endpoint, deliberately thin so the
routing/proxy shape is right before the LLM agent loop and Redis get layered on top.

- **`src/config.ts`** — one place that reads `process.env` (`PORT`, `HOST`,
  `INFERENCE_SERVICE_URL`), each with an explicit default, so no route or client
  reaches into the environment directly.
- **`src/inferenceClient.ts`** — the only thing in this package that talks to
  `inference/service.py`. `HttpInferenceClient` implements a small `InferenceClient`
  interface (`health()`, `predict(sceneId)`); routes depend on the interface, not the
  HTTP implementation, so tests swap in a fake client and exercise the routes with zero
  real network calls or a running Python service. A distinct
  `InferenceServiceUnavailableError` is thrown when the fetch itself fails (service
  down/unreachable), separate from the service responding with an error — the two cases
  map to different HTTP status codes back to the caller (503 vs 502).
- **`src/routes/health.ts`** — `GET /health` reports the gateway as `degraded` (503),
  not `ok`, when it cannot currently reach the inference service. A gateway that says
  "ok" while the service behind it is down is worse than no health check.
- **`src/routes/predict.ts`** — `POST /predict {scene_id}` forwards the scene id to the
  inference service and relays its district-stats answer straight back. No logic of its
  own beyond validating `scene_id` is present — orchestration, not re-derivation.
- **`src/server.ts`** — `buildServer(config, inferenceClient?)` takes the inference
  client as a parameter so `src/server.test.ts` can inject a fake one.

Run it: `npm install && npm run dev` (needs `inference/service.py` running separately,
or set `INFERENCE_SERVICE_URL` to point at one). Test it: `npm test`.

## Redis: job queue + aggregate cache

Two of spec section 5's four Redis jobs, wired in:

- **`src/sceneQueue.ts`** — a BullMQ queue (`scene-processing`) for the async scene
  pipeline. `POST /scenes {scene_id}` enqueues and returns `202` immediately rather than
  blocking the request for the ~35 minutes a real full scene takes at INT8 (see
  `inference/README.md`). The job id is the scene id itself, so re-enqueuing a scene
  already queued or running is a no-op that returns the existing job rather than
  duplicating expensive work.
- **`src/sceneWorker.ts`** — the consumer side, run as its own process
  (`npm run worker`, `src/worker.ts`): pulls a queued scene, calls the same inference
  client `/predict` uses, and writes the result into the cache below. Deliberately a
  separate process from the HTTP server — an API server and a long-running inference
  worker scale and restart independently.
- **`src/resultCache.ts`** (`ResultCache`) — the precomputed district/state aggregate
  cache (spec section 5, Redis job 2): once a scene is scored, `GET /scenes/:sceneId`
  is a cache hit, not a re-run. TTL defaults to an hour (`RESULT_CACHE_TTL_SECONDS`) —
  about caching repeat map views of the same pass, not freshness, since Sentinel-1's
  own revisit cadence is 6–12 days (spec section 2).
- **`src/routes/scenes.ts`** — `POST /scenes` (enqueue-or-return-cached) and
  `GET /scenes/:sceneId` (cache hit → job state → 404), the poll-based counterpart to
  the existing synchronous `/predict`.

Needs a local Redis for both running the gateway and running its Redis-backed tests
(`redis-server`, or `brew services start redis`); `REDIS_URL` defaults to
`redis://localhost:6379`. The `resultCache.test.ts` / `scenes.test.ts` suites use
`redis://localhost:6379/15` by default (a separate DB index, not the default one) so
running the tests doesn't collide with a real dev instance's data.

**Not built yet:** auth (API key), rate limiting, the tile cache and the LLM semantic
cache (Redis jobs 3 and 4 — the latter needs Phase 6's agent loop to exist first), the
LLM tool-calling agent loop and tool registry, SSE streaming, and the rest of the
Phase 6 REST surface (`query_flood_stats`, `get_worst_affected`, `set_map_view`, etc.).
PostGIS isn't wired in either — `ResultCache` is a cache in front of computed results,
not the system of record; that's the next piece.
