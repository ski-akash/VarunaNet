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

**Not built yet:** auth (API key), rate limiting, the LLM tool-calling agent loop and
tool registry, SSE streaming, and the rest of the Phase 6 REST surface
(`query_flood_stats`, `get_worst_affected`, `set_map_view`, etc.) — those come with
Phase 6. Redis (job queue, aggregate cache, tile cache, semantic cache) also isn't wired
in yet; `/predict` currently calls the inference service directly and synchronously,
which is fine for a single chip but not for a full scene at ~35 minutes of inference —
that's what the BullMQ queue in the architecture diagram is for, next.
