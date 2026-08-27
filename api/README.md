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

## The agent loop and chat (Phase 6, live)

`POST /chat {message}` — spec section 6's "one tool-calling agent over a common tool
registry," now real, not groundwork. Registered only when both a database and an LLM
provider key are configured (`GEMINI_API_KEY`); otherwise the route doesn't exist (404,
not a 500) and the rest of the API works exactly as before.

- **`src/llm/types.ts`** / **`src/llm/geminiClient.ts`** (`GeminiClient`) — the
  provider adapter. Shaped closely to Gemini's own turn structure (`role`/`parts`,
  `functionCall`/`functionResponse`) rather than an invented abstraction — a second
  provider adapter, if one is ever added, is what would prove the `LLMClient` interface
  boundary is the right one; a premature multi-provider abstraction over a single
  working implementation isn't. **Model catalog verified live, not from training
  knowledge, which turned out to be stale**: `gemini-2.5-flash`/`-flash-lite` both 404
  ("no longer available to new users") on a real key tested during this build.
  `gemini-3.5-flash-lite` is the model actually confirmed working end to end, including
  function calling. Also confirmed live: Gemini 3.x requires a function call's
  `thoughtSignature` to be echoed back verbatim on any later turn that replays it, or
  the API rejects the request outright — not documented anywhere obvious, found by a
  real `400 INVALID_ARGUMENT` during manual testing before writing any code.
- **`src/agent/toolDeclarations.ts`** — bridges `src/tools/registry.ts`'s
  provider-agnostic `ToolRegistry` to Gemini's function-declaration format, and
  dispatches a model-requested call back to the real registry method. Only the 3 tools
  `registry.ts` actually implements are declared — the model is never given a tool that
  doesn't exist.
- **`src/agent/agentLoop.ts`** (`runAgentTurn`) — the loop itself: call the model, run
  any requested tool calls against the real registry, feed results back, repeat (capped
  at 4 rounds) until a text-only response. **Every numeric claim in that final response
  is checked against the turn's own tool results before it leaves the loop** —
  `src/grounding/groundedness.ts`, a production copy of `ai-eval/src/groundedness.ts`'s
  checker (see that file's own comment for why duplicated rather than imported across
  packages). A response with any ungrounded number is still returned (refusing outright
  was judged too heavy-handed for a first cut) but flagged — `grounded: false` in the
  response body, logged as a warning server-side, and shown to the user in the frontend
  rather than silently passed through.
- Feature 3 (conversational map control) is already reachable through this same
  endpoint: `set_map_view` is one of the three real tools, so asking to zoom somewhere
  makes the model call it and the proposed view comes back in `tool_calls` for the
  frontend to apply — no separate integration needed, exactly the point of one shared
  registry (spec section 6).
- **Verified live end to end**, not just against the scripted-fake tests: real requests
  through the real gateway, real Postgres data (the `India_900498` demo scene from
  earlier pieces), and the real Gemini API — both "which district is worst affected"
  (correctly, honestly reports 0 flooded hectares for the untrained demo model rather
  than inventing a flood) and "zoom to Golaghat and show the flood extent layer"
  (correctly calls `set_map_view`) confirmed working, screenshotted in the actual
  frontend.
- 8 new tests (34 total in `api/`): 5 for the agent loop (tool execution, grounding
  catching a fabricated number, no-tool-needed path, `set_map_view`, a failed tool call
  surfacing as a `functionResponse` instead of throwing) via a scripted fake `LLMClient`
  — no real network calls in the test suite — plus 3 for the `/chat` route itself.

**Not built yet:** the tile cache and the LLM semantic cache (Redis jobs 3 and 4), SSE
streaming (responses are currently synchronous JSON, not token-by-token), the golden-set
eval harness (`ai-eval/`), and features 2/4 (sitreps, severity explanation/alerting) —
those need real Assam data and the report/uncertainty pipelines this endpoint doesn't
touch.

## PostGIS: the system of record

`ResultCache` above is a cache, not a database — a cache entry expiring should not mean
a scored scene's history is gone. `src/db/` is the durable store behind it:

- **`src/db/schema.sql`** — `scenes` (one row per scored scene: the typed summary
  columns plus the full JSON response in `raw_result`, so a field the Python side adds
  later is queryable immediately, without a migration) and `district_impacts` (per-
  district rows). Run once against a fresh database: `psql -d varunanet -f
  src/db/schema.sql`.
  **Known real limitation, stated rather than hidden**: `inference/service.py`'s
  `/predict` only returns the **top-5 worst-affected districts** (see
  `inference/districts.py`'s `summarize()`) and no raw polygon geometry — geometry is
  explicitly opt-in and not built yet (`inference/README.md`). So `district_impacts`
  holds the top 5, not the full district table, and there's no `flood_polygons`
  geometry table yet — both wait on the Python service exposing more than it does today.
- **`src/db/pool.ts`** — constructs the `pg.Pool` from `DATABASE_URL`
  (`postgres://localhost:5432/varunanet` default), the same "one place reads the env
  var" pattern as `redisConnection.ts`.
- **`src/db/sceneRepository.ts`** (`SceneRepository`) — `save()` upserts a scene's
  summary row and replaces its district rows in one transaction (so a re-run doesn't
  accumulate duplicate district rows); `get()` reads a scene back; `worstAffected()`
  ranks across all scored scenes. `save()` throws rather than silently writing a
  partial/null value if the inference service's response is ever missing a field the
  schema requires — a caught bug beats a row with a false zero in it.
- **`src/sceneWorker.ts`** now persists via `SceneRepository.save()` before writing to
  the Redis cache, and `GET /scenes/:sceneId` falls back to Postgres (repopulating the
  cache on a hit) before falling back to "not found" — so a result survives past the
  cache's TTL.

Needs a local Postgres+PostGIS for both running the gateway and its DB-backed tests:
`brew install postgresql@18 postgis`, then `createdb varunanet_test && psql -d
varunanet_test -f src/db/schema.sql` (tests default to
`postgres://localhost:5432/varunanet_test`, override with `TEST_DATABASE_URL`).

## Tool registry (Phase 6 groundwork)

`src/tools/registry.ts` (`ToolRegistry`) — spec section 6's shared tool registry, built
ahead of the actual LLM agent loop since the tools themselves are provider-agnostic:
plain typed functions against real data, independent of which model ends up calling
them (currently blocked on an LLM provider key this environment doesn't have).

Three of spec section 6.2's seven tools are implemented for real:
- `getSceneMetadata` — the exact stored payload for a named scene (or the most recently
  processed one), unmodified.
- `getWorstAffected` — ranked districts for one named scene (or the most recent), scoped
  by `scene_id` in the query itself so it can never conflate different passes once more
  than one scene has been processed.
- `setMapView` — a pure function, not a query: validates and normalizes a proposed view
  state (region/bounds/zoom/layers/timeIndex), rejecting an unknown layer name outright.
  Per spec section 6.5, this is only ever a *proposal* — the frontend stays authoritative
  and decides whether to apply it.

The other four (`query_flood_stats`, `compare_time_periods`, `get_model_confidence`,
`generate_report`) are deliberately **not** in the registry — each needs something real
that doesn't exist yet (multiple scenes over time for the same region, a real uncertainty
estimate, the PDF report pipeline), and building them against fabricated data would
violate the exact grounding rule this registry exists to enforce. See `NOT_YET_BUILT` in
the source for the reasoning per tool.

7 new tests (26 total in `api/`). **Found a real test-isolation bug while adding
these**: Node's test runner runs test *files* concurrently by default, and every
DB-backed test in this package starts with `TRUNCATE scenes, district_impacts` — two
files running at once could truncate out from under each other mid-test. Fixed by
running the whole suite with `--test-concurrency=1` (see `package.json`'s `test`
script); caught by a real, reproducible failing assertion, not a flaky-looking one.

## Auth and rate limiting

- **`src/auth.ts`** (`registerAuth`) — a single shared API key, checked against the
  `x-api-key` header on every route except `GET /health` (infra checks and the frontend's
  own status dot need to work without one, and it reveals nothing sensitive). Configured
  via `API_KEYS` (comma-separated, for rotation — old and new key both valid during a
  rotation window). **Empty by default**, a deliberate, explicit, logged-at-startup state
  for local dev (`npm run dev` shouldn't require minting a key first) — set `API_KEYS`
  for anything reachable outside a dev machine. This is a single-role dashboard, not
  multi-tenant (spec section 2), so one shared secret per deployment is the right amount
  of auth, not per-user accounts.
- **`@fastify/rate-limit`**, registered globally — `RATE_LIMIT_MAX` requests per
  `RATE_LIMIT_WINDOW_MS` (defaults 120/60s), applied before auth so a key-guessing script
  gets throttled the same as anything else.
- **Real bug hit and fixed while adding these**: routes registered directly after
  `app.register(cors, ...)` / `app.register(rateLimit, ...)` (no `await`, matching
  Fastify's normal non-blocking registration style) silently never got rate-limited —
  `app.ready()` reported success and there was no error anywhere, but two rapid requests
  both returned `200` instead of the second one being `429`. Both plugins wire their
  per-route behavior via an `onRoute` listener added during their own registration, and a
  route declared in the same synchronous tick can land before that listener attaches.
  Fixed by wrapping every route registration in `app.after(() => { ... })`, which defers
  it until every plugin registered above it has fully finished — without requiring
  `buildServer` itself to become `async` (it's called synchronously throughout the test
  suite and `index.ts`). Caught by a test that actually sent two requests and checked the
  second was `429`, not by reasoning about the code.

## CORS

`server.ts` registers `@fastify/cors`, allowing the origins in `CORS_ORIGINS`
(comma-separated, default `http://localhost:5173` — the Vite dev server). Found this was
missing the way it actually bites: `curl` against `/health` returned a normal `200` and
this process's own logs showed the request completing fine, while the real frontend tab
reported "Backend unreachable" for the identical request — a browser silently drops a
cross-origin response with no `Access-Control-Allow-Origin` header before the page's own
`fetch()` ever sees it, and none of that shows up server-side. Pinned with a test that
asserts the header is actually present, not just eyeballed once in a browser tab.
