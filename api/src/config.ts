// Central place for the gateway's own runtime config -- every value here is
// read once, from the environment, with an explicit default. No caller reads
// process.env directly, so there's exactly one place that has to change if a
// deployment target (docker compose, the cluster, a laptop) needs different
// values.
export interface GatewayConfig {
  port: number;
  host: string;
  inferenceServiceUrl: string;
  redisUrl: string;
  databaseUrl: string;
  // Comma-separated list of origins the browser-facing frontend is served
  // from (Vite dev server by default). Fastify has no CORS headers on by
  // default, so the frontend's own fetch() calls are silently rejected by
  // the browser -- not a server-side error, and not visible in this
  // process's own logs -- until this is set explicitly.
  corsOrigins: string[];
  // How long a scene's district-stats result stays in the aggregate cache
  // before a fresh /scenes request re-runs inference. A Sentinel-1 scene is
  // one pass every ~6-12 days (spec section 2 -- this is explicitly not a
  // real-time system), so an hour-scale TTL is about caching repeat map
  // views of the same pass, not about freshness.
  resultCacheTtlSeconds: number;
  // Comma-separated list of accepted API keys, checked against the
  // x-api-key header on every route except /health (infra/uptime checks
  // and the frontend's own status dot need to work without a key). Empty
  // means auth is off -- the default, so a bare `npm run dev` still works
  // for local development without first minting a key; set API_KEYS to
  // turn it on for anything reachable outside a dev machine.
  apiKeys: string[];
  rateLimitMax: number;
  rateLimitWindowMs: number;
  // Empty means the chat feature is unavailable (POST /chat returns 503)
  // rather than the gateway refusing to start -- the rest of the API
  // (scenes, health, auth) has nothing to do with the LLM layer and
  // shouldn't be held hostage by a missing key.
  geminiApiKey: string;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): GatewayConfig {
  return {
    port: Number(env.PORT ?? 3000),
    host: env.HOST ?? "0.0.0.0",
    inferenceServiceUrl: env.INFERENCE_SERVICE_URL ?? "http://localhost:8000",
    redisUrl: env.REDIS_URL ?? "redis://localhost:6379",
    databaseUrl: env.DATABASE_URL ?? "postgres://localhost:5432/varunanet",
    corsOrigins: (env.CORS_ORIGINS ?? "http://localhost:5173").split(",").map((s) => s.trim()),
    resultCacheTtlSeconds: Number(env.RESULT_CACHE_TTL_SECONDS ?? 3600),
    apiKeys: (env.API_KEYS ?? "")
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0),
    rateLimitMax: Number(env.RATE_LIMIT_MAX ?? 120),
    rateLimitWindowMs: Number(env.RATE_LIMIT_WINDOW_MS ?? 60_000),
    geminiApiKey: env.GEMINI_API_KEY ?? "",
  };
}
