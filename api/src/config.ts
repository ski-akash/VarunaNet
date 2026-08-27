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
  };
}
