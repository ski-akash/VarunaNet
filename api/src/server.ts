import Fastify, { type FastifyInstance } from "fastify";
import cors from "@fastify/cors";
import rateLimit from "@fastify/rate-limit";
import { HttpInferenceClient, type InferenceClient } from "./inferenceClient.js";
import { registerHealthRoute } from "./routes/health.js";
import { registerPredictRoute } from "./routes/predict.js";
import { registerScenesRoutes } from "./routes/scenes.js";
import { SceneQueue } from "./sceneQueue.js";
import { ResultCache } from "./resultCache.js";
import { SceneRepository } from "./db/sceneRepository.js";
import { registerAuth } from "./auth.js";
import type { GatewayConfig } from "./config.js";
import type { Redis } from "ioredis";
import type pg from "pg";

type BuildServerConfig = Pick<GatewayConfig, "inferenceServiceUrl" | "resultCacheTtlSeconds"> &
  Partial<Pick<GatewayConfig, "corsOrigins" | "apiKeys" | "rateLimitMax" | "rateLimitWindowMs">>;

// buildServer takes the inference client, Redis connection, and Postgres
// pool as parameters (rather than constructing them internally) so tests
// can swap in fakes and exercise the routes with zero real HTTP calls, no
// running Python service, and -- for the routes that don't touch the
// queue/cache/DB -- no Redis or Postgres either.
export function buildServer(
  config: BuildServerConfig,
  inference: InferenceClient = new HttpInferenceClient(config.inferenceServiceUrl),
  redis?: Redis,
  pgPool?: pg.Pool,
): FastifyInstance {
  const app = Fastify({ logger: true });

  // Without this, every fetch() from the frontend is rejected by the
  // browser itself before it ever reaches a route handler -- the server
  // sees and answers the request fine (it shows up 200 in this process's
  // own logs), but the response has no Access-Control-Allow-Origin header,
  // so the page's own fetch() promise rejects with an opaque "Failed to
  // fetch" and nothing useful appears server-side to explain why. Found by
  // comparing curl (which ignores CORS) against a real browser tab hitting
  // the exact same endpoint and getting a different outcome.
  app.register(cors, { origin: config.corsOrigins ?? ["http://localhost:5173"] });

  app.register(rateLimit, {
    max: config.rateLimitMax ?? 120,
    timeWindow: config.rateLimitWindowMs ?? 60_000,
  });

  // Routes are registered inside app.after(), not directly here. Both
  // @fastify/cors and @fastify/rate-limit wire themselves up via an onRoute
  // listener added during their own (asynchronous) registration -- a route
  // declared in the same synchronous tick, before that listener is
  // attached, silently never gets the plugin's per-route behavior even
  // though app.ready() later reports success. Caught by a rate-limit test
  // that kept passing 200 twice in a row instead of 429 the second time,
  // with no error anywhere to explain why. app.after() defers this callback
  // until every plugin registered above it has fully finished, without
  // requiring buildServer itself to become async.
  app.after(() => {
    registerAuth(app, config.apiKeys ?? []);

    registerHealthRoute(app, inference);
    registerPredictRoute(app, inference);

    if (redis && pgPool) {
      const queue = new SceneQueue(redis);
      const cache = new ResultCache(redis, config.resultCacheTtlSeconds);
      const repository = new SceneRepository(pgPool);
      registerScenesRoutes(app, queue, cache, repository);
    }
  });

  return app;
}
