import Fastify, { type FastifyInstance } from "fastify";
import { HttpInferenceClient, type InferenceClient } from "./inferenceClient.js";
import { registerHealthRoute } from "./routes/health.js";
import { registerPredictRoute } from "./routes/predict.js";
import { registerScenesRoutes } from "./routes/scenes.js";
import { SceneQueue } from "./sceneQueue.js";
import { ResultCache } from "./resultCache.js";
import { SceneRepository } from "./db/sceneRepository.js";
import type { GatewayConfig } from "./config.js";
import type { Redis } from "ioredis";
import type pg from "pg";

// buildServer takes the inference client, Redis connection, and Postgres
// pool as parameters (rather than constructing them internally) so tests
// can swap in fakes and exercise the routes with zero real HTTP calls, no
// running Python service, and -- for the routes that don't touch the
// queue/cache/DB -- no Redis or Postgres either.
export function buildServer(
  config: Pick<GatewayConfig, "inferenceServiceUrl" | "resultCacheTtlSeconds">,
  inference: InferenceClient = new HttpInferenceClient(config.inferenceServiceUrl),
  redis?: Redis,
  pgPool?: pg.Pool,
): FastifyInstance {
  const app = Fastify({ logger: true });

  registerHealthRoute(app, inference);
  registerPredictRoute(app, inference);

  if (redis && pgPool) {
    const queue = new SceneQueue(redis);
    const cache = new ResultCache(redis, config.resultCacheTtlSeconds);
    const repository = new SceneRepository(pgPool);
    registerScenesRoutes(app, queue, cache, repository);
  }

  return app;
}
