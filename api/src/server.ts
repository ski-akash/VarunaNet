import Fastify, { type FastifyInstance } from "fastify";
import { HttpInferenceClient, type InferenceClient } from "./inferenceClient.js";
import { registerHealthRoute } from "./routes/health.js";
import { registerPredictRoute } from "./routes/predict.js";
import { registerScenesRoutes } from "./routes/scenes.js";
import { SceneQueue } from "./sceneQueue.js";
import { ResultCache } from "./resultCache.js";
import type { GatewayConfig } from "./config.js";
import type { Redis } from "ioredis";

// buildServer takes the inference client and Redis connection as parameters
// (rather than constructing them internally) so tests can swap in fakes and
// exercise the routes with zero real HTTP calls, no running Python service,
// and -- for the routes that don't touch the queue/cache -- no Redis either.
export function buildServer(
  config: Pick<GatewayConfig, "inferenceServiceUrl" | "resultCacheTtlSeconds">,
  inference: InferenceClient = new HttpInferenceClient(config.inferenceServiceUrl),
  redis?: Redis,
): FastifyInstance {
  const app = Fastify({ logger: true });

  registerHealthRoute(app, inference);
  registerPredictRoute(app, inference);

  if (redis) {
    const queue = new SceneQueue(redis);
    const cache = new ResultCache(redis, config.resultCacheTtlSeconds);
    registerScenesRoutes(app, queue, cache);
  }

  return app;
}
