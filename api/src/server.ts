import Fastify, { type FastifyInstance } from "fastify";
import { HttpInferenceClient, type InferenceClient } from "./inferenceClient.js";
import { registerHealthRoute } from "./routes/health.js";
import { registerPredictRoute } from "./routes/predict.js";
import type { GatewayConfig } from "./config.js";

// buildServer takes the inference client as a parameter (rather than
// constructing an HttpInferenceClient internally) so tests can swap in a fake
// one and exercise the routes with zero real HTTP calls or a running Python
// service.
export function buildServer(
  config: Pick<GatewayConfig, "inferenceServiceUrl">,
  inference: InferenceClient = new HttpInferenceClient(config.inferenceServiceUrl),
): FastifyInstance {
  const app = Fastify({ logger: true });

  registerHealthRoute(app, inference);
  registerPredictRoute(app, inference);

  return app;
}
