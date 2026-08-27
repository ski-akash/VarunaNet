import type { FastifyInstance } from "fastify";
import type { InferenceClient } from "../inferenceClient.js";

// GET /health reports the gateway's own liveness AND whether it can currently
// reach the inference service -- a gateway that says "ok" while the thing it
// proxies to is down is worse than no health check (spec section 6.1's
// grounding principle applies here too: don't claim something is fine
// without having actually checked it this request).
export function registerHealthRoute(app: FastifyInstance, inference: InferenceClient): void {
  app.get("/health", async (_req, reply) => {
    try {
      const inferenceHealth = await inference.health();
      return { status: "ok", inference: inferenceHealth };
    } catch (err) {
      reply.code(503);
      return { status: "degraded", inference: null, error: (err as Error).message };
    }
  });
}
