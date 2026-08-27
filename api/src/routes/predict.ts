import type { FastifyInstance } from "fastify";
import { InferenceServiceUnavailableError, type InferenceClient } from "../inferenceClient.js";

interface PredictBody {
  scene_id?: unknown;
}

// POST /predict forwards a scene_id to the Python inference service and
// relays its district-level answer straight back. Deliberately no logic of
// its own -- the gateway's job is orchestration and auth, not re-deriving
// anything the inference service already computed.
export function registerPredictRoute(app: FastifyInstance, inference: InferenceClient): void {
  app.post<{ Body: PredictBody }>("/predict", async (req, reply) => {
    const sceneId = req.body?.scene_id;
    if (typeof sceneId !== "string" || sceneId.length === 0) {
      reply.code(400);
      return { error: "scene_id is required" };
    }

    try {
      return await inference.predict(sceneId);
    } catch (err) {
      if (err instanceof InferenceServiceUnavailableError) {
        reply.code(503);
        return { error: "inference service unreachable" };
      }
      reply.code(502);
      return { error: (err as Error).message };
    }
  });
}
