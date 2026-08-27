import type { FastifyInstance } from "fastify";
import type { SceneQueue } from "../sceneQueue.js";
import type { ResultCache } from "../resultCache.js";
import type { SceneRepository } from "../db/sceneRepository.js";

interface EnqueueBody {
  scene_id?: unknown;
}

// The async counterpart to POST /predict: instead of blocking the HTTP
// request for as long as a full-scene inference takes (~35 minutes at
// INT8), a scene is queued and its status/result polled separately. This is
// the route the dashboard's "process this pass" action and any future
// ingest worker are expected to call, rather than /predict directly, once a
// real (not single-chip) scene is involved.
export function registerScenesRoutes(
  app: FastifyInstance,
  queue: SceneQueue,
  cache: ResultCache,
  repository: SceneRepository,
): void {
  app.post<{ Body: EnqueueBody }>("/scenes", async (req, reply) => {
    const sceneId = req.body?.scene_id;
    if (typeof sceneId !== "string" || sceneId.length === 0) {
      reply.code(400);
      return { error: "scene_id is required" };
    }

    const cached = await cache.get(sceneId);
    if (cached) {
      return { status: "done", source: "cache", result: cached };
    }

    const job = await queue.enqueue(sceneId);
    reply.code(202);
    return { status: "queued", job_id: job.id };
  });

  app.get<{ Params: { sceneId: string } }>("/scenes/:sceneId", async (req, reply) => {
    const { sceneId } = req.params;

    const cached = await cache.get(sceneId);
    if (cached) {
      return { status: "done", source: "cache", result: cached };
    }

    // A miss here means either the result was never queued, or it was and
    // the cache entry has since expired -- Postgres is the durable copy,
    // so check it before falling through to "not found". A hit here also
    // repopulates the cache, so the next read doesn't pay this query again.
    const stored = await repository.get(sceneId);
    if (stored) {
      await cache.set(sceneId, stored);
      return { status: "done", source: "database", result: stored };
    }

    const job = await queue.getJob(sceneId);
    if (!job) {
      reply.code(404);
      return { status: "not_found" };
    }

    const state = await job.getState();
    if (state === "completed") {
      return { status: "done", source: "job", result: job.returnvalue };
    }
    if (state === "failed") {
      reply.code(500);
      return { status: "failed", error: job.failedReason };
    }
    return { status: state };
  });
}
