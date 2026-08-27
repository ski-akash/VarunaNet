import { Worker, type Job } from "bullmq";
import type { Redis } from "ioredis";
import { SCENE_QUEUE_NAME, type SceneJobData } from "./sceneQueue.js";
import type { InferenceClient } from "./inferenceClient.js";
import type { ResultCache } from "./resultCache.js";
import type { SceneRepository } from "./db/sceneRepository.js";

// The worker side of the scene-processing queue: pulls a queued scene id,
// calls the Python inference service (the same call /predict makes
// synchronously), persists the result to Postgres (the system of record --
// spec section 5 calls this out as PostGIS's job, so a run survives a cache
// TTL expiry or a Redis restart), and writes it into the aggregate cache for
// fast repeat reads. Kept separate from server.ts so it can run as its own
// process (`npm run worker`) -- an HTTP server and a long-running inference
// worker have very different scaling/restart needs and shouldn't share a
// process.
export function createSceneWorker(
  redis: Redis,
  inference: InferenceClient,
  cache: ResultCache,
  repository: SceneRepository,
): Worker<SceneJobData> {
  return new Worker<SceneJobData>(
    SCENE_QUEUE_NAME,
    async (job: Job<SceneJobData>) => {
      const result = await inference.predict(job.data.sceneId);
      await repository.save(result);
      await cache.set(job.data.sceneId, result);
      return result;
    },
    { connection: redis },
  );
}
