import { Queue, type Job } from "bullmq";
import type { Redis } from "ioredis";

// The BullMQ job queue for the async scene-processing pipeline (spec section
// 5, Redis job 1): ingest -> preprocess -> tile -> infer -> stitch ->
// vectorize -> aggregate. This gateway package only handles the tail of that
// chain today (a staged scene -> inference -> district stats, matching what
// inference/pipeline.py already does end to end); ingest/preprocess land
// here as the pipeline grows to actually fetch scenes (spec section 15.4).
export const SCENE_QUEUE_NAME = "scene-processing";

export interface SceneJobData {
  sceneId: string;
}

export class SceneQueue {
  private readonly queue: Queue<SceneJobData>;

  constructor(redis: Redis) {
    this.queue = new Queue<SceneJobData>(SCENE_QUEUE_NAME, { connection: redis });
  }

  // jobId = sceneId makes enqueuing idempotent: a second request for a scene
  // that's already queued or running returns the same job instead of
  // starting redundant work (a full scene is ~35 minutes of inference --
  // duplicating that by accident would be an expensive bug, not a cosmetic
  // one).
  async enqueue(sceneId: string): Promise<Job<SceneJobData>> {
    return this.queue.add(
      "process-scene",
      { sceneId },
      { jobId: sceneId, removeOnComplete: 1000, removeOnFail: 1000 },
    );
  }

  async getJob(sceneId: string): Promise<Job<SceneJobData> | undefined> {
    return this.queue.getJob(sceneId);
  }

  async close(): Promise<void> {
    await this.queue.close();
  }
}
