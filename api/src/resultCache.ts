import type { Redis } from "ioredis";
import type { PredictResponse } from "./inferenceClient.js";

// The "precomputed state/district aggregate cache" from spec section 5, job
// 2 of Redis's four: so the dashboard reads a scene's district stats
// instantly instead of re-running a ~35-minute full-scene inference (or even
// re-hitting PostGIS) on every map pan. Keyed by scene_id -- one entry per
// processed pass, not per query, since the same scene's stats never change
// once the model has scored it.
const KEY_PREFIX = "varunanet:scene-result:";

export class ResultCache {
  constructor(
    private readonly redis: Redis,
    private readonly ttlSeconds: number,
  ) {}

  async get(sceneId: string): Promise<PredictResponse | null> {
    const raw = await this.redis.get(KEY_PREFIX + sceneId);
    return raw === null ? null : (JSON.parse(raw) as PredictResponse);
  }

  async set(sceneId: string, result: PredictResponse): Promise<void> {
    await this.redis.set(KEY_PREFIX + sceneId, JSON.stringify(result), "EX", this.ttlSeconds);
  }
}
