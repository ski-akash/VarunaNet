import { test } from "node:test";
import assert from "node:assert/strict";
import { createRedisConnection } from "./redisConnection.js";
import { ResultCache } from "./resultCache.js";

// Needs a real local Redis (the same one docker-compose/the dashboard would
// use) -- run `redis-server` or `brew services start redis` before running
// this suite, matching how the GPU-dependent tests elsewhere in this project
// need the SLURM cluster rather than being faked out.
const TEST_REDIS_URL = process.env.TEST_REDIS_URL ?? "redis://localhost:6379/15";

test("ResultCache round-trips a result and expires it after its TTL", async () => {
  const redis = createRedisConnection(TEST_REDIS_URL);
  await redis.flushdb();
  const cache = new ResultCache(redis, 3600);

  assert.equal(await cache.get("India_900498"), null);

  await cache.set("India_900498", { scene_id: "India_900498", total_flooded_hectares: 1251.4 });
  const cached = await cache.get("India_900498");
  assert.deepEqual(cached, { scene_id: "India_900498", total_flooded_hectares: 1251.4 });

  const ttl = await redis.ttl("varunanet:scene-result:India_900498");
  assert.ok(ttl > 0 && ttl <= 3600, `expected a positive TTL <= 3600, got ${ttl}`);

  await redis.quit();
});
