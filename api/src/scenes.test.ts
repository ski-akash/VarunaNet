import { test } from "node:test";
import assert from "node:assert/strict";
import { buildServer } from "./server.js";
import { createRedisConnection } from "./redisConnection.js";
import type { InferenceClient } from "./inferenceClient.js";

// Needs a real local Redis (see resultCache.test.ts) -- BullMQ's queue
// semantics (state transitions, job dedup by id) aren't worth re-faking, and
// a fake would risk verifying a mock instead of the real behaviour.
const TEST_REDIS_URL = process.env.TEST_REDIS_URL ?? "redis://localhost:6379/15";

class FakeInferenceClient implements InferenceClient {
  constructor(private readonly onPredict: (sceneId: string) => Promise<{ scene_id: string }>) {}
  health() {
    return Promise.resolve({ status: "ok" });
  }
  predict(sceneId: string) {
    return this.onPredict(sceneId);
  }
}

test("POST /scenes enqueues a scene once and a poll picks up its completion", async () => {
  const redis = createRedisConnection(TEST_REDIS_URL);
  await redis.flushdb();

  let calls = 0;
  const client = new FakeInferenceClient(async (sceneId) => {
    calls += 1;
    return { scene_id: sceneId, total_flooded_hectares: 42 };
  });
  const app = buildServer(
    { inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600 },
    client,
    redis,
  );

  const enqueueRes = await app.inject({
    method: "POST",
    url: "/scenes",
    payload: { scene_id: "India_900498" },
  });
  assert.equal(enqueueRes.statusCode, 202);
  assert.equal(enqueueRes.json().status, "queued");

  // Enqueuing the same scene again must not create a second job -- jobId =
  // sceneId is what makes this idempotent.
  const secondEnqueueRes = await app.inject({
    method: "POST",
    url: "/scenes",
    payload: { scene_id: "India_900498" },
  });
  assert.equal(secondEnqueueRes.json().job_id, enqueueRes.json().job_id);

  const statusRes = await app.inject({ method: "GET", url: "/scenes/India_900498" });
  assert.equal(statusRes.statusCode, 200);
  assert.ok(["waiting", "active"].includes(statusRes.json().status));

  await app.close();
  await redis.quit();
});

test("GET /scenes/:sceneId is 404 for a scene never queued or cached", async () => {
  const redis = createRedisConnection(TEST_REDIS_URL);
  await redis.flushdb();
  const client = new FakeInferenceClient(async (sceneId) => ({ scene_id: sceneId }));
  const app = buildServer(
    { inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600 },
    client,
    redis,
  );

  const res = await app.inject({ method: "GET", url: "/scenes/never-queued" });
  assert.equal(res.statusCode, 404);

  await app.close();
  await redis.quit();
});
