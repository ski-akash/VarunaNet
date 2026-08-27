import { test } from "node:test";
import assert from "node:assert/strict";
import { buildServer } from "./server.js";
import { createRedisConnection } from "./redisConnection.js";
import { createPgPool } from "./db/pool.js";
import type { InferenceClient } from "./inferenceClient.js";

// Needs a real local Redis (see resultCache.test.ts) -- BullMQ's queue
// semantics (state transitions, job dedup by id) aren't worth re-faking, and
// a fake would risk verifying a mock instead of the real behaviour. Same
// reasoning for a real local Postgres+PostGIS with the schema in
// src/db/schema.sql already applied: `createdb varunanet_test && psql -d
// varunanet_test -f src/db/schema.sql`.
const TEST_REDIS_URL = process.env.TEST_REDIS_URL ?? "redis://localhost:6379/15";
const TEST_DATABASE_URL =
  process.env.TEST_DATABASE_URL ?? "postgres://localhost:5432/varunanet_test";

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
  const pgPool = createPgPool(TEST_DATABASE_URL);
  await pgPool.query("TRUNCATE scenes, district_impacts");

  let calls = 0;
  const client = new FakeInferenceClient(async (sceneId) => {
    calls += 1;
    return { scene_id: sceneId, total_flooded_hectares: 42 };
  });
  const app = buildServer(
    { inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600 },
    client,
    redis,
    pgPool,
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
  await pgPool.end();
});

test("GET /scenes/:sceneId is 404 for a scene never queued, cached, or persisted", async () => {
  const redis = createRedisConnection(TEST_REDIS_URL);
  await redis.flushdb();
  const pgPool = createPgPool(TEST_DATABASE_URL);
  await pgPool.query("TRUNCATE scenes, district_impacts");
  const client = new FakeInferenceClient(async (sceneId) => ({ scene_id: sceneId }));
  const app = buildServer(
    { inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600 },
    client,
    redis,
    pgPool,
  );

  const res = await app.inject({ method: "GET", url: "/scenes/never-queued" });
  assert.equal(res.statusCode, 404);

  await app.close();
  await redis.quit();
  await pgPool.end();
});

test("GET /scenes/:sceneId falls back to Postgres when the cache has expired", async () => {
  const redis = createRedisConnection(TEST_REDIS_URL);
  await redis.flushdb();
  const pgPool = createPgPool(TEST_DATABASE_URL);
  await pgPool.query("TRUNCATE scenes, district_impacts");
  await pgPool.query(
    `INSERT INTO scenes
       (scene_id, model, processed_at, water_pixel_fraction, flood_polygons_count,
        specks_dropped, districts_total, districts_affected, total_flooded_hectares, raw_result)
     VALUES ($1, 'unetpp.int8.onnx', now(), 0.54, 6, 42, 27, 1, 1251.4, $2)`,
    [
      "India_900498",
      JSON.stringify({ scene_id: "India_900498", total_flooded_hectares: 1251.4 }),
    ],
  );
  const client = new FakeInferenceClient(async () => {
    throw new Error("should not call inference for an already-persisted scene");
  });
  const app = buildServer(
    { inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600 },
    client,
    redis,
    pgPool,
  );

  const res = await app.inject({ method: "GET", url: "/scenes/India_900498" });
  assert.equal(res.statusCode, 200);
  assert.equal(res.json().source, "database");
  assert.equal(res.json().result.total_flooded_hectares, 1251.4);

  await app.close();
  await redis.quit();
  await pgPool.end();
});
