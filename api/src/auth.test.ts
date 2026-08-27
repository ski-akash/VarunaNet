import { test } from "node:test";
import assert from "node:assert/strict";
import { buildServer } from "./server.js";
import type { InferenceClient } from "./inferenceClient.js";

class FakeInferenceClient implements InferenceClient {
  health() {
    return Promise.resolve({ status: "ok" });
  }
  predict(sceneId: string) {
    return Promise.resolve({ scene_id: sceneId });
  }
}

test("with no API_KEYS configured, requests are accepted unauthenticated", async () => {
  const app = buildServer(
    { inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600, apiKeys: [] },
    new FakeInferenceClient(),
  );

  const res = await app.inject({ method: "GET", url: "/health" });
  assert.equal(res.statusCode, 200);
});

test("with API_KEYS configured, a protected route without a key is rejected", async () => {
  const app = buildServer(
    { inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600, apiKeys: ["secret-key"] },
    new FakeInferenceClient(),
  );

  const res = await app.inject({
    method: "POST",
    url: "/predict",
    payload: { scene_id: "India_900498" },
  });
  assert.equal(res.statusCode, 401);
});

test("with API_KEYS configured, a protected route with the right key succeeds", async () => {
  const app = buildServer(
    { inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600, apiKeys: ["secret-key"] },
    new FakeInferenceClient(),
  );

  const res = await app.inject({
    method: "POST",
    url: "/predict",
    headers: { "x-api-key": "secret-key" },
    payload: { scene_id: "India_900498" },
  });
  assert.equal(res.statusCode, 200);
});

test("with API_KEYS configured, the wrong key is still rejected", async () => {
  const app = buildServer(
    { inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600, apiKeys: ["secret-key"] },
    new FakeInferenceClient(),
  );

  const res = await app.inject({
    method: "POST",
    url: "/predict",
    headers: { "x-api-key": "wrong-key" },
    payload: { scene_id: "India_900498" },
  });
  assert.equal(res.statusCode, 401);
});

test("/health stays open even with API_KEYS configured, for infra checks and the frontend status dot", async () => {
  const app = buildServer(
    { inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600, apiKeys: ["secret-key"] },
    new FakeInferenceClient(),
  );

  const res = await app.inject({ method: "GET", url: "/health" });
  assert.equal(res.statusCode, 200);
});

test("requests are rate limited once the configured max is exceeded", async () => {
  const app = buildServer(
    {
      inferenceServiceUrl: "http://unused",
      resultCacheTtlSeconds: 3600,
      apiKeys: [],
      rateLimitMax: 1,
      rateLimitWindowMs: 60_000,
    },
    new FakeInferenceClient(),
  );

  const first = await app.inject({ method: "GET", url: "/health" });
  const second = await app.inject({ method: "GET", url: "/health" });

  assert.equal(first.statusCode, 200);
  assert.equal(second.statusCode, 429);
});
