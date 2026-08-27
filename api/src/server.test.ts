import { test } from "node:test";
import assert from "node:assert/strict";
import { buildServer } from "./server.js";
import { InferenceServiceUnavailableError, type InferenceClient } from "./inferenceClient.js";

class FakeInferenceClient implements InferenceClient {
  constructor(
    private readonly healthResult: () => Promise<{ status: string }>,
    private readonly predictResult: (sceneId: string) => Promise<{ scene_id: string }>,
  ) {}

  health() {
    return this.healthResult();
  }

  predict(sceneId: string) {
    return this.predictResult(sceneId);
  }
}

test("GET /health returns ok when the inference service is reachable", async () => {
  const client = new FakeInferenceClient(
    async () => ({ status: "ok" }),
    async () => {
      throw new Error("not used");
    },
  );
  const app = buildServer({ inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600 }, client);

  const res = await app.inject({ method: "GET", url: "/health" });

  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.json(), { status: "ok", inference: { status: "ok" } });
});

test("GET /health reports degraded, not ok, when the inference service is down", async () => {
  const client = new FakeInferenceClient(
    async () => {
      throw new InferenceServiceUnavailableError(new Error("ECONNREFUSED"));
    },
    async () => {
      throw new Error("not used");
    },
  );
  const app = buildServer({ inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600 }, client);

  const res = await app.inject({ method: "GET", url: "/health" });

  assert.equal(res.statusCode, 503);
  assert.equal(res.json().status, "degraded");
});

test("POST /predict forwards scene_id and relays the inference service's answer", async () => {
  let receivedSceneId: string | undefined;
  const client = new FakeInferenceClient(
    async () => ({ status: "ok" }),
    async (sceneId) => {
      receivedSceneId = sceneId;
      return { scene_id: sceneId, total_flooded_hectares: 1251.4 };
    },
  );
  const app = buildServer({ inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600 }, client);

  const res = await app.inject({
    method: "POST",
    url: "/predict",
    payload: { scene_id: "India_900498" },
  });

  assert.equal(res.statusCode, 200);
  assert.equal(receivedSceneId, "India_900498");
  assert.deepEqual(res.json(), { scene_id: "India_900498", total_flooded_hectares: 1251.4 });
});

test("POST /predict rejects a request with no scene_id", async () => {
  const client = new FakeInferenceClient(
    async () => ({ status: "ok" }),
    async () => {
      throw new Error("not used");
    },
  );
  const app = buildServer({ inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600 }, client);

  const res = await app.inject({ method: "POST", url: "/predict", payload: {} });

  assert.equal(res.statusCode, 400);
});

test("POST /predict returns 503, not a raw 500, when the inference service is unreachable", async () => {
  const client = new FakeInferenceClient(
    async () => ({ status: "ok" }),
    async () => {
      throw new InferenceServiceUnavailableError(new Error("ECONNREFUSED"));
    },
  );
  const app = buildServer({ inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600 }, client);

  const res = await app.inject({
    method: "POST",
    url: "/predict",
    payload: { scene_id: "India_900498" },
  });

  assert.equal(res.statusCode, 503);
});
