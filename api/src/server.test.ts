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

test("responds with CORS headers so a browser fetch() from the frontend origin isn't rejected", async () => {
  // A missing Access-Control-Allow-Origin header doesn't show up as a
  // server-side error -- the server answers fine and this process's own
  // logs show 200 -- the browser just silently refuses to hand the
  // response to the page's own JS. Caught once by hand (curl succeeding
  // while a real browser tab reported "Backend unreachable" for the exact
  // same request); pinned here so it can't regress silently again.
  const client = new FakeInferenceClient(
    async () => ({ status: "ok" }),
    async () => {
      throw new Error("not used");
    },
  );
  const app = buildServer(
    { inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600, corsOrigins: ["http://localhost:5173"] },
    client,
  );

  const res = await app.inject({
    method: "GET",
    url: "/health",
    headers: { origin: "http://localhost:5173" },
  });

  assert.equal(res.headers["access-control-allow-origin"], "http://localhost:5173");
});

test("also allows 127.0.0.1:5173 by default, not just localhost:5173", async () => {
  // A browser treats localhost and 127.0.0.1 as different origins even
  // though both mean the same machine -- a real user hit exactly this: the
  // gateway (built with only localhost:5173 allowed) silently rejected a
  // frontend opened at 127.0.0.1:5173, shown as "Backend unreachable" with
  // no error anywhere to explain why. Uses buildServer's own default (no
  // corsOrigins passed), the actual code path index.ts runs.
  const client = new FakeInferenceClient(
    async () => ({ status: "ok" }),
    async () => {
      throw new Error("not used");
    },
  );
  const app = buildServer({ inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600 }, client);

  const res = await app.inject({
    method: "GET",
    url: "/health",
    headers: { origin: "http://127.0.0.1:5173" },
  });

  assert.equal(res.headers["access-control-allow-origin"], "http://127.0.0.1:5173");
});

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
