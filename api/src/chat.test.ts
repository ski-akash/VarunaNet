import { test } from "node:test";
import assert from "node:assert/strict";
import { buildServer } from "./server.js";
import { createPgPool } from "./db/pool.js";
import { SceneRepository } from "./db/sceneRepository.js";
import type { InferenceClient } from "./inferenceClient.js";
import type { LLMClient, Message } from "./llm/types.js";

const TEST_DATABASE_URL =
  process.env.TEST_DATABASE_URL ?? "postgres://localhost:5432/varunanet_test";

class FakeInferenceClient implements InferenceClient {
  health() {
    return Promise.resolve({ status: "ok" });
  }
  predict(sceneId: string) {
    return Promise.resolve({ scene_id: sceneId });
  }
}

class FakeLLMClient implements LLMClient {
  async generateTurn(): Promise<Message> {
    return { role: "model", parts: [{ text: "There is no data for that region yet." }] };
  }
}

test("POST /chat answers a question via the agent loop", async () => {
  const pool = createPgPool(TEST_DATABASE_URL);
  await pool.query("TRUNCATE scenes, district_impacts");
  const app = buildServer(
    { inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600 },
    new FakeInferenceClient(),
    undefined,
    pool,
    new FakeLLMClient(),
  );

  const res = await app.inject({
    method: "POST",
    url: "/chat",
    payload: { message: "What districts are flooded?" },
  });

  assert.equal(res.statusCode, 200);
  const body = res.json();
  assert.equal(body.response, "There is no data for that region yet.");
  assert.equal(body.grounded, true);

  await app.close();
  await pool.end();
});

test("POST /chat rejects a request with no message", async () => {
  const pool = createPgPool(TEST_DATABASE_URL);
  const app = buildServer(
    { inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600 },
    new FakeInferenceClient(),
    undefined,
    pool,
    new FakeLLMClient(),
  );

  const res = await app.inject({ method: "POST", url: "/chat", payload: {} });

  assert.equal(res.statusCode, 400);

  await app.close();
  await pool.end();
});

test("POST /chat is not registered at all when no LLM client is configured", async () => {
  const app = buildServer(
    { inferenceServiceUrl: "http://unused", resultCacheTtlSeconds: 3600, geminiApiKey: "" },
    new FakeInferenceClient(),
  );

  const res = await app.inject({ method: "POST", url: "/chat", payload: { message: "hi" } });

  assert.equal(res.statusCode, 404);

  await app.close();
});
