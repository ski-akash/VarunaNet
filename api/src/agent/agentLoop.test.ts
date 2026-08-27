import { test } from "node:test";
import assert from "node:assert/strict";
import { runAgentTurn } from "./agentLoop.js";
import { ToolRegistry } from "../tools/registry.js";
import { createPgPool } from "../db/pool.js";
import { SceneRepository } from "../db/sceneRepository.js";
import type { LLMClient, Message } from "../llm/types.js";

const TEST_DATABASE_URL =
  process.env.TEST_DATABASE_URL ?? "postgres://localhost:5432/varunanet_test";

const REAL_RESULT = {
  scene_id: "India_900498",
  model: "unetpp_2218.int8.onnx",
  processed_at: "2026-08-27T00:00:00+00:00",
  water_pixel_fraction: 0.540615,
  flood_polygons: 6,
  specks_dropped: 42,
  districts_total: 27,
  districts_affected: 1,
  total_flooded_hectares: 1251.4,
  worst_affected: [{ name: "Golaghat", flooded_hectares: 1251.4, flooded_percent: 0.37 }],
};

// A scripted fake, not a mocking library -- returns queued turns in order,
// so a test can drive a real multi-round tool-calling loop (call tool ->
// get result -> call model again -> final text) without a network call.
class ScriptedLLMClient implements LLMClient {
  private calls: Message[][] = [];
  constructor(private readonly turns: Message[]) {}

  async generateTurn(history: Message[]): Promise<Message> {
    this.calls.push(history);
    const turn = this.turns[this.calls.length - 1];
    if (!turn) throw new Error("ScriptedLLMClient ran out of scripted turns");
    return turn;
  }

  get callCount(): number {
    return this.calls.length;
  }
}

test("runAgentTurn executes a tool call and returns a grounded final answer", async () => {
  const pool = createPgPool(TEST_DATABASE_URL);
  await pool.query("TRUNCATE scenes, district_impacts");
  const repo = new SceneRepository(pool);
  await repo.save(REAL_RESULT);
  const registry = new ToolRegistry(repo);

  const llm = new ScriptedLLMClient([
    {
      role: "model",
      parts: [
        {
          functionCall: { name: "get_worst_affected", args: { scene_id: "India_900498" }, id: "c1" },
          thoughtSignature: "sig1",
        },
      ],
    },
    {
      role: "model",
      parts: [
        {
          text: "Golaghat is worst affected, with 1251.4 hectares flooded (0.37% of the district).",
        },
      ],
    },
  ]);

  const result = await runAgentTurn("Which district is worst affected?", [], llm, registry);

  assert.equal(llm.callCount, 2);
  assert.equal(result.toolCalls.length, 1);
  assert.equal(result.toolCalls[0].name, "get_worst_affected");
  assert.ok(result.responseText.includes("Golaghat"));
  assert.equal(result.groundedness.groundednessRate, 1);

  await pool.end();
});

test("runAgentTurn flags a fabricated number the tool result never produced", async () => {
  const pool = createPgPool(TEST_DATABASE_URL);
  await pool.query("TRUNCATE scenes, district_impacts");
  const repo = new SceneRepository(pool);
  await repo.save(REAL_RESULT);
  const registry = new ToolRegistry(repo);

  const llm = new ScriptedLLMClient([
    {
      role: "model",
      parts: [
        {
          functionCall: { name: "get_worst_affected", args: { scene_id: "India_900498" }, id: "c1" },
          thoughtSignature: "sig1",
        },
      ],
    },
    {
      role: "model",
      // 5000 was never in the tool result (1251.4 was) -- the model
      // hallucinating a rounder, larger figure is exactly the failure
      // mode spec section 6.1 exists to catch.
      parts: [{ text: "Golaghat has roughly 5000 hectares flooded." }],
    },
  ]);

  const result = await runAgentTurn("Which district is worst affected?", [], llm, registry);

  assert.equal(result.groundedness.groundednessRate, 0);

  await pool.end();
});

test("runAgentTurn answers directly with no tool call when none is needed", async () => {
  const registry = new ToolRegistry(null as unknown as SceneRepository);
  const llm = new ScriptedLLMClient([
    { role: "model", parts: [{ text: "I can help with flood data for Assam districts." }] },
  ]);

  const result = await runAgentTurn("What can you help with?", [], llm, registry);

  assert.equal(llm.callCount, 1);
  assert.equal(result.toolCalls.length, 0);
  assert.equal(result.groundedness.groundednessRate, 1);
});

test("runAgentTurn executes set_map_view and surfaces its proposed view state", async () => {
  const registry = new ToolRegistry(null as unknown as SceneRepository);
  const llm = new ScriptedLLMClient([
    {
      role: "model",
      parts: [
        {
          functionCall: {
            name: "set_map_view",
            args: { region: "Golaghat", layers: "flood-extent" },
            id: "c1",
          },
          thoughtSignature: "sig1",
        },
      ],
    },
    { role: "model", parts: [{ text: "I've moved the map to Golaghat." }] },
  ]);

  const result = await runAgentTurn("Zoom to Golaghat", [], llm, registry);

  assert.equal(result.toolCalls[0].name, "set_map_view");
  const data = (result.toolCalls[0].result as { data: { region: string; layers: string[] } }).data;
  assert.equal(data.region, "Golaghat");
  assert.deepEqual(data.layers, ["flood-extent"]);
});

test("runAgentTurn surfaces a failed tool call as a functionResponse instead of throwing", async () => {
  const registry = new ToolRegistry(null as unknown as SceneRepository);
  const llm = new ScriptedLLMClient([
    {
      role: "model",
      parts: [
        {
          functionCall: { name: "set_map_view", args: {}, id: "c1" }, // missing region/bounds -> throws
          thoughtSignature: "sig1",
        },
      ],
    },
    { role: "model", parts: [{ text: "I need either a region name or bounds to move the map." }] },
  ]);

  const result = await runAgentTurn("Zoom somewhere", [], llm, registry);

  assert.ok((result.toolCalls[0].result as { error: string }).error.length > 0);
  assert.equal(result.responseText, "I need either a region name or bounds to move the map.");
});
