import { test } from "node:test";
import assert from "node:assert/strict";
import { createPgPool } from "../db/pool.js";
import { SceneRepository } from "../db/sceneRepository.js";
import { ToolRegistry } from "./registry.js";

const TEST_DATABASE_URL =
  process.env.TEST_DATABASE_URL ?? "postgres://localhost:5432/varunanet_test";

const SCENE_A = {
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

test("get_scene_metadata returns the exact stored payload for a named scene", async () => {
  const pool = createPgPool(TEST_DATABASE_URL);
  await pool.query("TRUNCATE scenes, district_impacts");
  const repo = new SceneRepository(pool);
  await repo.save(SCENE_A);
  const registry = new ToolRegistry(repo);

  const result = await registry.getSceneMetadata({ sceneId: "India_900498" });

  assert.equal(result.tool, "get_scene_metadata");
  assert.deepEqual(result.data, SCENE_A);
  assert.ok(new Date(result.queriedAt).getTime() > 0);

  await pool.end();
});

test("get_scene_metadata falls back to the most recently processed scene when none is named", async () => {
  const pool = createPgPool(TEST_DATABASE_URL);
  await pool.query("TRUNCATE scenes, district_impacts");
  const repo = new SceneRepository(pool);
  await repo.save(SCENE_A);
  const registry = new ToolRegistry(repo);

  const result = await registry.getSceneMetadata();

  assert.equal((result.data as typeof SCENE_A).scene_id, "India_900498");

  await pool.end();
});

test("get_scene_metadata returns null data, not a throw, for a scene never processed", async () => {
  const pool = createPgPool(TEST_DATABASE_URL);
  await pool.query("TRUNCATE scenes, district_impacts");
  const repo = new SceneRepository(pool);
  const registry = new ToolRegistry(repo);

  const result = await registry.getSceneMetadata({ sceneId: "never-processed" });

  assert.equal(result.data, null);

  await pool.end();
});

test("get_worst_affected is scoped to the named scene, not every scene ever stored", async () => {
  const pool = createPgPool(TEST_DATABASE_URL);
  await pool.query("TRUNCATE scenes, district_impacts");
  const repo = new SceneRepository(pool);
  await repo.save(SCENE_A);
  await repo.save({
    ...SCENE_A,
    scene_id: "Other_scene",
    worst_affected: [{ name: "Nagaon", flooded_hectares: 5000, flooded_percent: 90 }],
  });
  const registry = new ToolRegistry(repo);

  const result = await registry.getWorstAffected({ sceneId: "India_900498" });

  assert.deepEqual(result.data, [{ name: "Golaghat", flooded_hectares: 1251.4, flooded_percent: 0.37 }]);

  await pool.end();
});

test("set_map_view rejects an unknown layer rather than silently passing it through", () => {
  const registry = new ToolRegistry(null as unknown as SceneRepository);

  assert.throws(() => registry.setMapView({ region: "Assam", layers: ["not-a-real-layer"] }));
});

test("set_map_view requires a region or bounds", () => {
  const registry = new ToolRegistry(null as unknown as SceneRepository);

  assert.throws(() => registry.setMapView({}));
});

test("set_map_view returns a normalized view state with defaults filled in", () => {
  const registry = new ToolRegistry(null as unknown as SceneRepository);

  const result = registry.setMapView({ region: "Assam", layers: ["flood-extent"] });

  assert.deepEqual(result.data, {
    region: "Assam",
    bounds: null,
    zoom: 6,
    layers: ["flood-extent"],
    timeIndex: 0,
  });
});
