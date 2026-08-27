import { test } from "node:test";
import assert from "node:assert/strict";
import { createPgPool } from "./pool.js";
import { SceneRepository } from "./sceneRepository.js";

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

test("SceneRepository saves a scene result and reads it back unchanged", async () => {
  const pool = createPgPool(TEST_DATABASE_URL);
  await pool.query("TRUNCATE scenes, district_impacts");
  const repo = new SceneRepository(pool);

  await repo.save(REAL_RESULT);

  const stored = await repo.get("India_900498");
  assert.deepEqual(stored, REAL_RESULT);

  const worst = await repo.worstAffected();
  assert.deepEqual(worst, [{ name: "Golaghat", flooded_hectares: 1251.4, flooded_percent: 0.37 }]);

  await pool.end();
});

test("SceneRepository.save is a real upsert: re-saving replaces district rows, not duplicates them", async () => {
  const pool = createPgPool(TEST_DATABASE_URL);
  await pool.query("TRUNCATE scenes, district_impacts");
  const repo = new SceneRepository(pool);

  await repo.save(REAL_RESULT);
  await repo.save({ ...REAL_RESULT, total_flooded_hectares: 1300.0 });

  const stored = await repo.get("India_900498");
  assert.equal(stored?.total_flooded_hectares, 1300.0);

  const { rows } = await pool.query("SELECT count(*)::int FROM district_impacts WHERE scene_id = $1", [
    "India_900498",
  ]);
  assert.equal(rows[0].count, 1);

  await pool.end();
});

test("SceneRepository.save rejects a result missing a required field rather than writing a partial row", async () => {
  const pool = createPgPool(TEST_DATABASE_URL);
  await pool.query("TRUNCATE scenes, district_impacts");
  const repo = new SceneRepository(pool);

  await assert.rejects(() => repo.save({ scene_id: "incomplete" }));

  const stored = await repo.get("incomplete");
  assert.equal(stored, null);

  await pool.end();
});
