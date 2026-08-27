import type pg from "pg";
import type { PredictResponse } from "../inferenceClient.js";

interface WorstAffectedEntry {
  name: string;
  flooded_hectares: number;
  flooded_percent: number;
}

// The exact shape inference/service.py's /predict returns today (see its
// README for a real example payload). Narrower than PredictResponse's
// loose index signature -- the repository needs these fields to exist to
// satisfy the schema's NOT NULL columns, so it checks for them explicitly
// rather than writing `undefined` into a required column.
function requireField<T>(result: PredictResponse, key: string): T {
  const value = (result as Record<string, unknown>)[key];
  if (value === undefined || value === null) {
    throw new Error(`scene result is missing required field "${key}"`);
  }
  return value as T;
}

// The system-of-record write: persists a scene's summary and its top-N
// district figures to Postgres. Redis's ResultCache is a cache in front of
// this, not a replacement for it -- a cache entry expiring should not mean
// the scene's history is gone.
export class SceneRepository {
  constructor(private readonly pool: pg.Pool) {}

  async save(result: PredictResponse): Promise<void> {
    const sceneId = requireField<string>(result, "scene_id");
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      await client.query(
        `INSERT INTO scenes
           (scene_id, model, processed_at, water_pixel_fraction,
            flood_polygons_count, specks_dropped, districts_total,
            districts_affected, total_flooded_hectares, raw_result)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
         ON CONFLICT (scene_id) DO UPDATE SET
           model = EXCLUDED.model,
           processed_at = EXCLUDED.processed_at,
           water_pixel_fraction = EXCLUDED.water_pixel_fraction,
           flood_polygons_count = EXCLUDED.flood_polygons_count,
           specks_dropped = EXCLUDED.specks_dropped,
           districts_total = EXCLUDED.districts_total,
           districts_affected = EXCLUDED.districts_affected,
           total_flooded_hectares = EXCLUDED.total_flooded_hectares,
           raw_result = EXCLUDED.raw_result`,
        [
          sceneId,
          requireField<string>(result, "model"),
          requireField<string>(result, "processed_at"),
          requireField<number>(result, "water_pixel_fraction"),
          requireField<number>(result, "flood_polygons"),
          requireField<number>(result, "specks_dropped"),
          requireField<number>(result, "districts_total"),
          requireField<number>(result, "districts_affected"),
          requireField<number>(result, "total_flooded_hectares"),
          JSON.stringify(result),
        ],
      );

      // Re-running a scene replaces its district rows rather than
      // accumulating duplicates alongside the ON CONFLICT upsert above.
      await client.query("DELETE FROM district_impacts WHERE scene_id = $1", [sceneId]);
      const worstAffected = ((result as Record<string, unknown>).worst_affected ??
        []) as WorstAffectedEntry[];
      for (const district of worstAffected) {
        await client.query(
          `INSERT INTO district_impacts (scene_id, district_name, flooded_hectares, flooded_percent)
           VALUES ($1, $2, $3, $4)`,
          [sceneId, district.name, district.flooded_hectares, district.flooded_percent],
        );
      }

      await client.query("COMMIT");
    } catch (err) {
      await client.query("ROLLBACK");
      throw err;
    } finally {
      client.release();
    }
  }

  async get(sceneId: string): Promise<PredictResponse | null> {
    const { rows } = await this.pool.query<{ raw_result: PredictResponse }>(
      "SELECT raw_result FROM scenes WHERE scene_id = $1",
      [sceneId],
    );
    return rows.length === 0 ? null : rows[0].raw_result;
  }

  async worstAffected(limit = 10): Promise<WorstAffectedEntry[]> {
    const { rows } = await this.pool.query<WorstAffectedEntry>(
      `SELECT district_name AS name, flooded_hectares, flooded_percent
       FROM district_impacts
       ORDER BY flooded_percent DESC
       LIMIT $1`,
      [limit],
    );
    return rows;
  }
}
