-- The system of record for scored scenes. Redis's ResultCache sits in front
-- of this for fast repeat reads (spec section 5, Redis job 2); this table is
-- what survives a cache TTL expiry or a Redis restart, and what the Phase 6
-- tool registry will eventually query.
--
-- Run once against a fresh database: psql -d varunanet -f src/db/schema.sql

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS scenes (
    scene_id text PRIMARY KEY,
    model text NOT NULL,
    processed_at timestamptz NOT NULL,
    water_pixel_fraction double precision NOT NULL,
    flood_polygons_count integer NOT NULL,
    specks_dropped integer NOT NULL,
    districts_total integer NOT NULL,
    districts_affected integer NOT NULL,
    total_flooded_hectares double precision NOT NULL,
    -- The full response as returned by inference/service.py, kept alongside
    -- the typed columns above. The typed columns are what real queries
    -- filter/sort on; the raw payload means a field added to the Python
    -- side later doesn't need a migration before it's queryable at all.
    raw_result jsonb NOT NULL,
    inserted_at timestamptz NOT NULL DEFAULT now()
);

-- Per-district figures for a scene. Populated from the top-N
-- "worst_affected" list inference/districts.py's summarize() returns today
-- -- NOT the full district table, because that's genuinely all the Python
-- service currently exposes over HTTP (see inference/README.md: geometry
-- and the complete per-district breakdown are opt-in / not yet built).
-- Recorded here rather than silently assumed complete.
CREATE TABLE IF NOT EXISTS district_impacts (
    id bigserial PRIMARY KEY,
    scene_id text NOT NULL REFERENCES scenes (scene_id) ON DELETE CASCADE,
    district_name text NOT NULL,
    flooded_hectares double precision NOT NULL,
    flooded_percent double precision NOT NULL
);

CREATE INDEX IF NOT EXISTS district_impacts_scene_id_idx ON district_impacts (scene_id);
CREATE INDEX IF NOT EXISTS district_impacts_district_name_idx ON district_impacts (district_name);

-- Flood polygons themselves are NOT stored here yet -- inference/service.py
-- doesn't return geometry from /predict (README calls this out as a
-- deliberate opt-in it hasn't built), so there is nothing real to persist
-- into a geometry column without fabricating it. A `flood_polygons` table
-- (scene_id, geom geometry(MultiPolygon, 4326), area_hectares) is the
-- natural next addition once that endpoint exists.
