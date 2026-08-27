/**
 * The shared tool registry spec section 6 calls for: one set of tools every
 * AI feature (chat, sitreps, map control, severity explanation) calls
 * through, so there is one integration instead of four. Built ahead of the
 * agent loop itself (Phase 6 proper needs an LLM provider key this
 * environment doesn't have) because the tools are provider-agnostic --
 * they're plain typed functions against real data, and whichever model
 * ends up calling them is a separate concern.
 *
 * spec section 6.1's hard rule shapes every tool here: the LLM may never
 * state a number that didn't come from a tool result. That means every
 * tool below either returns something a real query in this project's own
 * database actually produced, or is a pure, fully-specified computation
 * (set_map_view) -- never a plausible-looking placeholder. Where the full
 * spec section 6.2 signature can't honestly be met yet (see NOT_YET_BUILT
 * below), the tool is left out of this registry entirely rather than
 * built against fabricated data.
 */
import type { SceneRepository } from "../db/sceneRepository.js";

export interface ToolResult<T> {
  tool: string;
  // ISO timestamp of when the query actually ran -- part of the
  // provenance every grounded figure needs to carry (spec section 6.1).
  queriedAt: string;
  data: T;
}

export interface GetSceneMetadataInput {
  sceneId?: string;
}

export interface GetWorstAffectedInput {
  topN?: number;
  sceneId?: string;
}

export interface SetMapViewInput {
  region?: string;
  bounds?: [number, number, number, number]; // [west, south, east, north]
  zoom?: number;
  layers?: string[];
  timeIndex?: number;
}

export interface MapViewState {
  region: string | null;
  bounds: [number, number, number, number] | null;
  zoom: number;
  layers: string[];
  timeIndex: number;
}

const KNOWN_LAYERS = ["flood-extent", "permanent-water", "confidence-heatmap", "sar-backscatter"];

export class ToolRegistry {
  constructor(private readonly scenes: SceneRepository) {}

  /**
   * get_scene_metadata(scene_id) -- spec section 6.2. Returns the exact
   * payload inference/service.py produced for that scene (or the most
   * recently processed one, if none is named), unmodified: no
   * recomputation, no summarization that could drift from what was
   * actually measured.
   */
  async getSceneMetadata(input: GetSceneMetadataInput = {}): Promise<ToolResult<unknown | null>> {
    const sceneId = input.sceneId ?? (await this.scenes.mostRecentSceneId());
    const data = sceneId ? await this.scenes.get(sceneId) : null;
    return { tool: "get_scene_metadata", queriedAt: new Date().toISOString(), data };
  }

  /**
   * get_worst_affected(admin_level, date, top_n, metric) -- spec section
   * 6.2, narrowed to what's honestly implementable today: admin_level is
   * always "district" (no state-level aggregation exists yet), metric is
   * always flooded_percent (spec section 15.1's own finding: ranking by
   * share of district area, not absolute hectares, is what actually
   * identifies the worst-hit places -- see inference/README.md's
   * Kamrup-Metropolitan-vs-Nagaon example), and "date" is implicitly
   * "the named scene, or the most recent one" -- true date-range
   * filtering needs more than one scene ever processed for the same
   * region, which doesn't exist yet without real Assam ingestion.
   */
  async getWorstAffected(input: GetWorstAffectedInput = {}): Promise<ToolResult<unknown[]>> {
    const sceneId = input.sceneId ?? (await this.scenes.mostRecentSceneId()) ?? undefined;
    const data = sceneId ? await this.scenes.worstAffected(input.topN ?? 10, sceneId) : [];
    return { tool: "get_worst_affected", queriedAt: new Date().toISOString(), data };
  }

  /**
   * set_map_view(bounds | region, zoom, layers, time_index) -- spec
   * section 6.2/6.5. A pure function, not a database query: it validates
   * and normalizes a proposed view state, but per spec section 6.5 ("the
   * UI state machine is authoritative; the agent proposes view states,
   * the frontend validates and applies them"), this is only ever a
   * *proposal* -- the frontend still decides whether to apply it.
   * Rejects an unknown layer name rather than silently dropping or
   * passing it through, since a typo'd layer would otherwise fail
   * silently on the frontend side with no indication why the view didn't
   * change as asked.
   */
  setMapView(input: SetMapViewInput): ToolResult<MapViewState> {
    if (!input.region && !input.bounds) {
      throw new Error("set_map_view requires either region or bounds");
    }
    const layers = input.layers ?? [];
    const unknown = layers.filter((l) => !KNOWN_LAYERS.includes(l));
    if (unknown.length > 0) {
      throw new Error(`unknown layer(s): ${unknown.join(", ")} (known: ${KNOWN_LAYERS.join(", ")})`);
    }

    const data: MapViewState = {
      region: input.region ?? null,
      bounds: input.bounds ?? null,
      zoom: input.zoom ?? 6,
      layers,
      timeIndex: input.timeIndex ?? 0,
    };
    return { tool: "set_map_view", queriedAt: new Date().toISOString(), data };
  }
}

/**
 * spec section 6.2's remaining tools, deliberately not in this registry
 * yet -- each needs something real that doesn't exist, and building them
 * against fabricated data would violate the same grounding rule this
 * registry exists to enforce:
 *   - query_flood_stats / compare_time_periods: need more than one scene
 *     processed for the same region over time. Real Assam ingestion
 *     (spec section 15.4) is blocked on Earth Engine credentials.
 *   - get_model_confidence: needs a real uncertainty estimate (spec
 *     section 4.4's stretch goal -- MC Dropout / deep ensemble), not
 *     built.
 *   - generate_report: needs the PDF report pipeline (spec section 6.4),
 *     not built; ReportViewer.tsx's fields are still honest placeholders.
 */
export const NOT_YET_BUILT = [
  "query_flood_stats",
  "compare_time_periods",
  "get_model_confidence",
  "generate_report",
] as const;
