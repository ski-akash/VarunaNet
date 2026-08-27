import type { FunctionDeclaration } from "../llm/types.js";
import type { ToolRegistry } from "../tools/registry.js";

// The bridge between the provider-agnostic ToolRegistry (src/tools/registry.ts,
// built without any LLM in mind) and Gemini's specific function-calling wire
// format. Kept separate from ToolRegistry itself so the registry stays a
// plain TypeScript API usable outside an LLM context too (a REST endpoint
// could call getWorstAffected() directly, for instance).
export const TOOL_DECLARATIONS: FunctionDeclaration[] = [
  {
    name: "get_scene_metadata",
    description:
      "Get metadata and summary statistics for a processed satellite scene: model used, " +
      "when it was processed, water pixel fraction, flood polygon count, and district " +
      "impact totals. If no scene_id is given, returns the most recently processed scene.",
    parameters: {
      type: "OBJECT",
      properties: {
        scene_id: { type: "STRING", description: "The scene identifier, e.g. India_900498" },
      },
    },
  },
  {
    name: "get_worst_affected",
    description:
      "Get the districts worst affected by flooding for a scene, ranked by percent of " +
      "district area flooded (not absolute hectares -- a small district mostly underwater " +
      "is worse than a large district with more total flooded area but a smaller share). " +
      "If no scene_id is given, uses the most recently processed scene.",
    parameters: {
      type: "OBJECT",
      properties: {
        top_n: { type: "INTEGER", description: "how many districts to return, default 10" },
        scene_id: { type: "STRING", description: "The scene identifier, e.g. India_900498" },
      },
    },
  },
  {
    name: "set_map_view",
    description:
      "Propose a map view change: which region/bounds to focus on, zoom level, which " +
      "layers to show, and which time index to display. This only proposes a view -- the " +
      "frontend validates and decides whether to apply it.",
    parameters: {
      type: "OBJECT",
      properties: {
        region: { type: "STRING", description: "Region name, e.g. Golaghat" },
        zoom: { type: "INTEGER" },
        layers: {
          type: "STRING",
          description:
            "Comma-separated layer names from: flood-extent, permanent-water, " +
            "confidence-heatmap, sar-backscatter",
        },
        time_index: { type: "INTEGER" },
      },
    },
  },
];

// Dispatches a model-requested function call to the real ToolRegistry
// method, translating Gemini's snake_case arg names to the registry's own
// camelCase params. Throws on an unknown tool name rather than silently
// no-op'ing -- the agent loop turns that into a functionResponse error the
// model can see and recover from, rather than a hallucinated success.
export async function dispatchToolCall(
  registry: ToolRegistry,
  name: string,
  args: Record<string, unknown>,
): Promise<unknown> {
  switch (name) {
    case "get_scene_metadata":
      return registry.getSceneMetadata({ sceneId: args.scene_id as string | undefined });
    case "get_worst_affected":
      return registry.getWorstAffected({
        topN: args.top_n as number | undefined,
        sceneId: args.scene_id as string | undefined,
      });
    case "set_map_view":
      return registry.setMapView({
        region: args.region as string | undefined,
        zoom: args.zoom as number | undefined,
        layers:
          typeof args.layers === "string"
            ? args.layers.split(",").map((l) => l.trim())
            : undefined,
        timeIndex: args.time_index as number | undefined,
      });
    default:
      throw new Error(`unknown tool: ${name}`);
  }
}
