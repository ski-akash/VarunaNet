import type { FastifyInstance } from "fastify";
import type { LLMClient } from "../llm/types.js";
import type { ToolRegistry } from "../tools/registry.js";
import { runAgentTurn } from "../agent/agentLoop.js";

interface ChatBody {
  message?: unknown;
}

// The one endpoint every AI feature (spec section 6) will eventually route
// through -- right now, natural-language chat. Feature 3 (conversational
// map control) is already reachable through this same endpoint today,
// since set_map_view is one of the three real tools: asking "zoom to
// Golaghat" will make the model call set_map_view and the response will
// carry that proposed view in tool_calls for the frontend to apply.
export function registerChatRoute(app: FastifyInstance, llm: LLMClient, registry: ToolRegistry): void {
  app.post<{ Body: ChatBody }>("/chat", async (req, reply) => {
    const message = req.body?.message;
    if (typeof message !== "string" || message.length === 0) {
      reply.code(400);
      return { error: "message is required" };
    }

    try {
      const result = await runAgentTurn(message, [], llm, registry);
      const grounded = result.groundedness.groundednessRate === 1;

      if (!grounded) {
        // Caught for real in testing, not a hypothetical: with an empty
        // database (no scenes processed yet), the model still answered
        // with a specific fabricated scene id, model name, percentages,
        // and district counts -- none of it from the (null) tool result.
        // spec section 6.1 is explicit that this is actively dangerous in
        // a disaster-response tool, so the fabricated text is REPLACED
        // here, not merely flagged and shown anyway -- a caller reading
        // only `response` (most callers) must never see numbers that
        // didn't come from a tool. The real tool_calls/results still go
        // out untouched, since those are what actually happened this
        // turn, not model output.
        app.log.warn(
          { groundedness: result.groundedness, original_response: result.responseText },
          "response contained a numeric claim not traceable to a tool result this turn -- replaced before sending",
        );
        return {
          response:
            "I can't verify part of that answer against real data, so I'm not going to state it. " +
            (result.toolCalls.length > 0
              ? "Here's exactly what the tools actually returned this turn -- expand the trace below."
              : "No tool was queried for this question, so there's no data behind it yet."),
          tool_calls: result.toolCalls,
          grounded: false,
          groundedness_rate: result.groundedness.groundednessRate,
        };
      }

      return {
        response: result.responseText,
        // result is included, not just name/args: set_map_view's result is
        // the actual proposed view state the frontend needs to apply
        // (spec section 6.5), and showing the other tools' real results is
        // what makes the tool-call trace a genuine "see the grounding"
        // demo moment (spec section 7) rather than a name-only log line.
        tool_calls: result.toolCalls,
        grounded: true,
        groundedness_rate: result.groundedness.groundednessRate,
      };
    } catch (err) {
      reply.code(502);
      return { error: (err as Error).message };
    }
  });
}
