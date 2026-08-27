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

      if (result.groundedness.groundednessRate < 1) {
        // Not blocked -- surfaced. Spec section 6.1's rule is that a
        // number must trace to a tool result; when the check catches a
        // miss, the honest response is to say so visibly (the frontend
        // shows this, not swallows it), rather than either silently
        // serving a possibly-hallucinated figure or hiding the failure
        // from a caller who has no other way to know it happened.
        app.log.warn(
          { groundedness: result.groundedness },
          "response contains a numeric claim not traceable to a tool result this turn",
        );
      }

      return {
        response: result.responseText,
        // result is included, not just name/args: set_map_view's result is
        // the actual proposed view state the frontend needs to apply
        // (spec section 6.5), and showing the other tools' real results is
        // what makes the tool-call trace a genuine "see the grounding"
        // demo moment (spec section 7) rather than a name-only log line.
        tool_calls: result.toolCalls,
        grounded: result.groundedness.groundednessRate === 1,
        groundedness_rate: result.groundedness.groundednessRate,
      };
    } catch (err) {
      reply.code(502);
      return { error: (err as Error).message };
    }
  });
}
