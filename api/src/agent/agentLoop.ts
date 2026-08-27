import type { FunctionCallPart, LLMClient, Message } from "../llm/types.js";
import type { ToolRegistry } from "../tools/registry.js";
import { TOOL_DECLARATIONS, dispatchToolCall } from "./toolDeclarations.js";
import { checkGroundedness, type GroundednessCheck } from "../grounding/groundedness.js";

// Spec section 6: "one tool-calling agent over a common tool registry" for
// every AI feature. This loop is that agent -- it doesn't know or care
// whether it's answering a chat question, driving map control, or backing
// a severity explanation; all four are just different prompts over the
// same tool-calling turn.

const MAX_TOOL_ROUNDS = 4;

export interface ToolCallRecord {
  name: string;
  args: Record<string, unknown>;
  result: unknown;
}

export interface AgentTurnResult {
  responseText: string;
  toolCalls: ToolCallRecord[];
  groundedness: GroundednessCheck;
  history: Message[];
}

function isFunctionCallPart(part: Message["parts"][number]): part is FunctionCallPart {
  return "functionCall" in part;
}

export async function runAgentTurn(
  userMessage: string,
  history: Message[],
  llm: LLMClient,
  registry: ToolRegistry,
): Promise<AgentTurnResult> {
  const messages: Message[] = [...history, { role: "user", parts: [{ text: userMessage }] }];
  const toolCalls: ToolCallRecord[] = [];

  for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
    const turn = await llm.generateTurn(messages, TOOL_DECLARATIONS);
    const functionCallParts = turn.parts.filter(isFunctionCallPart);

    if (functionCallParts.length === 0) {
      // No more tool calls -- this is the model's final answer. Every
      // numeric claim in it must trace back to a tool result from this
      // turn (spec section 6.1's hard rule), checked here, at the point
      // the response is about to leave the agent loop, not left to a
      // caller who might forget to check.
      const responseText = turn.parts
        .filter((p): p is { text: string } => "text" in p)
        .map((p) => p.text)
        .join("");
      messages.push(turn);
      return {
        responseText,
        toolCalls,
        groundedness: checkGroundedness(
          responseText,
          toolCalls.map((c) => c.result),
        ),
        history: messages,
      };
    }

    // The model's message (carrying the functionCall parts, with their
    // thoughtSignature intact) goes back into history verbatim before the
    // results are appended -- Gemini requires seeing its own prior
    // functionCall exactly as issued, not a paraphrase of it.
    messages.push(turn);

    const responseParts: Message["parts"] = [];
    for (const part of functionCallParts) {
      const { name, args, id } = part.functionCall;
      let result: unknown;
      try {
        result = await dispatchToolCall(registry, name, args);
      } catch (err) {
        // A failed tool call becomes a functionResponse the model can see
        // and react to (e.g. "that scene hasn't been processed yet"),
        // not a thrown error that kills the whole turn.
        result = { error: (err as Error).message };
      }
      toolCalls.push({ name, args, result });
      responseParts.push({ functionResponse: { name, id, response: result } });
    }
    messages.push({ role: "user", parts: responseParts });
  }

  throw new Error(`agent loop exceeded ${MAX_TOOL_ROUNDS} tool-call rounds without a final answer`);
}
