// Message/tool shapes the agent loop and every LLM adapter share. Shaped
// closely to Gemini's own turn structure today (parts, functionCall/
// functionResponse) rather than an invented abstraction, because
// "provider-agnostic" (spec section 6.7) is a real goal but a premature
// abstraction over a single working provider is worse than an honest,
// slightly-Gemini-shaped interface with one concrete implementation --
// the boundary that matters (LLMClient) already exists; a second
// provider adapter is what would prove it's the right one.

export interface FunctionDeclaration {
  name: string;
  description: string;
  // JSON-schema-ish (Gemini's OBJECT/STRING/INTEGER/... type names, not
  // full JSON Schema) -- matches what the Gemini API itself expects.
  parameters: {
    type: "OBJECT";
    properties: Record<string, { type: string; description?: string }>;
    required?: string[];
  };
}

export interface TextPart {
  text: string;
}

export interface FunctionCallPart {
  functionCall: { name: string; args: Record<string, unknown>; id?: string };
  // Gemini 3.x requires this to be echoed back verbatim on any later turn
  // that replays this functionCall part, or the API rejects the request
  // (INVALID_ARGUMENT: "missing a thought_signature") -- confirmed by a
  // live call, not assumed from documentation, since this is a real API
  // requirement that wasn't true of earlier Gemini model generations.
  thoughtSignature?: string;
}

export interface FunctionResponsePart {
  functionResponse: { name: string; id?: string; response: unknown };
}

export type MessagePart = TextPart | FunctionCallPart | FunctionResponsePart;

export interface Message {
  role: "user" | "model";
  parts: MessagePart[];
}

export interface LLMClient {
  generateTurn(history: Message[], tools: FunctionDeclaration[]): Promise<Message>;
}
