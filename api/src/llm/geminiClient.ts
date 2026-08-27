import type { FunctionDeclaration, LLMClient, Message } from "./types.js";

// Verified live against the real API before writing this (not assumed from
// training/documentation, which is stale here -- the model catalog moved on
// significantly): gemini-2.5-flash and gemini-2.5-flash-lite both return 404
// "no longer available to new users" on this key. gemini-3.5-flash-lite is
// the model actually confirmed working, including function calling and the
// thought_signature round-trip (see types.ts's FunctionCallPart comment).
const DEFAULT_MODEL = "gemini-3.5-flash-lite";

// spec section 6.1's hard rule, given to the model directly, not just
// checked after the fact. Added after a real live failure: with an empty
// database (no scene ever processed), the model answered with a specific
// fabricated scene id ("India_900498"), analysis date, and district count
// -- ai-eval/api's numeric groundedness checker didn't catch the scene id
// itself (it deliberately excludes chip-id-shaped tokens from numeric
// checks, since a *real* chip id like that appearing in real tool output
// is not a claim to verify) so the checker alone is not sufficient here.
// This system instruction is the first line of defense; the numeric
// checker in src/routes/chat.ts and its full-response replacement remain
// the enforced backstop for whatever still gets through.
const GROUNDING_SYSTEM_INSTRUCTION =
  "You are VarunaNet's flood-data assistant. You may state a scene id, date, district " +
  "name, or number ONLY if it appears verbatim in a tool result from this turn. If a " +
  "tool result is null, empty, or missing a field, say so plainly (e.g. 'no scene has " +
  "been processed yet') -- never invent a plausible-sounding scene id, date, or figure " +
  "to fill the gap. When you are unsure or have no data, say that directly rather than " +
  "answering with confident-sounding invented specifics.";

export class GeminiClient implements LLMClient {
  constructor(
    private readonly apiKey: string,
    private readonly model: string = DEFAULT_MODEL,
  ) {}

  async generateTurn(history: Message[], tools: FunctionDeclaration[]): Promise<Message> {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${this.model}:generateContent?key=${this.apiKey}`;
    const body = {
      systemInstruction: { parts: [{ text: GROUNDING_SYSTEM_INSTRUCTION }] },
      contents: history,
      // Omitted entirely (not an empty array) when there are no tools --
      // Gemini's own docs and this project's live testing both treat an
      // empty tools array as "no tools" too, but omitting is unambiguous.
      ...(tools.length > 0 ? { tools: [{ functionDeclarations: tools }] } : {}),
    };

    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      throw new Error(`Gemini API error ${res.status}: ${await res.text()}`);
    }

    const parsed = (await res.json()) as {
      candidates?: { content: { role: "model"; parts: Message["parts"] } }[];
    };
    const candidate = parsed.candidates?.[0];
    if (!candidate) {
      throw new Error(`Gemini returned no candidates: ${JSON.stringify(parsed)}`);
    }
    return { role: "model", parts: candidate.content.parts };
  }
}
