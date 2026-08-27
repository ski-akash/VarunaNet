import type { FunctionDeclaration, LLMClient, Message } from "./types.js";

// Verified live against the real API before writing this (not assumed from
// training/documentation, which is stale here -- the model catalog moved on
// significantly): gemini-2.5-flash and gemini-2.5-flash-lite both return 404
// "no longer available to new users" on this key. gemini-3.5-flash-lite is
// the model actually confirmed working, including function calling and the
// thought_signature round-trip (see types.ts's FunctionCallPart comment).
const DEFAULT_MODEL = "gemini-3.5-flash-lite";

export class GeminiClient implements LLMClient {
  constructor(
    private readonly apiKey: string,
    private readonly model: string = DEFAULT_MODEL,
  ) {}

  async generateTurn(history: Message[], tools: FunctionDeclaration[]): Promise<Message> {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${this.model}:generateContent?key=${this.apiKey}`;
    const body = {
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
