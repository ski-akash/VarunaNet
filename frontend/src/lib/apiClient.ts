// The one place the frontend talks to the Node gateway (api/) from. Every
// other component that needs backend data should go through this module
// rather than calling fetch() directly, so there's one place to add auth
// headers/error handling once Phase 6 needs them.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:3000'

export interface GatewayHealth {
  status: 'ok' | 'degraded'
  inference: { status: string; model?: string } | null
}

export async function getGatewayHealth(signal?: AbortSignal): Promise<GatewayHealth> {
  const res = await fetch(`${API_BASE_URL}/health`, { signal })
  // The gateway itself returns 503 (not 200) when it's up but can't reach
  // the inference service -- that's still a real, parseable health body,
  // not a network failure, so it's read either way rather than thrown.
  return (await res.json()) as GatewayHealth
}

export interface ToolCall {
  name: string
  args: Record<string, unknown>
  result: unknown
}

export interface ChatResponse {
  response: string
  tool_calls: ToolCall[]
  grounded: boolean
  groundedness_rate: number
}

export class ChatUnavailableError extends Error {}

// Feature 1 (natural-language query) and feature 3 (conversational map
// control, via the set_map_view tool) both go through this one endpoint --
// spec section 6's "one tool-calling agent over a common tool registry."
export async function sendChatMessage(message: string, signal?: AbortSignal): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ message }),
    signal,
  })
  if (res.status === 404) {
    // The gateway only registers /chat when it has both a database and an
    // LLM provider key configured (see api/src/server.ts) -- a 404 here
    // means "not configured," not "this message failed."
    throw new ChatUnavailableError('Chat is not configured on the gateway yet.')
  }
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { error?: string }
    throw new Error(body.error ?? `chat request failed with status ${res.status}`)
  }
  return (await res.json()) as ChatResponse
}
