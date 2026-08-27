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
