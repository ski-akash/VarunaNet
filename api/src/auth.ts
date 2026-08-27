import type { FastifyInstance } from "fastify";

// A simple shared-key check (spec section 5: "auth (a simple API key) if
// anything" -- this is a single-role dashboard, not multi-tenant, so one
// shared secret per deployment is enough; no user accounts, no per-key
// scoping). Checked in an onRequest hook rather than per-route, so a new
// route added later is protected by default instead of needing an opt-in
// that's easy to forget.
export function registerAuth(app: FastifyInstance, apiKeys: string[]): void {
  if (apiKeys.length === 0) {
    // Auth off is a deliberate, explicit state (see config.ts) for local
    // dev -- not a silent gap, so it's logged once at startup rather than
    // just working quietly in a way that's easy to forget is insecure.
    app.log.warn("API_KEYS is not set -- the gateway is accepting unauthenticated requests");
    return;
  }

  app.addHook("onRequest", async (req, reply) => {
    // /health is exempt: infra health checks and the frontend's own status
    // dot (frontend/src/components/BackendStatus.tsx) need to work without
    // a key, and it reveals nothing sensitive -- reachability and which
    // model is loaded, not any flood data.
    if (req.url === "/health") return;

    const key = req.headers["x-api-key"];
    if (typeof key !== "string" || !apiKeys.includes(key)) {
      reply.code(401);
      throw new Error("missing or invalid API key");
    }
  });
}
