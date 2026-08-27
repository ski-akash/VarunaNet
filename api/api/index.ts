// Vercel serverless entrypoint. A file under a top-level `api/` directory
// (this one, relative to this package's own root -- the Vercel project is
// rooted at the `api/` package, not the repo root) becomes a Node.js
// serverless function; `vercel.json`'s rewrite sends every path here, so
// `/health`, `/chat`, etc. all resolve to real clean paths rather than
// living under `/api/index`.
//
// Deliberately built WITHOUT Redis: this deployment exists specifically to
// get the chat feature (and /health, /predict) reachable publicly, and
// neither needs Redis or the BullMQ worker -- only /scenes does (see
// server.ts: it gates scene routes on `redis && pgPool`, so omitting redis
// here just means those routes don't register, not that anything breaks).
// The worker/Redis/inference-service deployment is a real, separate,
// deferred piece (see VarunaNet_Spec.md), not silently dropped.
import type { IncomingMessage, ServerResponse } from "node:http";
import { buildServer } from "../src/server.js";
import { loadConfig } from "../src/config.js";
import { createPgPool } from "../src/db/pool.js";
import type { FastifyInstance } from "fastify";

let appPromise: Promise<FastifyInstance> | null = null;

async function buildReadyApp(): Promise<FastifyInstance> {
  const config = loadConfig();
  const pgPool = createPgPool(config.databaseUrl);
  const app = buildServer(config, undefined, undefined, pgPool);
  await app.ready();
  return app;
}

function getApp(): Promise<FastifyInstance> {
  // Built once per warm serverless instance, not per request -- constructing
  // the Fastify app and its Postgres pool on every invocation would be both
  // slow and would leak a new pool each time.
  if (!appPromise) {
    appPromise = buildReadyApp();
  }
  return appPromise;
}

export default async function handler(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const app = await getApp();
  app.server.emit("request", req, res);
}
