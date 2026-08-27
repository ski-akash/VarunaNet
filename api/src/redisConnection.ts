import { Redis } from "ioredis";

// BullMQ requires maxRetriesPerRequest: null on the connection it's handed
// (it does its own retry/backoff internally; a finite value here makes
// BullMQ's blocking calls throw instead of just waiting) -- this factory
// exists so every caller gets that right rather than re-discovering it.
export function createRedisConnection(redisUrl: string): Redis {
  return new Redis(redisUrl, { maxRetriesPerRequest: null });
}
