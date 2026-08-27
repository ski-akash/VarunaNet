import { loadConfig } from "./config.js";
import { createRedisConnection } from "./redisConnection.js";
import { HttpInferenceClient } from "./inferenceClient.js";
import { ResultCache } from "./resultCache.js";
import { createSceneWorker } from "./sceneWorker.js";

// Separate process from the HTTP server (src/index.ts) -- run with
// `npm run worker`. This is what actually pulls jobs off the scene-
// processing queue and runs inference; the HTTP server only ever enqueues
// and reads results.
const config = loadConfig();
const redis = createRedisConnection(config.redisUrl);
const inference = new HttpInferenceClient(config.inferenceServiceUrl);
const cache = new ResultCache(redis, config.resultCacheTtlSeconds);

const worker = createSceneWorker(redis, inference, cache);

worker.on("completed", (job) => {
  console.log(`scene ${job.data.sceneId} processed`);
});
worker.on("failed", (job, err) => {
  console.error(`scene ${job?.data.sceneId} failed: ${err.message}`);
});
