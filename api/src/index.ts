import { buildServer } from "./server.js";
import { loadConfig } from "./config.js";
import { createRedisConnection } from "./redisConnection.js";

const config = loadConfig();
const redis = createRedisConnection(config.redisUrl);
const app = buildServer(config, undefined, redis);

app
  .listen({ port: config.port, host: config.host })
  .catch((err) => {
    app.log.error(err);
    process.exit(1);
  });
