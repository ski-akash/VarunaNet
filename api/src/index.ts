import { buildServer } from "./server.js";
import { loadConfig } from "./config.js";

const config = loadConfig();
const app = buildServer(config);

app
  .listen({ port: config.port, host: config.host })
  .catch((err) => {
    app.log.error(err);
    process.exit(1);
  });
