// Central place for the gateway's own runtime config -- every value here is
// read once, from the environment, with an explicit default. No caller reads
// process.env directly, so there's exactly one place that has to change if a
// deployment target (docker compose, the cluster, a laptop) needs different
// values.
export interface GatewayConfig {
  port: number;
  host: string;
  inferenceServiceUrl: string;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): GatewayConfig {
  return {
    port: Number(env.PORT ?? 3000),
    host: env.HOST ?? "0.0.0.0",
    inferenceServiceUrl: env.INFERENCE_SERVICE_URL ?? "http://localhost:8000",
  };
}
