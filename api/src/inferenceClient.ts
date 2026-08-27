// Thin wrapper around the Python inference service (inference/service.py).
// The gateway never runs the model itself -- it only ever forwards a scene_id
// and relays back whatever the Python side computed. Keeping this in one
// small class (rather than calling fetch() inline in each route) means the
// routes can be tested against a fake client with no real HTTP involved.
export interface HealthResponse {
  status: string;
  model?: string;
  [key: string]: unknown;
}

export interface PredictResponse {
  scene_id: string;
  [key: string]: unknown;
}

export class InferenceServiceUnavailableError extends Error {
  constructor(cause: unknown) {
    super("inference service unreachable");
    this.cause = cause;
  }
}

export interface InferenceClient {
  health(): Promise<HealthResponse>;
  predict(sceneId: string): Promise<PredictResponse>;
}

export class HttpInferenceClient implements InferenceClient {
  constructor(private readonly baseUrl: string) {}

  async health(): Promise<HealthResponse> {
    return this.get<HealthResponse>("/health");
  }

  async predict(sceneId: string): Promise<PredictResponse> {
    let res: Response;
    try {
      res = await fetch(new URL("/predict", this.baseUrl), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ scene_id: sceneId }),
      });
    } catch (err) {
      throw new InferenceServiceUnavailableError(err);
    }
    if (!res.ok) {
      throw new Error(`inference service returned ${res.status}: ${await res.text()}`);
    }
    return (await res.json()) as PredictResponse;
  }

  private async get<T>(path: string): Promise<T> {
    let res: Response;
    try {
      res = await fetch(new URL(path, this.baseUrl));
    } catch (err) {
      throw new InferenceServiceUnavailableError(err);
    }
    if (!res.ok) {
      throw new Error(`inference service returned ${res.status}`);
    }
    return (await res.json()) as T;
  }
}
