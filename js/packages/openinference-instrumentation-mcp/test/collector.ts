import http from "http";

import express from "express";

// We use JSON instead of protobuf because opentelemetry-js does not publish proto
// generated code in a consumable form. We define the JSON interfaces here as
// needed for assertions.

export interface ExportedSpan {
  name: string;
  traceId: string;
  spanId: string;
  parentSpanId?: string;
}

interface Scope {
  name: string;
}

interface ScopeSpans {
  scope: Scope;
  spans: ExportedSpan[];
}

interface ResourceSpans {
  resource: object;
  scopeSpans: ScopeSpans[];
}

export class Telemetry {
  readonly resourceSpans: ResourceSpans[] = [];

  clear() {
    this.resourceSpans.length = 0;
  }
}

export async function startCollector(
  telemetry: Telemetry,
): Promise<http.Server> {
  const app = express();
  app.use(express.json());

  app.post("/v1/traces", (req, res) => {
    telemetry.resourceSpans.push(...req.body.resourceSpans);
    res.end("{}");
  });

  return new Promise((resolve, reject) => {
    const server = app.listen(0, (err) => {
      if (err) {
        reject(err);
      } else {
        resolve(server);
      }
    });
  });
}
