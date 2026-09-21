import { context, trace, SpanStatusCode } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { DurableObject } from "cloudflare:workers";

import {
  getInputAttributes,
  getOutputAttributes,
  getToolAttributes,
  setSession,
  setUser,
  setMetadata,
  setTags,
} from "@arizeai/openinference-core";
import {
  OpenInferenceSpanKind,
  MimeType,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import {
  createWorkersTracing,
  injectTraceHeaders,
  withRequestSpan,
  type WorkersTracing,
} from "@arizeai/openinference-workers";

interface Env {
  COUNTERS: DurableObjectNamespace<Counter>;
  OTEL_ENDPOINT: string;
  OTEL_AUTHORIZATION?: string;
  PHOENIX_PROJECT: string;
}
let tracing: WorkersTracing | undefined;
function tracingFor(env: Env) {
  return (tracing ??= createWorkersTracing({
    endpoint: env.OTEL_ENDPOINT,
    headers: env.OTEL_AUTHORIZATION ? { authorization: env.OTEL_AUTHORIZATION } : undefined,
    projectName: env.PHOENIX_PROJECT,
    serviceName: "openinference-workers-example",
    traceConfig: { hideInputs: true },
  }));
}
export class Counter extends DurableObject<Env> {
  async fetch(request: Request) {
    const t = tracingFor(this.env);
    return withRequestSpan(
      {
        tracer: t.tracer,
        request,
        name: "counter.request",
        kind: OpenInferenceSpanKind.CHAIN,
        execution: this.ctx,
        flush: t.flush,
      },
      async () => {
        return t.tracer.startActiveSpan(
          "counter.increment",
          {
            attributes: {
              ...getToolAttributes({ name: "increment", parameters: {} }),
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
            },
          },
          async (span) => {
            try {
              const count = ((await this.ctx.storage.get<number>("count")) ?? 0) + 1;
              await this.ctx.storage.put("count", count);
              span.setAttributes(getOutputAttributes(String(count)));
              return Response.json({ count, traceId: span.spanContext().traceId });
            } finally {
              span.end();
            }
          },
        );
      },
    );
  }
}
export default {
  async fetch(request: Request, env: Env, execution: ExecutionContext): Promise<Response> {
    const t = tracingFor(env);
    const url = new URL(request.url);
    let ctx = setSession(context.active(), {
      sessionId: url.searchParams.get("session") ?? "workers-demo",
    });
    ctx = setUser(ctx, { userId: "example-user" });
    ctx = setMetadata(ctx, { runtime: "workers", scenario: "counter" });
    ctx = setTags(ctx, ["example"]);
    if (url.pathname === "/suppressed") ctx = suppressTracing(ctx);
    return context
      .with(ctx, () =>
        withRequestSpan(
          {
            tracer: t.tracer,
            request,
            name: "worker.request",
            kind: OpenInferenceSpanKind.CHAIN,
            execution,
            flush: t.flush,
            attributes: getInputAttributes("private example input"),
          },
          async (span) => {
            if (url.pathname === "/suppressed") return Response.json({ suppressed: true });
            if (url.pathname === "/status-error") {
              span.setStatus({ code: SpanStatusCode.ERROR, message: "Explicit failure" });
              return Response.json({ error: "Explicit failure" }, { status: 500 });
            }
            if (url.pathname === "/error") throw new Error("Example application failure");
            const stub = env.COUNTERS.get(
              env.COUNTERS.idFromName(url.searchParams.get("counter") ?? "demo"),
            );
            const response = await stub.fetch("http://counter/increment", {
              headers: injectTraceHeaders(),
            });
            const result: unknown = await response.json();
            span.setAttributes(
              getOutputAttributes({ value: JSON.stringify(result), mimeType: MimeType.JSON }),
            );
            return Response.json({ result, traceId: trace.getActiveSpan()?.spanContext().traceId });
          },
        ),
      )
      .catch(() => Response.json({ error: "Example application failure" }, { status: 500 }));
  },
} satisfies ExportedHandler<Env>;
