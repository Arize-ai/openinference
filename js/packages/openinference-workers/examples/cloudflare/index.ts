import { context, trace, SpanStatusCode } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { DurableObject } from "cloudflare:workers";

import {
  OITracer,
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
  instrumentAi,
  createWorkersTracing,
  injectTraceHeaders,
  withRequestSpan,
  type WorkersTracing,
} from "@arizeai/openinference-workers";

interface Env {
  AI?: Ai;
  AI_GATEWAY_ID?: string;
  AI_WORKER_URL?: string;
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
    if (new URL(request.url).pathname.startsWith("/ai")) {
      return context.with(
        setSession(context.active(), {
          sessionId: request.headers.get("x-example-session") ?? "ai-demo",
        }),
        () =>
          withRequestSpan(
            {
              tracer: t.tracer,
              request,
              name: "ai.object",
              kind: OpenInferenceSpanKind.CHAIN,
              execution: this.ctx,
              flush: t.flush,
            },
            () => aiResponse(request, this.env, this.ctx, t),
          ),
      );
    }
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
    ctx = setMetadata(ctx, {
      runtime: "workers",
      scenario: url.pathname.startsWith("/ai") ? "ai" : "counter",
    });
    ctx = setTags(ctx, ["example"]);
    if (url.pathname === "/suppressed" || url.pathname === "/ai/suppressed")
      ctx = suppressTracing(ctx);
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
            if (url.pathname.startsWith("/ai")) {
              if (url.pathname === "/ai/object") {
                const stub = env.COUNTERS.get(env.COUNTERS.idFromName("ai-demo"));
                return stub.fetch(`http://counter/ai${url.search}`, {
                  headers: injectTraceHeaders({
                    headers: { "x-example-session": url.searchParams.get("session") ?? "ai-demo" },
                  }),
                });
              }
              return aiResponse(request, env, execution, t);
            }
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

async function aiResponse(
  request: Request,
  env: Env,
  execution: Pick<ExecutionContext, "waitUntil">,
  t: WorkersTracing,
): Promise<Response> {
  const url = new URL(request.url);
  if (!env.AI) {
    if (!env.AI_WORKER_URL)
      return Response.json(
        { error: "Set AI_WORKER_URL to a deployed AI example for celld" },
        { status: 503 },
      );
    return fetch(new URL(url.pathname + url.search, env.AI_WORKER_URL), {
      headers: injectTraceHeaders(),
    });
  }
  const masked = url.pathname.startsWith("/ai/masked");
  const ai = instrumentAi({
    ai: env.AI,
    models: [
      "@cf/meta/llama-3.2-1b-instruct",
      "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
      "@cf/example/nonexistent-model",
    ],
    tracer: new OITracer({
      tracer: t.otelTracer,
      traceConfig: { hideInputs: masked, hideOutputs: masked },
    }),
    execution,
    flush: t.flush,
  });
  const input = {
    messages: [{ role: "user", content: "Say hello in one short sentence." }],
    max_tokens: 32,
  };
  const model = "@cf/meta/llama-3.2-1b-instruct";
  const options = env.AI_GATEWAY_ID ? { gateway: { id: env.AI_GATEWAY_ID } } : undefined;
  if (url.pathname === "/ai/error") await ai.run("@cf/example/nonexistent-model", input);
  if (url.pathname === "/ai/tools") {
    const result = await ai.run(
      "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
      {
        messages: [
          { role: "user", content: "What is the weather in Boston? Use the get_weather tool." },
        ],
        tools: [
          {
            type: "function",
            function: {
              name: "get_weather",
              description: "Get the current weather for a city",
              parameters: {
                type: "object",
                properties: { city: { type: "string", description: "City name" } },
                required: ["city"],
              },
            },
          },
        ],
        max_tokens: 128,
      },
      options,
    );
    return Response.json({ result, traceId: trace.getActiveSpan()?.spanContext().traceId });
  }
  if (url.pathname.endsWith("/stream")) {
    const stream = await ai.run(model, { ...input, stream: true }, options);
    return new Response(stream, {
      headers: {
        "content-type": "text/event-stream",
        "x-trace-id": trace.getActiveSpan()?.spanContext().traceId ?? "",
      },
    });
  }
  const result = await ai.run(model, input, options);
  return Response.json({ result, traceId: trace.getActiveSpan()?.spanContext().traceId });
}
