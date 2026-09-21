# OpenInference Workers

OpenInference tracing for Cloudflare Workers, Durable Objects, and celld. This package combines `OITracer`, an AsyncLocalStorage context manager, and an OTLP/HTTP protobuf exporter that uses `fetch`.

## Install

```sh
pnpm add @arizeai/openinference-workers @arizeai/openinference-core @arizeai/openinference-semantic-conventions @opentelemetry/api
```

Use a runtime with `node:async_hooks` support. The examples explicitly enable `nodejs_compat` for compatibility across workerd and celld.

## Quick start

Initialize once per isolate, lazily inside the handler when environment bindings are available:

```ts
import {
  createWorkersTracing,
  instrumentAi,
  withRequestSpan,
  type WorkersTracing,
} from "@arizeai/openinference-workers";
import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";

let tracing: WorkersTracing | undefined;

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext) {
    const t = (tracing ??= createWorkersTracing({
      endpoint: env.OTEL_ENDPOINT, // collector base URL; /v1/traces is appended
      projectName: "my-worker",
      serviceName: "my-worker",
      traceConfig: { hideInputs: true },
    }));
    return withRequestSpan(
      {
        tracer: t.tracer,
        request,
        name: "request",
        kind: OpenInferenceSpanKind.CHAIN,
        execution: ctx,
        flush: t.flush,
      },
      async () => {
        const model = "@cf/meta/llama-3.2-1b-instruct";
        const ai = instrumentAi({
          ai: env.AI,
          models: [model],
          tracer: t.tracer,
          execution: ctx,
          flush: t.flush,
        });
        const result = await ai.run(model, {
          messages: [{ role: "user", content: "Say hello." }],
          max_tokens: 32,
        });
        return Response.json(result);
      },
    ).catch(() => new Response("Internal server error", { status: 500 }));
  },
};
```

`Env` is your application's generated bindings type, including an `AI` binding configured in Wrangler. `models` lists the text-generation model IDs to trace; this prevents image, audio, and embedding requests from being mislabeled as LLM spans. Create the binding wrapper per request so it uses the current execution context. In a Durable Object, pass `this.ctx` as `execution`. Catch application errors at the HTTP boundary, as shown: the wrapper records and rethrows them, while the handler returns a response that lets background export complete.

## API

- `instrumentAi({ ai, models, tracer, execution, flush, maxStreamCaptureLength? })` returns a typed binding wrapper that records LLM spans for selected text-generation `run()` calls. It captures input/output messages, tool definitions and calls, invocation parameters, model, and reported token usage. It respects suppression and the supplied OITracer's masking and context configuration. It also supports selected third-party model IDs called through `AI.run` with Gateway options. Binding methods other than `run`, unselected models, raw-response, batch, and WebSocket modes pass through without LLM instrumentation.
- `createWorkersTracing(options)` returns an `OITracer`, provider, `flush()`, and `shutdown()`. Supply a collector base `endpoint`, `projectName`, and `serviceName`. Optional `headers` configure collector authentication; `traceConfig` controls masking; `spanProcessors` adds processors; `exportTimeoutMillis` defaults to 10 seconds. `onExportError` receives failed exports. The low-level `otelTracer` bypasses OpenInference masking and context attributes; use `tracer` for application spans.
- `withRequestSpan(options, fn)` extracts the inbound parent, activates an OpenInference span, records thrown exceptions, and ends the span when `fn` settles. It schedules flushing through `execution.waitUntil`, or awaits flushing when `execution` is absent. Flush errors are diagnostic events and do not replace the application's result or exception. A returned HTTP 500 is not a thrown exception: set the span status explicitly for such responses.
- `injectTraceHeaders({ headers?, ctx? })` returns a copy containing W3C trace headers. Pass it to `fetch` or `stub.fetch`.
- `extractTraceContext({ headers })` reads a remote parent without inheriting an ambient span when the headers are missing or malformed. Local OpenInference context attributes and tracing suppression are retained.
- `FetchTraceExporter` accepts a full OTLP traces URL, optional headers, timeout, custom fetch, and error callback. A custom fetch must honor the supplied abort signal.
- `AsyncLocalStorageContextManager` implements OpenTelemetry context propagation across awaits and bound functions.

Use core helpers such as `getInputAttributes`, `getOutputAttributes`, and `getToolAttributes` for span attributes. Use `setSession`, `setUser`, `setMetadata`, and `setTags` with `context.with` to attach request context. These attributes propagate within the runtime; W3C trace headers do not carry application metadata across HTTP boundaries.

## Registration and lifecycle

Global registration defaults to enabled. Existing global registrations are preserved; conflicts are reported through OpenTelemetry `diag`. Use `register: false` when the application already configures a compatible context manager and propagator, or when creating additional providers. The returned tracer always belongs to its returned provider. Shut down only when retiring the provider, not at the end of each request. Shutdown does not remove globals owned by the application.

The simple span processor starts one export per ended span. This avoids waiting for a batch timer, but adds a collector request per span. `flush` waits for pending exports; it does not guarantee delivery through runtime termination or a collector outage. Exports have a bounded timeout and no retries. Configure a `diag` logger in the application to see failures.

`instrumentAi` observes a returned SSE stream as the application consumes it, forwarding original bytes and cancellation. The LLM span ends at EOF, cancellation, or read failure; a separate `waitUntil` promise flushes it then. Consume or cancel the stream. Capture defaults to 65,536 decoded characters. Larger streams still reach the caller, but omit captured output and usage and set `cloudflare.ai.output_omitted`. The bound includes SSE framing. Provider failures are rethrown unchanged, while recorded error descriptions are generic to avoid leaking prompt text from error bodies.

The request wrapper covers the handler promise, not consumption of a returned response stream. Finish streaming/background spans explicitly and schedule their final flush with the runtime's execution context. Durable Object `waitUntil` has different lifecycle semantics from Worker `waitUntil`; neither makes telemetry durable.

## Examples and verification

See [examples/README.md](./examples/README.md) for real Workers AI calls (including streaming, tool calls, and calls from a Durable Object), celld → Cloudflare tracing, and Phoenix read-back verification. A separate counter example exercises runtime plumbing without model calls. Both runtimes use the package's compiled workspace build.
