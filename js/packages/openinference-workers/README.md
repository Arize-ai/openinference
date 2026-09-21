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
      async () => new Response("Hello"),
    ).catch(() => new Response("Internal server error", { status: 500 }));
  },
};
```

`Env` is your application's generated bindings type. In a Durable Object, pass `this.ctx` as `execution`. Catch application errors at the HTTP boundary, as shown: the wrapper records and rethrows them, while the handler returns a response that lets background export complete.

## API

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

The request wrapper covers the handler promise, not consumption of a returned response stream. Finish streaming/background spans explicitly and schedule their final flush with the runtime's execution context. Durable Object `waitUntil` has different lifecycle semantics from Worker `waitUntil`; neither makes telemetry durable.

## Examples and verification

See [examples/README.md](./examples/README.md) for the real Worker → Durable Object counter example, the celld Docker runner, and a Phoenix read-back verification script. Both runtimes use the package's compiled workspace build.
