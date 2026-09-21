import { context, diag, propagation, trace, type Tracer } from "@opentelemetry/api";
import { W3CTraceContextPropagator } from "@opentelemetry/core";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { SimpleSpanProcessor, TracerProvider, type SpanProcessor } from "@opentelemetry/sdk-trace";

import { OITracer, type TraceConfigOptions } from "@arizeai/openinference-core";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import { AsyncLocalStorageContextManager } from "./context.js";
import { FetchTraceExporter, type FetchTraceExporterOptions } from "./exporter.js";
import { VERSION } from "./version.js";

/**
 * A tracer provider for a worker or Durable Object: the standard SDK with a fetch exporter,
 * a simple (not batching) processor because an isolate may go away between requests, the
 * AsyncLocalStorage context manager, and the OpenInference project name on the resource so
 * Phoenix files the traces where they belong.
 */
export interface WorkersTracingOptions {
  /** The collector base URL; `/v1/traces` is appended. */
  endpoint: string;
  /** The Phoenix project the traces land in. */
  projectName: string;
  /** `service.name` on the resource. */
  serviceName: string;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
  /** OpenInference trace config: masking of inputs, outputs, embeddings and so on. */
  traceConfig?: TraceConfigOptions;
  /** Extra processors, run before the exporter's. */
  spanProcessors?: SpanProcessor[];
  /** Maximum duration of a collector request, in milliseconds. */
  exportTimeoutMillis?: number;
  onExportError?: FetchTraceExporterOptions["onError"];
  /** Register the provider, context manager and W3C propagator globally. Default true. */
  register?: boolean;
}

export interface WorkersTracing {
  provider: TracerProvider;
  /** The OpenInference tracer: `startActiveSpan` with OpenInference span methods. */
  tracer: OITracer;
  /** The plain OpenTelemetry tracer, for code that wants the API surface only. */
  otelTracer: Tracer;
  /** Wait for every export in flight; hand this to `ctx.waitUntil` at the end of a request. */
  flush: () => Promise<void>;
  shutdown: () => Promise<void>;
}

export function createWorkersTracing(options: WorkersTracingOptions): WorkersTracing {
  const exporter = new FetchTraceExporter({
    url: `${options.endpoint.replace(/\/+$/, "")}/v1/traces`,
    headers: options.headers,
    fetch: options.fetch,
    onError: options.onExportError,
    timeoutMillis: options.exportTimeoutMillis,
  });
  const provider = new TracerProvider({
    resource: resourceFromAttributes({
      [SEMRESATTRS_PROJECT_NAME]: options.projectName,
      "service.name": options.serviceName,
    }),
    spanProcessors: [...(options.spanProcessors ?? []), new SimpleSpanProcessor({ exporter })],
  });
  if (options.register !== false) {
    const manager = new AsyncLocalStorageContextManager().enable();
    if (!context.setGlobalContextManager(manager)) {
      manager.disable();
      diag.warn("OpenInference Workers is using the existing global context manager");
    }
    if (!propagation.setGlobalPropagator(new W3CTraceContextPropagator())) {
      diag.warn("OpenInference Workers is using the existing global propagator");
    }
    if (!trace.setGlobalTracerProvider(provider)) {
      diag.warn(
        "OpenInference Workers could not register its provider; use the returned tracer explicitly",
      );
    }
  }
  const otelTracer = provider.getTracer("@arizeai/openinference-workers", VERSION);
  const tracer = new OITracer({ tracer: otelTracer, traceConfig: options.traceConfig });
  return {
    provider,
    tracer,
    otelTracer,
    flush: () => provider.forceFlush(),
    shutdown: () => provider.shutdown(),
  };
}
