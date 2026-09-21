import { context, diag } from "@opentelemetry/api";
import { ExportResultCode, suppressTracing, type ExportResult } from "@opentelemetry/core";
import { ProtobufTraceSerializer } from "@opentelemetry/otlp-transformer";
import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace";

export interface FetchTraceExporterOptions {
  /** Full OTLP traces endpoint, including /v1/traces. */
  url: string;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
  /** Maximum duration of a collector request. Defaults to 10 seconds. */
  timeoutMillis?: number;
  /** Errors are also reported through OpenTelemetry diagnostics. */
  onError?: (error: unknown) => void;
}

/** OTLP/HTTP protobuf transport using the runtime's fetch implementation. No retries. */
export class FetchTraceExporter implements SpanExporter {
  readonly #options: FetchTraceExporterOptions;
  readonly #fetch: typeof fetch;
  readonly #inFlight = new Set<Promise<void>>();
  #shutdown = false;

  constructor(options: FetchTraceExporterOptions) {
    if (
      options.timeoutMillis !== undefined &&
      (!Number.isFinite(options.timeoutMillis) || options.timeoutMillis <= 0)
    ) {
      throw new Error("timeoutMillis must be a positive finite number");
    }
    this.#options = { ...options, headers: { ...options.headers } };
    this.#fetch = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    const send = this.#send(spans).then((result) => {
      if (result.code === ExportResultCode.FAILED) {
        diag.warn("OpenInference export failed", result.error);
        try {
          this.#options.onError?.(result.error);
        } catch (error) {
          diag.warn("OpenInference export error callback failed", error);
        }
      }
      try {
        resultCallback(result);
      } catch (error) {
        diag.warn("OpenInference export result callback failed", error);
      }
    });
    this.#inFlight.add(send);
    void send.then(() => this.#inFlight.delete(send));
  }

  async #send(spans: ReadableSpan[]): Promise<ExportResult> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      if (this.#shutdown) throw new Error("exporter is shut down");
      const body = ProtobufTraceSerializer.serializeRequest(spans);
      if (!body) return { code: ExportResultCode.SUCCESS };
      const controller = new AbortController();
      timer = setTimeout(() => controller.abort(), this.#options.timeoutMillis ?? 10_000);
      const response = await context.with(suppressTracing(context.active()), () =>
        this.#fetch(this.#options.url, {
          method: "POST",
          headers: { ...this.#options.headers, "content-type": "application/x-protobuf" },
          body,
          signal: controller.signal,
        }),
      );
      // Release the response body so repeated exports do not retain connections.
      await response.body?.cancel();
      if (!response.ok) throw new Error(`collector answered ${response.status}`);
      return { code: ExportResultCode.SUCCESS };
    } catch (error) {
      return {
        code: ExportResultCode.FAILED,
        error: error instanceof Error ? error : new Error(String(error)),
      };
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
  }

  /** Wait for exports already in flight when this method is called. */
  async forceFlush(): Promise<void> {
    await Promise.all(this.#inFlight);
  }

  async shutdown(): Promise<void> {
    this.#shutdown = true;
    await this.forceFlush();
  }
}
