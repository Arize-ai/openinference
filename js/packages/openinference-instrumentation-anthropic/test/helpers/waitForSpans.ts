import type { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";

/**
 * Polls an in-memory exporter until it holds at least `count` finished spans.
 *
 * The instrumentation ends spans from a promise continuation, so a span is not
 * necessarily exported by the time the awaited SDK call resolves.
 */
export async function waitForSpans(exporter: InMemorySpanExporter, count: number) {
  for (let i = 0; i < 50; i++) {
    if (exporter.getFinishedSpans().length >= count) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
