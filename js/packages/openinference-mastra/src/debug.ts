import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

let debugSpans: Pick<
  ReadableSpan,
  | "name"
  | "attributes"
  | "parentSpanContext"
  | "kind"
  | "status"
  | "resource"
  | "startTime"
  | "endTime"
>[] = [];

/**
 * Strictly for debugging purposes and not exposed as a package level export.
 *
 * You can place this in an exporter export function to capture mastra spans for debugging.
 *
 * Accumulate items across invocations until the item has no parentId, then dump items to json file
 * $HOME/debug-mastra-instrumentation/spans-{new Date().toISOString()}.json
 */
export const debug = async (spans: ReadableSpan[]) => {
  // only import fs if we need it
  // this allows the module to be used in environments that don't have fs
  const fs = await import("node:fs");
  debugSpans.push(
    // @ts-expect-error -just grabbing incomplete fields for testing
    ...spans
      .map((span) => ({
        name: span.name,
        attributes: span.attributes,
        parentSpanId: span.parentSpanContext?.spanId,
        kind: span.kind,
        status: span.status,
        resource: {},
        startTime: span.startTime,
        endTime: span.endTime,
      }))
      .filter((span) =>
        ["post", "agent", "ai"].some((prefix) =>
          span.name.toLocaleLowerCase().startsWith(prefix),
        ),
      ),
  );
  const root = spans.find((span) => span.parentSpanContext?.spanId == null);
  if (root) {
    fs.mkdirSync("debug-mastra-instrumentation", { recursive: true });
    fs.writeFileSync(
      `debug-mastra-instrumentation/${encodeURIComponent(root.name)}-${new Date().toISOString()}.json`,
      JSON.stringify(debugSpans, null, 2),
    );
    debugSpans = [];
  }
};
