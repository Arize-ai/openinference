import { ReadableSpan } from "@opentelemetry/sdk-trace-base";

let debugSpans: Pick<
  ReadableSpan,
  | "name"
  | "attributes"
  | "parentSpanId"
  | "kind"
  | "status"
  | "resource"
  | "startTime"
  | "endTime"
>[] = [];

/**
 * DEBUG
 *
 * Accumulate items across invocations until the item has no parentId, then dump items to json file
 * $HOME/debug-mastra-instrumentation/spans-{new Date().toISOString()}.json
 */
export const debug = async (items: ReadableSpan[]) => {
  // only import fs if we need it
  // this allows the module to be used in environments that don't have fs
  const fs = await import("node:fs");
  debugSpans.push(
    // @ts-expect-error -just grabbing incomplete fields for testing
    ...items
      .map((i) => ({
        name: i.name,
        attributes: i.attributes,
        parentSpanId: i.parentSpanId,
        kind: i.kind,
        status: i.status,
        resource: {},
        startTime: i.startTime,
        endTime: i.endTime,
      }))
      .filter((i) =>
        ["post", "agent", "ai"].some((prefix) =>
          i.name.toLocaleLowerCase().startsWith(prefix),
        ),
      ),
  );
  const root = items.find((i) => i.parentSpanId == null);
  if (root) {
    fs.mkdirSync("debug-mastra-instrumentation", { recursive: true });
    fs.writeFileSync(
      `debug-mastra-instrumentation/${encodeURIComponent(root.name)}-${new Date().toISOString()}.json`,
      JSON.stringify(debugSpans, null, 2),
    );
    debugSpans = [];
  }
};
