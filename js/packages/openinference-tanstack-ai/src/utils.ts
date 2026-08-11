import type { Tool as TanStackTool } from "@tanstack/ai";

import type { OISpan } from "@arizeai/openinference-core";

/**
 * Extracts a human-readable tool description when TanStack provides one.
 */
export function getToolDescription(tool: TanStackTool | undefined): string | undefined {
  if (tool == null || typeof tool !== "object") {
    return undefined;
  }
  const description = Reflect.get(tool, "description");
  return typeof description === "string" ? description : undefined;
}

/**
 * Normalizes arbitrary tool arguments into an object for helper utilities.
 */
export function toRecord(value: unknown): Record<string, unknown> {
  if (value != null && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return { value };
}

/**
 * Ends a span if it was created.
 */
export function finalizeSpan(span: OISpan | undefined) {
  span?.end();
}
