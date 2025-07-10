import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { SpanKind } from "@opentelemetry/api";

// Helper function to verify response structure
export const verifyResponseStructure = (result: any) => {
  expect(result.body).toBeDefined();
  expect(result.contentType).toBe("application/json");
};

// Helper function to verify basic span structure and return the span
export const verifySpanBasics = (spanExporter: InMemorySpanExporter) => {
  const spans = spanExporter.getFinishedSpans();
  expect(spans).toHaveLength(1);

  const span = spans[0];
  expect(span.name).toBe("bedrock.invoke_model");
  expect(span.kind).toBe(SpanKind.CLIENT);

  return span;
};
