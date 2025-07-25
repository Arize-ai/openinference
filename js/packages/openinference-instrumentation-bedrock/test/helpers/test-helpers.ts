import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { SpanKind } from "@opentelemetry/api";

// Interface for API response structure verification
interface ApiResponse {
  body: unknown;
  contentType?: string;
}

// Helper function to verify response structure
export const verifyResponseStructure = (result: ApiResponse) => {
  expect(result.body).toBeDefined();
  // Only check contentType for non-streaming responses
  if (result.contentType !== undefined) {
    expect(result.contentType).toBe("application/json");
  }
};

// Helper function to consume a streaming response and wait for completion
export const consumeStreamResponse = async (result: {
  body: AsyncIterable<unknown> | undefined;
}) => {
  if (!result.body) {
    return;
  }

  // Consume the stream to trigger instrumentation
  for await (const _chunk of result.body) {
    // Just consume chunks, don't need to do anything with them
  }

  // Add a delay to ensure background processing completes
  await new Promise((resolve) => setTimeout(resolve, 500));
};

// Helper function to verify basic span structure and return the span
export const verifySpanBasics = (
  spanExporter: InMemorySpanExporter,
  expectedSpanName?: string,
) => {
  const spans = spanExporter.getFinishedSpans();
  expect(spans).toHaveLength(1);

  const span = spans[0];
  if (expectedSpanName) {
    expect(span.name).toBe(expectedSpanName);
  } else {
    // Default to original behavior for backward compatibility
    expect(span.name).toBe("bedrock.invoke_model");
  }
  expect(span.kind).toBe(SpanKind.INTERNAL);

  return span;
};
