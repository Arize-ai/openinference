import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { SpanKind, Context, context } from "@opentelemetry/api";
import {
  setSession,
  setUser,
  setMetadata,
  setTags,
  setPromptTemplate,
} from "@arizeai/openinference-core";

// Helper function to verify response structure
export const verifyResponseStructure = (result: any) => {
  expect(result.body).toBeDefined();
  // Only check contentType for non-streaming responses
  if (result.contentType !== undefined) {
    expect(result.contentType).toBe("application/json");
  }
};

// Helper function to verify basic span structure and return the span
export const verifySpanBasics = (spanExporter: InMemorySpanExporter, expectedSpanName?: string) => {
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

// Context helper functions for easier test setup
export interface TestContextOptions {
  sessionId?: string;
  userId?: string;
  metadata?: Record<string, any>;
  tags?: string[];
  promptTemplate?: {
    template: string;
    version?: string;
    variables?: Record<string, any>;
  };
}

/**
 * Creates a test context with all specified OpenInference attributes
 * @param options - Context configuration options
 * @returns Context with OpenInference attributes set
 */
export const createTestContext = (options: TestContextOptions = {}): Context => {
  let testContext = context.active();

  if (options.sessionId) {
    testContext = setSession(testContext, { sessionId: options.sessionId });
  }

  if (options.userId) {
    testContext = setUser(testContext, { userId: options.userId });
  }

  if (options.metadata) {
    testContext = setMetadata(testContext, options.metadata);
  }

  if (options.tags) {
    testContext = setTags(testContext, options.tags);
  }

  if (options.promptTemplate) {
    testContext = setPromptTemplate(testContext, options.promptTemplate);
  }

  return testContext;
};

/**
 * Executes a function within a test context with OpenInference attributes
 * @param options - Context configuration options
 * @param fn - Function to execute within the context
 * @returns Promise resolving to the function's return value
 */
export const withTestContext = async <T>(
  options: TestContextOptions,
  fn: () => Promise<T>
): Promise<T> => {
  const testContext = createTestContext(options);
  return context.with(testContext, fn);
};

/**
 * Creates a comprehensive test context with commonly used attributes
 * @param testName - Name of the test for unique identifiers
 * @returns Context with standard test attributes
 */
export const createStandardTestContext = (testName: string): Context => {
  return createTestContext({
    sessionId: `test-session-${testName}`,
    userId: `test-user-${testName}`,
    metadata: {
      test_name: testName,
      experiment_name: "bedrock-context-test",
      version: "1.0.0",
      environment: "testing"
    },
    tags: ["test", "bedrock", "context"],
    promptTemplate: {
      template: "You are a helpful assistant. User message: {{message}}",
      version: "1.0.0",
      variables: { message: "test message" }
    }
  });
};
