import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { OpenInferenceOTLPTraceExporter } from "../src/OpenInferenceTraceExporter.js";
import weatherAgentSpans from "./__fixtures__/weatherAgentSpans.json";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

vi.mock(import("@opentelemetry/exporter-trace-otlp-proto"), () => {
  const mockedClass = vi.fn();
  mockedClass.prototype.export = vi.fn();
  return {
    OTLPTraceExporter: mockedClass,
  };
});

describe("OpenInferenceTraceExporter", () => {
  afterEach(() => {
    vi.resetAllMocks();
  });

  it("should initialize without throwing an error", () => {
    new OpenInferenceOTLPTraceExporter({
      url: "http://example.com/v1/traces",
      headers: {
        Authorization: "Bearer test-api-key",
      },
    });
  });

  // Quickly capture a known working state of the instrumentation to ensure
  // we don't regress.
  // TODO: Replace with a more fine-grained test that is easier to update over
  // time with the changes in the instrumentation.
  it("(snapshot) should export spans with openinference properties", async () => {
    const exporter = new OpenInferenceOTLPTraceExporter({
      url: "http://example.com/v1/traces",
      headers: {
        Authorization: "Bearer test-api-key",
      },
    });
    exporter.export(weatherAgentSpans as unknown as ReadableSpan[], () => {});
    await expect(
      // @ts-expect-error - mock.calls is provided by vitest
      OTLPTraceExporter.prototype.export.mock.calls,
    ).toMatchFileSnapshot(
      `./__snapshots__/OpenInferenceTraceExporter.test.ts.export.json`,
    );
  });

  it("should filter spans based on the spanFilter function", async () => {
    const exporter = new OpenInferenceOTLPTraceExporter({
      url: "http://example.com/v1/traces",
      headers: {
        Authorization: "Bearer test-api-key",
      },
      spanFilter: (span) =>
        span.name === "POST /api/agents/weatherAgent/stream",
    });
    exporter.export(weatherAgentSpans as unknown as ReadableSpan[], () => {});
    expect(
      // @ts-expect-error - mock.calls is provided by vitest
      OTLPTraceExporter.prototype.export.mock.calls,
    ).toMatchInlineSnapshot(`
      [
        [
          [
            {
              "attributes": {
                "http.flavor": "1.1",
                "http.host": "localhost:4111",
                "http.method": "POST",
                "http.request_content_length_uncompressed": 251,
                "http.request_id": "98823c5f-b2ec-4a4e-a056-f22f7bcb53ae",
                "http.scheme": "http",
                "http.status_code": 200,
                "http.status_text": "OK",
                "http.target": "/api/agents/weatherAgent/stream",
                "http.url": "http://localhost:4111/api/agents/weatherAgent/stream",
                "http.user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
                "net.host.ip": "::1",
                "net.host.name": "localhost",
                "net.host.port": 4111,
                "net.peer.ip": "::1",
                "net.peer.port": 51258,
                "net.transport": "ip_tcp",
                "openinference.span.kind": "AGENT",
              },
              "endTime": [
                1747754797,
                193654459,
              ],
              "kind": 1,
              "name": "POST /api/agents/weatherAgent/stream",
              "resource": {
                "attributes": {
                  "openinference.project.name": "mock",
                  "service.name": "mock",
                },
              },
              "startTime": [
                1747754793,
                713000000,
              ],
              "status": {
                "code": 0,
              },
            },
          ],
          [Function],
        ],
      ]
    `);
  });

  it("should map threadId to SESSION_ID attribute", async () => {
    const mockSpan = {
      name: "agent.test",
      attributes: {
        threadId: "test-thread-id-123",
        "other.attribute": "value",
      },
      resource: {
        attributes: {
          "service.name": "test-service",
        },
      },
    } as unknown as ReadableSpan;

    const exporter = new OpenInferenceOTLPTraceExporter({
      url: "http://example.com/v1/traces",
      headers: {
        Authorization: "Bearer test-api-key",
      },
    });

    exporter.export([mockSpan], () => {});

    // Check that the threadId was mapped to SESSION_ID
    expect(mockSpan.attributes[SemanticConventions.SESSION_ID]).toBe(
      "test-thread-id-123",
    );
    // Original threadId should still be present
    expect(mockSpan.attributes.threadId).toBe("test-thread-id-123");
    // Other attributes should remain unchanged
    expect(mockSpan.attributes["other.attribute"]).toBe("value");
  });

  it("should map numeric threadId to SESSION_ID attribute", async () => {
    const mockSpan = {
      name: "agent.test",
      attributes: {
        threadId: 123456, // numeric threadId
        "other.attribute": "value",
      },
      resource: {
        attributes: {
          "service.name": "test-service",
        },
      },
    } as unknown as ReadableSpan;

    const exporter = new OpenInferenceOTLPTraceExporter({
      url: "http://example.com/v1/traces",
      headers: {
        Authorization: "Bearer test-api-key",
      },
    });

    exporter.export([mockSpan], () => {});

    // Should have SESSION_ID with numeric threadId
    expect(mockSpan.attributes[SemanticConventions.SESSION_ID]).toBe(123456);
    // Original threadId should still be present
    expect(mockSpan.attributes.threadId).toBe(123456);
  });

  it("should not add SESSION_ID if threadId is not string or number", async () => {
    const mockSpan = {
      name: "agent.test",
      attributes: {
        threadId: true, // boolean instead of string/number
        "other.attribute": "value",
      },
      resource: {
        attributes: {
          "service.name": "test-service",
        },
      },
    } as unknown as ReadableSpan;

    const exporter = new OpenInferenceOTLPTraceExporter({
      url: "http://example.com/v1/traces",
      headers: {
        Authorization: "Bearer test-api-key",
      },
    });

    exporter.export([mockSpan], () => {});

    // Should not have SESSION_ID since threadId is boolean
    expect(mockSpan.attributes[SemanticConventions.SESSION_ID]).toBeUndefined();
  });

  it("should not add SESSION_ID if threadId is missing", async () => {
    const mockSpan = {
      name: "agent.test",
      attributes: {
        "other.attribute": "value",
      },
      resource: {
        attributes: {
          "service.name": "test-service",
        },
      },
    } as unknown as ReadableSpan;

    const exporter = new OpenInferenceOTLPTraceExporter({
      url: "http://example.com/v1/traces",
      headers: {
        Authorization: "Bearer test-api-key",
      },
    });

    exporter.export([mockSpan], () => {});

    // Should not have SESSION_ID since threadId is missing
    expect(mockSpan.attributes[SemanticConventions.SESSION_ID]).toBeUndefined();
  });

  it("should not overwrite existing SESSION_ID when threadId is present", async () => {
    const mockSpan = {
      name: "agent.test",
      attributes: {
        threadId: "new-thread-123",
        [SemanticConventions.SESSION_ID]: "existing-session-456", // Pre-existing SESSION_ID
      },
      resource: {
        attributes: {
          "service.name": "test-service",
        },
      },
    } as unknown as ReadableSpan;

    const exporter = new OpenInferenceOTLPTraceExporter({
      url: "http://example.com/v1/traces",
      headers: {
        Authorization: "Bearer test-api-key",
      },
    });

    exporter.export([mockSpan], () => {});

    // Should preserve existing SESSION_ID, not overwrite with threadId
    expect(mockSpan.attributes[SemanticConventions.SESSION_ID]).toBe(
      "existing-session-456",
    );
    // threadId should remain unchanged
    expect(mockSpan.attributes.threadId).toBe("new-thread-123");
  });

  it("should add AGENT span kind to root spans without span kind", async () => {
    const rootSpan = {
      name: "POST /api/agents/weatherAgent/stream",
      parentSpanContext: undefined, // Root span has no parent
      attributes: {
        "http.method": "POST",
        "http.url": "/api/agents/weatherAgent/stream",
      },
      resource: {
        attributes: {
          "service.name": "test-service",
        },
      },
    } as unknown as ReadableSpan;

    const exporter = new OpenInferenceOTLPTraceExporter({
      url: "http://example.com/v1/traces",
      headers: {
        Authorization: "Bearer test-api-key",
      },
    });

    exporter.export([rootSpan], () => {});

    // Root span should now have AGENT span kind set
    expect(rootSpan.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
      "AGENT",
    );
  });

  it("should not overwrite existing span kind on root spans", async () => {
    const rootSpan = {
      name: "POST /api/agents/weatherAgent/stream",
      parentSpanContext: undefined, // Root span has no parent
      attributes: {
        "http.method": "POST",
        "http.url": "/api/agents/weatherAgent/stream",
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: "CHAIN", // Pre-existing span kind
      },
      resource: {
        attributes: {
          "service.name": "test-service",
        },
      },
    } as unknown as ReadableSpan;

    const exporter = new OpenInferenceOTLPTraceExporter({
      url: "http://example.com/v1/traces",
      headers: {
        Authorization: "Bearer test-api-key",
      },
    });

    exporter.export([rootSpan], () => {});

    // Should preserve existing span kind, not overwrite with AGENT
    expect(rootSpan.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
      "CHAIN",
    );
  });

  it("should not add span kind to child spans without span kind", async () => {
    const childSpan = {
      name: "agent.getMostRecentUserMessage",
      parentSpanContext: { spanId: "parent-span-id" }, // Child span has parent
      attributes: {
        "agent.getMostRecentUserMessage.argument.0": "some data",
        "agent.getMostRecentUserMessage.result": "result data",
      },
      resource: {
        attributes: {
          "service.name": "test-service",
        },
      },
    } as unknown as ReadableSpan;

    const exporter = new OpenInferenceOTLPTraceExporter({
      url: "http://example.com/v1/traces",
      headers: {
        Authorization: "Bearer test-api-key",
      },
    });

    exporter.export([childSpan], () => {});

    // Child span should get AGENT span kind from the existing logic based on name pattern
    expect(childSpan.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
      "AGENT",
    );
  });
});
