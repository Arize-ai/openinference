import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { OpenInferenceOTLPTraceExporter } from "../src/OpenInferenceTraceExporter.js";
import weatherAgentSpans from "./__fixtures__/weatherAgentSpans.json";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";

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
});
