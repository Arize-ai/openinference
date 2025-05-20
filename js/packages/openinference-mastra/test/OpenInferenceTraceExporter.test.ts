import { ReadableSpan } from "@opentelemetry/sdk-trace-base";
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
      collectorEndpoint: "http://example.com/v1/traces",
      apiKey: "test-api-key",
    });
  });

  // Quickly capture a known working state of the instrumentation to ensure
  // we don't regress.
  // TODO: Replace with a more fine-grained test that is easier to update over
  // time with the changes in the instrumentation.
  it("(snapshot) should export spans with openinference properties", async () => {
    const exporter = new OpenInferenceOTLPTraceExporter({
      collectorEndpoint: "http://example.com/v1/traces",
      apiKey: "test-api-key",
    });
    exporter.export(weatherAgentSpans as unknown as ReadableSpan[], () => {});
    await expect(
      // @ts-expect-error - mock.calls is provided by vitest
      OTLPTraceExporter.prototype.export.mock.calls,
    ).toMatchFileSnapshot(
      `./__snapshots__/OpenInferenceTraceExporter.test.ts.export.json`,
    );
  });
});
