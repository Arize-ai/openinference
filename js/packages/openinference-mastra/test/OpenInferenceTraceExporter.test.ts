import { OpenInferenceOTLPTraceExporter } from "../src/OpenInferenceTraceExporter.js";

describe("OpenInferenceTraceExporter", () => {
  it("should initialize without throwing an error", () => {
    new OpenInferenceOTLPTraceExporter({
      collectorEndpoint: "http://localhost:6006/v1/traces",
      apiKey: "test-api-key",
    });
  });
});
