import { registerOTel } from "@vercel/otel";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import {
  isOpenInferenceSpan,
  OpenInferenceSimpleSpanProcessor,
} from "@arizeai/openinference-vercel";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

export function register() {
  const spaceId = process.env.ARIZE_SPACE_ID;
  const apiKey = process.env.ARIZE_API_KEY;

  if (!spaceId || !apiKey) {
    throw new Error("ARIZE_SPACE_ID and ARIZE_API_KEY must be set");
  }

  registerOTel({
    serviceName: "parker-vercel-test",
    attributes: {
      model_id: "parker-vercel-model",
      model_version: "1.0.0",
    },
    spanProcessors: [
      new OpenInferenceSimpleSpanProcessor({
        exporter: new OTLPTraceExporter({
          url: "https://otlp.arize.com/v1/traces",
          headers: {
            space_id: spaceId,
            api_key: apiKey,
          },
        }),
        spanFilter: (span) => isOpenInferenceSpan(span),
      }),
    ],
  });
}
