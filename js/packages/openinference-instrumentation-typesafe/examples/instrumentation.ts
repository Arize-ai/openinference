import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { ConsoleSpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import * as TypeSafe from "@typesafe-ai/sdk";

import { TypeSafeInstrumentation } from "../src";

export function setupTracing(projectName: string) {
  const provider = new NodeTracerProvider({
    resource: resourceFromAttributes({ [SEMRESATTRS_PROJECT_NAME]: projectName }),
    spanProcessors: [
      new SimpleSpanProcessor(new ConsoleSpanExporter()),
      new SimpleSpanProcessor(new OTLPTraceExporter({ url: "http://localhost:6006/v1/traces" })),
    ],
  });
  provider.register();
  const instrumentation = new TypeSafeInstrumentation({ tracerProvider: provider });
  instrumentation.manuallyInstrument(TypeSafe);
  return {
    provider,
    async shutdown() {
      await provider.forceFlush();
      instrumentation.disable();
      await provider.shutdown();
    },
  };
}
