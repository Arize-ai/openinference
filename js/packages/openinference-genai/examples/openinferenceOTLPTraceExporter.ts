import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import type { ExportResult } from "@opentelemetry/core";

import { convertGenAISpanAttributesToOpenInferenceSpanAttributes } from "../src/index.js";
import { Mutable } from "../src/types.js";

export class OpenInferenceOTLPTraceExporter extends OTLPTraceExporter {
  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ) {
    const processedSpans = spans.map((span) => {
      const processedAttributes =
        convertGenAISpanAttributesToOpenInferenceSpanAttributes(
          span.attributes,
        );
      // optionally you can replace the entire attributes object with the
      // processed attributes if you want _only_ the OpenInference attributes
      (span as Mutable<ReadableSpan>).attributes = {
        ...span.attributes,
        ...processedAttributes,
      };
      return span;
    });

    super.export(processedSpans, resultCallback);
  }
}
