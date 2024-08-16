import { addOpenInferenceAttributesToSpans } from "./vercel/utils";
import { OTLPTraceExporter as HttpExporter } from "@opentelemetry/exporter-trace-otlp-http";

import { OTLPExporterNodeConfigBase } from "@opentelemetry/otlp-exporter-base";
import { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { ExportResult } from "@opentelemetry/core";

export class OpenInferenceHttpTraceExporter extends HttpExporter {
  constructor(config?: OTLPExporterNodeConfigBase) {
    super(config);
  }

  async export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ) {
    const openInferenceSpans = addOpenInferenceAttributesToSpans(spans);
    return super.export(openInferenceSpans, resultCallback);
  }
}
