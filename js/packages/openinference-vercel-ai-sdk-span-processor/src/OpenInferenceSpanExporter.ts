import { addOpenInferenceAttributesToSpans } from "./vercel/utils";
import { OTLPTraceExporter as ProtoExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { OTLPTraceExporter as HttpExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { OTLPTraceExporter as GrpcExporter } from "@opentelemetry/exporter-trace-otlp-grpc";

import { OTLPExporterNodeConfigBase } from "@opentelemetry/otlp-exporter-base";
import { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { ExportResult } from "@opentelemetry/core";

export class OpenInferenceProtoTraceExporter extends ProtoExporter {
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

export class OpenInferenceGrpcTraceExporter extends GrpcExporter {
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
