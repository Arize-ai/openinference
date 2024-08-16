import {
  copySpans,
  getOISpanKindFromAttributes,
  getOpenInferenceAttributes,
} from "./utils";
import { Span } from "@opentelemetry/api";
import { OTLPTraceExporter as ProtoExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { OTLPTraceExporter as HttpExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { OTLPTraceExporter as GrpcExporter } from "@opentelemetry/exporter-trace-otlp-grpc";

import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import { OTLPExporterNodeConfigBase } from "@opentelemetry/otlp-exporter-base";

export class OpenInferenceProtoTraceExporter extends ProtoExporter {
  constructor(config?: OTLPExporterNodeConfigBase) {
    super(config);
  }
  async export(...args: Parameters<ProtoExporter["export"]>) {
    const spans = args[0];
    const mutableSpans = copySpans(spans);
    mutableSpans.forEach((span) => {
      const initialAttributes = span.attributes;
      const spanKind = getOISpanKindFromAttributes(initialAttributes);

      if (spanKind == null) {
        return;
      }
      span.attributes = {
        ...span.attributes,
        ...getOpenInferenceAttributes(initialAttributes),
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: spanKind,
      };
    });

    return super.export(mutableSpans, args[1]);
  }
}
// type OtlpExporterType =
//   | typeof ProtoExporter
//   | typeof HttpExporter
//   | typeof GrpcExporter;

// type Constructor<T extends OtlpExporterType> = new (
//   ...args: ConstructorParameters<T>[]
// ) => T;

// function CustomOtlpExporterMixin<TBase extends OtlpExporterType>(Base: TBase) {
//   return class extends Base {
//     constructor(...args: ConstructorParameters<TBase>[]) {
//       super(...args);
//     }

//     async export(...args: Parameters<TBase["prototype"]["export"]>) {
//       const spans = args[0];
//       // Perform custom processing on spans
//       const mutableSpans: Span[] = [...spans];

//       const processedSpans = mutableSpans.map((span) => {
//         const initialAttributes = span.;
//         if (!hasAIAttributes(initialAttributes)) {
//           return span;
//         }

//         const spanKind = getOISpanKindFromAttributes(initialAttributes);

//         if (spanKind == null) {
//           return span;
//         }
//         // @ts-expect-error - This is a read-only span and thus has no setter for attributes
//         // Manually patch attributes here
//         span.attributes = {
//           ...span.attributes,
//           ...getOIModelNameAttribute(initialAttributes),
//           [SemanticConventions.OPENINFERENCE_SPAN_KIND]: spanKind,
//         };
//         return span;
//       });

//       // Call the parent class's export method
//       return super.export(processedSpans, ...args.slice(1));
//     }

//     // You can add more custom methods here if needed
//   };
// }

// // Create custom classes for each exporter type
// class CustomProtoExporter extends CustomOtlpExporterMixin(ProtoExporter) {}
// class CustomHttpExporter extends CustomOtlpExporterMixin(HttpExporter) {}
// class CustomGrpcExporter extends CustomOtlpExporterMixin(GrpcExporter) {}

// // Export these classes for users of your library
// export {
//   CustomProtoExporter,
//   CustomHttpExporter,
//   CustomGrpcExporter,
//   CustomOtlpExporterMixin,
// };
