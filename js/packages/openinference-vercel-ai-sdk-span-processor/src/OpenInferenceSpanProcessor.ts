import { ReadableSpan, SpanProcessor } from "@opentelemetry/sdk-trace-base";
import {
  getOIModelNameAttribute,
  getOISpanKindFromAttributes,
  hasAIAttributes,
} from "./utils";
import { Span } from "@opentelemetry/api";
import { OTLPTraceExporter as ProtoExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { OTLPTraceExporter as HttpExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { OTLPTraceExporter as GrpcExporter } from "@opentelemetry/exporter-trace-otlp-grpc";

import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import { OTLPExporterNodeConfigBase } from "@opentelemetry/otlp-exporter-base";

export class OpenInferenceSpanProcessor implements SpanProcessor {
  async forceFlush() {
    // No-op
  }

  onStart(_: Span): void {
    // No-op
  }

  async shutdown(): Promise<void> {
    // No-op
  }
  onEnd(span: ReadableSpan): void {
    const initialAttributes = span.attributes;
    if (!hasAIAttributes(initialAttributes)) {
      return;
    }

    const spanKind = getOISpanKindFromAttributes(initialAttributes);

    if (spanKind == null) {
      return;
    }
    // @ts-expect-error - This is a read-only span and thus has no setter for attributes
    // Manually patch attributes here
    span.attributes = {
      ...span.attributes,
      ...getOIModelNameAttribute(initialAttributes),
      [SemanticConventions.OPENINFERENCE_SPAN_KIND]: spanKind,
    };
  }
}

// make all keys in record mutable
type Mutable<T> = {
  -readonly [P in keyof T]: T[P];
};

const makeSpansMutable = (
  spans: ReadableSpan[],
): Mutable<
  ReadableSpan & {
    setAttributes: Span["setAttributes"];
    setAttribute: Span["setAttribute"];
  }
>[] => {
  return spans.map((span) => {
    // spanContext is a getter
    const mutableSpan = { ...span, spanContext: span.spanContext };
    // @ts-expect-error - This is a read-only span and thus has no setter for attributes
    console.log("test--hhhhhhhhhhh------", span.setAttribute);
    console.log("test--hey hey", mutableSpan.spanContext);
    return mutableSpan;
  });
};

export class OpenInferenceProtoTraceExporter extends ProtoExporter {
  constructor(config?: OTLPExporterNodeConfigBase) {
    super(config);
  }
  async export(...args: Parameters<ProtoExporter["export"]>) {
    const spans = args[0];
    const mutableSpans = makeSpansMutable(spans);
    mutableSpans.forEach((span) => {
      const initialAttributes = span.attributes;
      const spanKind = getOISpanKindFromAttributes(initialAttributes);

      if (spanKind == null) {
        return;
      }
      span.attributes = {
        ...span.attributes,
        ...getOIModelNameAttribute(initialAttributes),
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
