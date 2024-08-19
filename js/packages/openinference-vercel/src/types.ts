import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { Mutable, ValueOf } from "./typeUtils";

/**
 * A ReadWriteSpan is a mutable ReadableSpan.
 * Within the OpenTelemetry SpanProcessors, the spans are typed as readonly
 * However, mutation of spans in processors has been added to the OTEL spec and implementation is in progress @see https://github.com/open-telemetry/opentelemetry-specification/pull/4024
 * This is just the typescript way of enforcing that these finished spans are immutable but the spans coming into the exporter are actually the Span instantiations that have ended.
 * We use this type after copying the spans passed to the exporter to mutate the spans with the OpenInference attributes.
 */
export type ReadWriteSpan = Mutable<ReadableSpan>;

export type OpenInferenceSemanticConvention = ValueOf<
  typeof SemanticConventions
>;

export type OpenInferenceIOConvention = Extract<
  OpenInferenceSemanticConvention,
  | typeof SemanticConventions.OUTPUT_VALUE
  | typeof SemanticConventions.INPUT_VALUE
>;
