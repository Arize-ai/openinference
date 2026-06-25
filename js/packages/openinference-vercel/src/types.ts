import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

import type { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import type { Mutable, ValueOf } from "./typeUtils";

/**
 * A ReadWriteSpan is a mutable {@link ReadableSpan}.
 * Within the OpenTelemetry SpanProcessors, the spans are typed as readonly
 * However, mutation of spans in processors has been added to the OTEL spec and implementation is in progress @see https://github.com/open-telemetry/opentelemetry-specification/pull/4024
 * We use this type to directly mutate the attributes of spans in the OpenInferenceSpanProcessor
 */
export type ReadWriteSpan = Mutable<ReadableSpan>;

export type OpenInferenceSemanticConventionKey = ValueOf<typeof SemanticConventions>;

export type OpenInferenceIOConventionKey = Extract<
  OpenInferenceSemanticConventionKey,
  typeof SemanticConventions.OUTPUT_VALUE | typeof SemanticConventions.INPUT_VALUE
>;

/**
 * A filter to apply to a {@link ReadableSpan}.
 */
export type SpanFilter = (span: ReadableSpan) => boolean;

/**
 * A hook invoked on each span at `onEnd`, before OpenInference attribute
 * conversion runs. Use it to enrich or remap framework-specific attributes
 * (e.g. mapping `eve.*` attributes onto OpenInference conventions) so the
 * conversion and span filter see the result. The span is mutated in place.
 */
export type PreProcessSpan = (span: ReadableSpan) => void;
