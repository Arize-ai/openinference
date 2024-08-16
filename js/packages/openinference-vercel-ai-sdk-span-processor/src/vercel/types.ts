import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { Mutable, ValueOf } from "../utils/typeUtils";

/**
 * A ReadWriteSpan is a mutable ReadableSpan.
 * Within the OpenTelemetry exporters, the spans are typed as readonly
 * Although the maintainers do mention that spans can be mutated in exporters @see https://github.com/open-telemetry/opentelemetry-specification/issues/1089#issuecomment-2045376590
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
