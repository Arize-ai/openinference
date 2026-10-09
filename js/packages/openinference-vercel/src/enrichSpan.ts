import type { OpenTelemetry } from "@ai-sdk/otel";

import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

type EnrichSpan = NonNullable<
  NonNullable<ConstructorParameters<typeof OpenTelemetry>[0]>["enrichSpan"]
>;

/**
 * Provides OpenInference attributes when `@ai-sdk/otel` creates a span.
 *
 * Pass this callback as `OpenTelemetry`'s `enrichSpan` option alongside an
 * OpenInference span processor. It uses the SDK's span type and operation ID
 * when the emitted GenAI attributes do not distinguish the span's role.
 * For decision calls, it marks the outer operation as `CHAIN` and the model
 * call as `DECISION`. Other spans are left to the processor's normal mapping.
 *
 * @example
 * registerTelemetry(
 *   new OpenTelemetry({ enrichSpan: enrichSpanWithOpenInference }),
 * );
 */
export const enrichSpanWithOpenInference: EnrichSpan = ({ spanType, operationId }) => {
  if (spanType === "operation" && operationId === "ai.decide") {
    return { [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN };
  }
  if (spanType === "experimental_decision" && operationId === "ai.decide.doDecide") {
    return { [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.DECISION };
  }
  return undefined;
};
