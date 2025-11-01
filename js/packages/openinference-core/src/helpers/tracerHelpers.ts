import { trace, Tracer } from "@opentelemetry/api";

import { OITracer } from "../trace";

const DEFAULT_TRACER_NAME = "openinference-core";

/**
 * A function that ensures the tracer is wrapped in an OITracer if necessary.
 * @param tracer The tracer to wrap if necessary
 * @returns {OITracer} an OpenInferenceTracer that wraps the OTel Tracer
 */
export function wrapTracer(tracer: Tracer) {
  if (tracer instanceof OITracer) {
    return tracer;
  }
  return new OITracer({
    tracer,
  });
}

/**
 * A helper function to get a tracer for the decorators.
 */
export function getTracer(name: string = DEFAULT_TRACER_NAME) {
  return new OITracer({
    tracer: trace.getTracer(name),
  });
}
