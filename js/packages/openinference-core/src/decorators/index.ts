import {
  SemanticConventions,
  OpenInferenceSpanKind,
  MimeType,
  OUTPUT_MIME_TYPE,
} from "@arizeai/openinference-semantic-conventions";
import { type Tracer, trace } from "@opentelemetry/api";
import { OITracer } from "trace";
import { safelyJSONStringify } from "utils";

const DEFAULT_TRACER_NAME = "openinference-core";

const { OPENINFERENCE_SPAN_KIND, INPUT_VALUE, OUTPUT_VALUE, INPUT_MIME_TYPE } =
  SemanticConventions;
export interface TraceDecoratorOptions {
  /**
   * An optional name for the span. If not provided, the name if the decorated function will be used
   */
  name?: string;
  /**
   * An optional tracer to be used for tracing the chain operation. If not provided, the global tracer will be used.
   */
  tracer?: Tracer;
  /**
   * A callback function that will be used to set the input attribute of the span from the arguments of the function.
   */
  processInput?: (args: unknown) => string;
  /**
   * A callback function that will be used to set the output attribute of the span from the result of the function.
   */
  processOutput?: (result: unknown) => string;
}

/**
 * A decorator factory for tracing chain operations in an LLM application.
 */
export function chain<Target>(options: TraceDecoratorOptions) {
  const {
    tracer: _tracer,
    name = "chain",
    processInput: _processInput,
    processOutput: _processOutput,
  } = options;
  const tracer: OITracer = _tracer ? wrapTracer(_tracer) : getTracer();
  const processInput = _processInput ?? safelyJSONStringify;
  const processOutput = _processOutput ?? safelyJSONStringify;
  // TODO: infer the name from the target
  return function (
    target: Target,
    propertyKey: string,
    descriptor: TypedPropertyDescriptor<(...args: unknown[]) => unknown>,
  ) {
    const originalFn = descriptor.value!;
    // override the value to  wrap the original function in a span
    descriptor.value = function (...args: unknown[]) {
      return tracer.startActiveSpan(
        name,
        {
          attributes: {
            [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
            [INPUT_VALUE]: processInput(args) ?? undefined,
            // TODO: infer the mime type from the arguments
            [INPUT_MIME_TYPE]: MimeType.JSON,
          },
        },
        (span) => {
          const result = originalFn.apply(this, args);
          span.setAttributes({
            [OUTPUT_VALUE]: processOutput(result) ?? undefined,
            // TODO: infer the mime type from the result
            [OUTPUT_MIME_TYPE]: MimeType.JSON,
          });
          // TODO: set the status of the span based on the result
          span.end();
          return result;
        },
      );
    };

    return descriptor;
  };
}
/**
 * A function that ensures the tracer is wrapped in an OITracer if necessary.
 * @param tracer The tracer to wrap if necessary
 * @returns
 */
function wrapTracer(tracer: Tracer) {
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
function getTracer() {
  return new OITracer({
    tracer: trace.getTracer(DEFAULT_TRACER_NAME),
  });
}
