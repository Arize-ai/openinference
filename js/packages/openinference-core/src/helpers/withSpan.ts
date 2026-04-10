import { SpanKind, SpanStatusCode } from "@opentelemetry/api";

import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import type { OITracer } from "../trace";
import { isPromise } from "../utils/typeUtils";
import { defaultProcessInput, defaultProcessOutput } from "./attributeHelpers";
import { getTracer, wrapTracer } from "./tracerHelpers";
import type { AnyFn, InputToAttributesFn, OutputToAttributesFn, SpanTraceOptions } from "./types";

const { OPENINFERENCE_SPAN_KIND } = SemanticConventions;

/**
 * Wraps a function with openinference tracing capabilities, creating spans for execution monitoring.
 *
 * This function provides comprehensive tracing for both synchronous and asynchronous functions,
 * automatically handling span lifecycle, input/output processing, error tracking, and promise
 * resolution.
 *
 * @experimental This API is experimental and may change in future versions
 *
 * @template Fn - The function type being wrapped, preserving original signature
 * @param fn - The function to wrap with tracing capabilities
 * @param options - Configuration options for tracing behavior
 * @param options.tracer - Custom OpenTelemetry tracer instance (defaults to global tracer)
 * @param options.name - Custom span name (defaults to function name)
 * @param options.openTelemetrySpanKind - OpenTelemetry span kind (defaults to INTERNAL)
 * @param options.kind - OpenInference span kind for semantic categorization (defaults to CHAIN)
 * @param options.processInput - Custom function to process input arguments into attributes
 * @param options.processOutput - Custom function to process output values into attributes
 * @param options.attributes - Base attributes to be added to every span created
 *
 * @returns A wrapped function with identical signature that creates spans during execution
 *
 * @example
 * ```typescript
 * // Basic function wrapping
 * const add = (a: number, b: number) => a + b;
 * const tracedAdd = withSpan(add);
 * const result = tracedAdd(2, 3); // Creates a span named "add"
 *
 * // Async function with custom options
 * const fetchData = async (url: string) => {
 *   const response = await fetch(url);
 *   return response.json();
 * };
 * const tracedFetch = withSpan(fetchData, {
 *   name: "api-request",
 *   kind: OpenInferenceSpanKind.LLM
 * });
 *
 * // Custom input/output processing with base attributes
 * const processUser = (user: User) => ({ ...user, processed: true });
 * const tracedProcess = withSpan(processUser, {
 *   attributes: {
 *     'service.name': 'user-processor',
 *     'service.version': '1.0.0'
 *   },
 *   processInput: (user) => ({ "user.id": user.id }),
 *   processOutput: (result) => ({ "result.processed": result.processed })
 * });
 * ```
 */
export function withSpan<Fn extends AnyFn = AnyFn>(fn: Fn, options?: SpanTraceOptions<Fn>): Fn {
  const {
    tracer: _tracer,
    name: optionsName,
    processInput: _processInput,
    processOutput: _processOutput,
    openTelemetrySpanKind = SpanKind.INTERNAL,
    kind = OpenInferenceSpanKind.CHAIN,
    attributes: baseAttributes,
  } = options || {};
  const configuredTracer: OITracer | undefined = _tracer ? wrapTracer(_tracer) : undefined;
  const processInput: InputToAttributesFn = _processInput ?? defaultProcessInput;
  const processOutput: OutputToAttributesFn = _processOutput ?? defaultProcessOutput;
  const spanName = optionsName || fn.name;
  const getErrorMessage = (error: unknown) => {
    if (typeof error === "object" && error !== null && "message" in error) {
      return String(error.message);
    }
    return String(error);
  };
  // TODO: infer the name from the target
  const wrappedFn: Fn = function (this: ThisParameterType<Fn>, ...args: Parameters<Fn>) {
    const tracer = configuredTracer ?? getTracer();
    return tracer.startActiveSpan(
      spanName,
      {
        kind: openTelemetrySpanKind,
        attributes: {
          ...baseAttributes,
          [OPENINFERENCE_SPAN_KIND]: kind,
          ...processInput(...args),
        },
      },
      (span) => {
        const recordError = (error: unknown) => {
          span.recordException(error as Error);
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: getErrorMessage(error),
          });
        };

        try {
          const result = fn.apply(this, args) as ReturnType<Fn>;
          if (isPromise(result)) {
            // Execute the promise and return the promise chain
            return result
              .then((value: Awaited<ReturnType<Fn>>) => {
                span.setAttributes({
                  ...processOutput(value),
                });
                span.setStatus({
                  code: SpanStatusCode.OK,
                });
                return value;
              })
              .catch((error: unknown) => {
                recordError(error);
                throw error;
              })
              .finally(() => span.end());
          }

          // It is a normal function
          span.setAttributes({
            ...processOutput(result),
          });
          span.setStatus({
            code: SpanStatusCode.OK,
          });
          span.end();
          return result;
        } catch (error) {
          recordError(error);
          span.end();
          throw error;
        }
      },
    );
  } as Fn;
  return wrappedFn;
}
