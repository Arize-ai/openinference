import { AnyFn, SpanTraceOptions } from "./types";
import { withSpan } from "./withSpan";

/**
 * A decorator factory for tracing operations in an LLM application.
 *
 * This decorator wraps class methods to automatically create OpenTelemetry spans
 * for tracing purposes. It leverages the `withSpan` function internally to ensure
 * consistent tracing behavior across the library.
 *
 * The decorator uses an optimized caching mechanism to avoid rebinding methods on
 * every call, improving performance for frequently called methods.
 *
 * @experimental This API is experimental and may change in future versions
 *
 * @param options - Configuration options for the tracing behavior
 * @param options.tracer - Custom tracer instance to use (optional)
 * @param options.name - Custom span name (defaults to method name)
 * @param options.openTelemetrySpanKind - OpenTelemetry span kind (defaults to INTERNAL)
 * @param options.kind - OpenInference span kind (defaults to CHAIN)
 * @param options.processInput - Custom input processing function (optional)
 * @param options.processOutput - Custom output processing function (optional)
 *
 * @returns A decorator function that can be applied to class methods
 *
 * @example
 * ```typescript
 * class MyService {
 *   @observe({ name: "processData", kind: OpenInferenceSpanKind.LLM })
 *   async processData(input: string) {
 *     // Method implementation
 *     return `processed: ${input}`;
 *   }
 * }
 * ```
 */
export function observe<Fn extends AnyFn>(options: SpanTraceOptions = {}) {
  return function (originalMethod: Fn, ctx: ClassMethodDecoratorContext) {
    const methodName = String(ctx.name);

    // Create options with method name as fallback for span name
    const traceOptions: SpanTraceOptions = {
      ...options,
      name: options.name || methodName,
    };

    // Create a wrapper that preserves 'this' context for class methods
    const wrappedMethod = function (this: unknown, ...args: Parameters<Fn>) {
      // Bind the original method to the current 'this' context
      const boundMethod = originalMethod.bind(this) as Fn;

      // Use withSpan to wrap the bound method, ensuring consistent tracing behavior
      const tracedMethod = withSpan(boundMethod, traceOptions);

      return tracedMethod(...args);
    } as Fn;

    return wrappedMethod;
  };
}
