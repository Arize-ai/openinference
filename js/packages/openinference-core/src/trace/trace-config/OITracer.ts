import {
  context as apiContext,
  Context,
  Span,
  SpanOptions,
  Tracer,
} from "@opentelemetry/api";
import {
  OpenInferenceActiveSpanCallback,
  TraceConfig,
  TraceConfigOptions,
} from "./types";
import { OISpan } from "./OISpan";
import { generateTraceConfig } from "./traceConfig";
import { getAttributesFromContext } from "../contextAttributes";

/**
 * Formats the params for the startActiveSpan method
 * The method has multiple overloads, so we need to format the arguments
 * Taken from @see https://github.com/open-telemetry/opentelemetry-js/blob/main/packages/opentelemetry-sdk-trace-base/src/Tracer.ts#L220C3-L235C6
 *
 */
function formatStartActiveSpanParams<F extends OpenInferenceActiveSpanCallback>(
  arg2?: SpanOptions | F,
  arg3?: Context | F,
  arg4?: F,
) {
  let opts: SpanOptions | undefined;
  let ctx: Context | undefined;
  let fn: F;

  if (typeof arg2 === "function") {
    fn = arg2;
  } else if (typeof arg3 === "function") {
    opts = arg2;
    fn = arg3;
  } else {
    opts = arg2;
    ctx = arg3;
    fn = arg4 as F;
  }

  opts = opts ?? {};
  ctx = ctx ?? apiContext.active();

  return { opts, ctx, fn };
}

/**
 * A wrapper around the OpenTelemetry {@link Tracer} interface that masks sensitive information based on the passed in {@link TraceConfig}.
 */
export class OITracer implements Tracer {
  private readonly tracer: Tracer;
  private readonly config: TraceConfig;
  /**
   *
   * @param tracer The OpenTelemetry {@link Tracer} to wrap
   * @param traceConfig The {@link TraceConfigOptions} to set to control the behavior of the tracer
   */
  constructor({
    tracer,
    traceConfig,
  }: {
    tracer: Tracer;
    traceConfig?: TraceConfigOptions;
  }) {
    this.tracer = tracer;
    this.config = generateTraceConfig(traceConfig);
  }
  startActiveSpan<F extends (span: OISpan) => unknown>(
    name: string,
    fn: F,
  ): ReturnType<F>;
  startActiveSpan<F extends (span: OISpan) => unknown>(
    name: string,
    options: SpanOptions,
    fn: F,
  ): ReturnType<F>;
  startActiveSpan<F extends (span: OISpan) => unknown>(
    name: string,
    options: SpanOptions,
    context: Context,
    fn: F,
  ): ReturnType<F>;
  startActiveSpan<F extends (span: OISpan) => ReturnType<F>>(
    name: string,
    arg2?: F | SpanOptions,
    arg3?: F | Context,
    arg4?: F,
  ): ReturnType<F> | undefined {
    const formattedArgs = formatStartActiveSpanParams(arg2, arg3, arg4);
    if (formattedArgs == null) {
      return;
    }
    const { opts, ctx, fn } = formattedArgs;
    const { attributes } = opts ?? {};
    const contextAttributes = getAttributesFromContext(ctx);
    const mergedAttributes = { ...contextAttributes, ...attributes };
    return this.tracer.startActiveSpan(
      name,
      { ...opts, attributes: undefined },
      ctx,
      (span: Span) => {
        const openInferenceSpan = new OISpan({
          span,
          config: this.config,
        });
        openInferenceSpan.setAttributes(mergedAttributes);
        return fn(openInferenceSpan);
      },
    );
  }

  startSpan(name: string, options?: SpanOptions, context?: Context): OISpan {
    const attributes = options?.attributes;
    const ctx = context ?? apiContext.active();
    const contextAttributes = getAttributesFromContext(ctx);
    const mergedAttributes = { ...contextAttributes, ...attributes };
    const span = new OISpan({
      span: this.tracer.startSpan(
        name,
        { ...options, attributes: undefined },
        ctx,
      ),
      config: this.config,
    });
    span.setAttributes(mergedAttributes);
    return span;
  }
}
