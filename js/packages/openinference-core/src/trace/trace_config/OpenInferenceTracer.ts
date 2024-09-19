import { Context, Span, SpanOptions, Tracer } from "@opentelemetry/api";
import { OpenInferenceActiveSpanCallback, TraceConfig } from "./types";
import { OpenInferenceSpan } from "./OpenInferenceSpan";

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

  if (arguments.length < 2) {
    return;
  } else if (arguments.length === 2) {
    fn = arg2 as F;
  } else if (arguments.length === 3) {
    opts = arg2 as SpanOptions | undefined;
    fn = arg3 as F;
  } else {
    opts = arg2 as SpanOptions | undefined;
    ctx = arg3 as Context | undefined;
    fn = arg4 as F;
  }

  return { opts, ctx, fn, name };
}

export class OpenInferenceTracer implements Tracer {
  private readonly tracer: Tracer;
  private readonly config: TraceConfig;
  constructor(tracer: Tracer, config: TraceConfig) {
    this.tracer = tracer;
    this.config = config;
  }
  startActiveSpan<F extends (span: OpenInferenceSpan) => unknown>(
    name: string,
    fn: F,
  ): ReturnType<F>;
  startActiveSpan<F extends (span: OpenInferenceSpan) => unknown>(
    name: string,
    options: SpanOptions,
    fn: F,
  ): ReturnType<F>;
  startActiveSpan<F extends (span: OpenInferenceSpan) => unknown>(
    name: string,
    options: SpanOptions,
    context: Context,
    fn: F,
  ): ReturnType<F>;
  startActiveSpan<F extends (span: OpenInferenceSpan) => ReturnType<F>>(
    name: string,
    arg2?: F | SpanOptions,
    arg3?: F | Context,
    arg4?: F,
  ): ReturnType<F> | undefined {
    const { opts, ctx, fn } = formatStartActiveSpanParams(arg2, arg3, arg4);
    return this.tracer.startActiveSpan(name, opts, ctx, (span: Span) =>
      fn(new OpenInferenceSpan({ span, config: this.config })),
    );
  }

  startSpan(
    name: string,
    options?: SpanOptions,
    context?: Context,
  ): OpenInferenceSpan {
    return new OpenInferenceSpan({
      span: this.tracer.startSpan(name, options, context),
      config: this.config,
    });
  }
}
