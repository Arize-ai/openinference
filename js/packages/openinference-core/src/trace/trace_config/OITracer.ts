import {
  context,
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
import { OpenInferenceSpan } from "./OpenInferenceSpan";
import { generateTraceConfig } from "./traceConfig";

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
  ctx = ctx ?? context.active();

  return { opts, ctx, fn };
}

export class OITracer implements Tracer {
  private readonly tracer: Tracer;
  private readonly config: TraceConfig;
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
    const formattedArgs = formatStartActiveSpanParams(arg2, arg3, arg4);
    if (formattedArgs == null) {
      return;
    }
    const { opts, ctx, fn } = formattedArgs;
    const { attributes } = opts ?? {};
    return this.tracer.startActiveSpan(
      name,
      { ...opts, attributes: undefined },
      ctx,
      (span: Span) => {
        const openInferenceSpan = new OpenInferenceSpan({
          span,
          config: this.config,
        });
        openInferenceSpan.setAttributes(attributes ?? {});
        return fn(openInferenceSpan);
      },
    );
  }

  startSpan(
    name: string,
    options?: SpanOptions,
    context?: Context,
  ): OpenInferenceSpan {
    const attributes = options?.attributes;
    const span = new OpenInferenceSpan({
      span: this.tracer.startSpan(
        name,
        { ...options, attributes: undefined },
        context,
      ),
      config: this.config,
    });
    span.setAttributes(attributes ?? {});
    return span;
  }
}
