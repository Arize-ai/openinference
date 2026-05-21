import {
  context,
  trace,
  type Attributes,
  type AttributeValue,
  type Context,
  type Exception,
  type Link,
  type Span,
  type SpanContext,
  type SpanOptions,
  type SpanStatus,
  type TimeInput,
  type Tracer,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import type { ChatMiddleware } from "@tanstack/ai";
import {
  otelMiddleware,
  type OtelMiddlewareOptions,
  type OtelSpanInfo,
} from "@tanstack/ai/middlewares/otel";

import {
  getAttributesFromContext,
  generateTraceConfig,
  REDACTED_VALUE,
  type TraceConfig,
  type TraceConfigOptions,
} from "@arizeai/openinference-core";
import type { ConvertGenAISpanOptions, GenAISpanEvent } from "@arizeai/openinference-genai";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { INSTRUMENTATION_NAME } from "./constants";
import { convertTanStackAISpanToOpenInference } from "./genai";
import { VERSION } from "./version";

export type OpenInferenceTanStackAIMiddlewareOptions = Omit<
  OtelMiddlewareOptions,
  "tracer" | "attributeEnricher"
> & {
  tracer?: Tracer;
  traceConfig?: TraceConfigOptions;
  attributeEnricher?: OtelMiddlewareOptions["attributeEnricher"];
  spanKindResolver?: ConvertGenAISpanOptions["spanKindResolver"];
};

class OpenInferenceConvertingSpan implements Span {
  private readonly name: string;
  private readonly span: Span;
  private readonly traceConfig: TraceConfig;
  private readonly convertOptions: ConvertGenAISpanOptions;
  private readonly attributes: Attributes;
  private readonly events: GenAISpanEvent[] = [];
  private ended = false;

  constructor({
    name,
    span,
    traceConfig,
    convertOptions,
    initialAttributes,
    contextAttributes,
  }: {
    name: string;
    span: Span;
    traceConfig: ReturnType<typeof generateTraceConfig>;
    convertOptions: ConvertGenAISpanOptions;
    initialAttributes?: Attributes;
    contextAttributes?: Attributes;
  }) {
    this.name = name;
    this.span = span;
    this.attributes = { ...(initialAttributes ?? {}), ...(contextAttributes ?? {}) };
    this.traceConfig = traceConfig;
    this.convertOptions = convertOptions;
    if (contextAttributes != null) {
      this.span.setAttributes(withoutUndefinedValues(contextAttributes));
    }
  }

  setAttribute(key: string, value: AttributeValue): this {
    this.attributes[key] = value;
    this.span.setAttribute(key, value);
    return this;
  }

  setAttributes(attributes: Attributes): this {
    Object.assign(this.attributes, attributes);
    this.span.setAttributes(attributes);
    return this;
  }

  addEvent(
    name: string,
    attributesOrStartTime?: Attributes | TimeInput,
    startTime?: TimeInput,
  ): this {
    if (isAttributes(attributesOrStartTime)) {
      this.events.push({ name, attributes: attributesOrStartTime });
    }
    this.span.addEvent(name, attributesOrStartTime, startTime);
    return this;
  }

  addLink(link: Link): this {
    this.span.addLink(link);
    return this;
  }

  addLinks(links: Link[]): this {
    this.span.addLinks(links);
    return this;
  }

  end(endTime?: TimeInput): void {
    if (!this.ended) {
      this.ended = true;
      this.span.setAttributes(
        maskOpenInferenceAttributes(
          convertTanStackAISpanToOpenInference(
            {
              name: this.name,
              attributes: this.attributes,
              events: this.events,
            },
            this.convertOptions,
          ),
          this.traceConfig,
        ),
      );
    }
    this.span.end(endTime);
  }

  isRecording(): boolean {
    return this.span.isRecording();
  }

  recordException(exception: Exception, time?: TimeInput): void {
    this.span.recordException(exception, time);
  }

  spanContext(): SpanContext {
    return this.span.spanContext();
  }

  setStatus(status: SpanStatus): this {
    this.span.setStatus(status);
    return this;
  }

  updateName(name: string): this {
    this.span.updateName(name);
    return this;
  }
}

const maskOpenInferenceAttributes = (
  attributes: Attributes,
  traceConfig: TraceConfig,
): Attributes => {
  return Object.entries(attributes).reduce((masked, [key, value]) => {
    if (traceConfig.hideInputs && key.startsWith("input.")) return masked;
    if (traceConfig.hideOutputs && key.startsWith("output.")) return masked;
    if (traceConfig.hideInputMessages && key.startsWith(SemanticConventions.LLM_INPUT_MESSAGES)) {
      return masked;
    }
    if (traceConfig.hideOutputMessages && key.startsWith(SemanticConventions.LLM_OUTPUT_MESSAGES)) {
      return masked;
    }
    if (
      traceConfig.hideInputText &&
      key.startsWith(SemanticConventions.LLM_INPUT_MESSAGES) &&
      isMessageContentAttribute(key)
    ) {
      masked[key] = REDACTED_VALUE;
      return masked;
    }
    if (
      traceConfig.hideOutputText &&
      key.startsWith(SemanticConventions.LLM_OUTPUT_MESSAGES) &&
      isMessageContentAttribute(key)
    ) {
      masked[key] = REDACTED_VALUE;
      return masked;
    }
    masked[key] = value;
    return masked;
  }, {} as Attributes);
};

const isMessageContentAttribute = (key: string): boolean =>
  key.endsWith(SemanticConventions.MESSAGE_CONTENT_TEXT) ||
  key.endsWith(SemanticConventions.MESSAGE_CONTENT);

const isAttributes = (value: Attributes | TimeInput | undefined): value is Attributes => {
  return (
    typeof value === "object" && value != null && !Array.isArray(value) && !(value instanceof Date)
  );
};

const createOpenInferenceTracer = ({
  tracer,
  traceConfig,
  convertOptions,
}: {
  tracer: Tracer;
  traceConfig: TraceConfig;
  convertOptions: ConvertGenAISpanOptions;
}): Tracer => {
  const wrapSpan = ({
    name,
    span,
    options,
    spanContext,
  }: {
    name: string;
    span: Span;
    options?: SpanOptions;
    spanContext: Context;
  }): Span =>
    new OpenInferenceConvertingSpan({
      name,
      span,
      traceConfig,
      convertOptions,
      initialAttributes: options?.attributes,
      contextAttributes: getAttributesFromContext(spanContext),
    });

  const wrappedTracer: Tracer = {
    startSpan(name: string, options?: SpanOptions, ctx?: Context): Span {
      const spanContext = ctx ?? context.active();
      return wrapSpan({
        name,
        span: tracer.startSpan(name, options, ctx),
        options,
        spanContext,
      });
    },
    startActiveSpan(
      name: string,
      optionsOrCallback?: SpanOptions | ((span: Span) => unknown),
      contextOrCallback?: Context | ((span: Span) => unknown),
      callback?: (span: Span) => unknown,
    ): unknown {
      if (typeof optionsOrCallback === "function") {
        return tracer.startActiveSpan(name, (span) =>
          optionsOrCallback(
            wrapSpan({
              name,
              span,
              spanContext: context.active(),
            }),
          ),
        );
      }

      const options = optionsOrCallback;
      if (typeof contextOrCallback === "function") {
        return tracer.startActiveSpan(name, options ?? {}, (span) =>
          contextOrCallback(
            wrapSpan({
              name,
              span,
              options,
              spanContext: context.active(),
            }),
          ),
        );
      }

      if (callback != null) {
        const spanContext = contextOrCallback ?? context.active();
        return tracer.startActiveSpan(name, options ?? {}, spanContext, (span) =>
          callback(
            wrapSpan({
              name,
              span,
              options,
              spanContext,
            }),
          ),
        );
      }

      throw new TypeError(
        "startActiveSpan requires a callback function as one of its arguments",
      );
    },
  };
  return wrappedTracer;
};

const shouldCaptureContent = (options: OpenInferenceTanStackAIMiddlewareOptions): boolean => {
  if (options.captureContent != null) {
    return options.captureContent;
  }
  return !(options.traceConfig?.hideInputs === true && options.traceConfig?.hideOutputs === true);
};

const createRedactor = (options: OpenInferenceTanStackAIMiddlewareOptions) => {
  const userRedact = options.redact ?? ((text: string) => text);
  return (text: string): string => {
    if (
      options.traceConfig?.hideInputs === true ||
      options.traceConfig?.hideOutputs === true ||
      options.traceConfig?.hideInputText === true ||
      options.traceConfig?.hideOutputText === true
    ) {
      return REDACTED_VALUE;
    }
    return userRedact(text);
  };
};

const isSuppressed = () => isTracingSuppressed(context.active());

const withoutUndefinedValues = (attributes: Attributes): Record<string, AttributeValue> => {
  return Object.entries(attributes).reduce(
    (result, [key, value]) => {
      if (value != null) {
        result[key] = value;
      }
      return result;
    },
    {} as Record<string, AttributeValue>,
  );
};

export function openInferenceMiddleware(
  options: OpenInferenceTanStackAIMiddlewareOptions = {},
): ChatMiddleware {
  const traceConfig = generateTraceConfig(options.traceConfig);
  const tracer = createOpenInferenceTracer({
    tracer: options.tracer ?? trace.getTracer(INSTRUMENTATION_NAME, VERSION),
    traceConfig,
    convertOptions: {
      spanKindResolver: options.spanKindResolver,
    },
  });
  const attributeEnricher: OtelMiddlewareOptions["attributeEnricher"] = (info: OtelSpanInfo) =>
    withoutUndefinedValues({
      ...getAttributesFromContext(context.active()),
      ...(options.attributeEnricher?.(info) ?? {}),
    });
  const nativeMiddleware = otelMiddleware({
    ...options,
    tracer,
    captureContent: shouldCaptureContent(options),
    redact: createRedactor(options),
    attributeEnricher,
  });

  return {
    name: INSTRUMENTATION_NAME,
    onConfig(ctx, config) {
      if (isSuppressed()) return undefined;
      return nativeMiddleware.onConfig?.(ctx, config);
    },
    onStart(ctx) {
      if (isSuppressed()) return undefined;
      return nativeMiddleware.onStart?.(ctx);
    },
    onIteration(ctx, info) {
      if (isSuppressed()) return undefined;
      return nativeMiddleware.onIteration?.(ctx, info);
    },
    onChunk(ctx, chunk) {
      if (isSuppressed()) return undefined;
      return nativeMiddleware.onChunk?.(ctx, chunk);
    },
    onBeforeToolCall(ctx, hookCtx) {
      if (isSuppressed()) return undefined;
      return nativeMiddleware.onBeforeToolCall?.(ctx, hookCtx);
    },
    onAfterToolCall(ctx, info) {
      if (isSuppressed()) return undefined;
      return nativeMiddleware.onAfterToolCall?.(ctx, info);
    },
    onToolPhaseComplete(ctx, info) {
      if (isSuppressed()) return undefined;
      return nativeMiddleware.onToolPhaseComplete?.(ctx, info);
    },
    onUsage(ctx, usage) {
      if (isSuppressed()) return undefined;
      return nativeMiddleware.onUsage?.(ctx, usage);
    },
    onFinish(ctx, info) {
      if (isSuppressed()) return undefined;
      return nativeMiddleware.onFinish?.(ctx, info);
    },
    onAbort(ctx, info) {
      if (isSuppressed()) return undefined;
      return nativeMiddleware.onAbort?.(ctx, info);
    },
    onError(ctx, info) {
      if (isSuppressed()) return undefined;
      return nativeMiddleware.onError?.(ctx, info);
    },
  };
}

export { convertTanStackAISpanToOpenInference, tanStackAISpanKindResolver } from "./genai";
export type { ConvertTanStackAISpanOptions } from "./genai";
