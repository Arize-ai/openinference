import type Anthropic from "@anthropic-ai/sdk";
import type { APIPromise } from "@anthropic-ai/sdk";
import type { Stream } from "@anthropic-ai/sdk/streaming";
import type { Attributes, Span, Tracer, TracerProvider } from "@opentelemetry/api";
import { context, diag, SpanKind, SpanStatusCode, trace } from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import type {
  InstrumentationConfig,
  InstrumentationModuleDefinition,
} from "@opentelemetry/instrumentation";
import {
  InstrumentationBase,
  InstrumentationNodeModuleDefinition,
  safeExecuteInTheMiddle,
} from "@opentelemetry/instrumentation";

import type { TraceConfigOptions } from "@arizeai/openinference-core";
import { OITracer, safelyJSONStringify } from "@arizeai/openinference-core";
import {
  LLMProvider,
  LLMSystem,
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

// oxlint-disable-next-line typescript/prefer-ts-expect-error
// @ts-ignore - No version file until build
import { VERSION } from "./version";

const MODULE_NAME = "@anthropic-ai/sdk";

const INSTRUMENTATION_NAME = "@arizeai/openinference-instrumentation-anthropic";

type MessageCreateParams =
  | Parameters<typeof Anthropic.Messages.prototype.create>[0]
  | Parameters<typeof Anthropic.Beta.Messages.prototype.create>[0];
type MessageParam = Anthropic.Messages.MessageParam | Anthropic.Beta.Messages.BetaMessageParam;
type Message = Anthropic.Messages.Message | Anthropic.Beta.Messages.BetaMessage;
type RawMessageStreamEvent =
  | Anthropic.Messages.RawMessageStreamEvent
  | Anthropic.Beta.Messages.BetaRawMessageStreamEvent;
type MessageUsage =
  | Anthropic.Messages.Usage
  | Anthropic.Messages.MessageDeltaUsage
  | Anthropic.Beta.Messages.BetaUsage
  | Anthropic.Beta.Messages.BetaMessageDeltaUsage
  | Anthropic.Beta.Messages.BetaFallbackMessageIterationUsage;
type AnthropicModuleWithOptionalBeta = Omit<typeof Anthropic, "Beta"> & {
  Beta?: { Messages?: typeof Anthropic.Beta.Messages };
};

/**
 * Resolves the Anthropic namespace and its optional `Beta.Messages` from a
 * module export, unwrapping the ES-module default. `patch` and `unpatch` must
 * agree on the object they (un)wrap, so they share this one resolution.
 */
function resolveAnthropicModule(moduleExports: typeof Anthropic) {
  const anthropicModule =
    (moduleExports as typeof Anthropic & { default?: typeof Anthropic }).default || moduleExports;
  return {
    anthropicModule,
    betaMessages: (anthropicModule as AnthropicModuleWithOptionalBeta).Beta?.Messages,
  };
}

/**
 * Flag to check if the anthropic module has been patched
 * Note: This is a fallback in case the module is made immutable (e.x. Deno, webpack, etc.)
 */
let _isOpenInferencePatched = false;

/**
 * The Anthropic classes that have already been patched, tracked by identity.
 * The SDK ships separate CJS and ESM builds with separate class objects, so a
 * module-global boolean cannot guard them independently: whichever build was
 * patched first would block the other one forever (#3557). A WeakSet is
 * scoped to the object, and needs no write to the module, so it also keeps
 * the double-patch guard working when the module is immutable (e.g. Deno,
 * webpack) and the `openInferencePatched` property cannot be set.
 */
const _patchedModules = new WeakSet<object>();

/**
 * function to check if instrumentation is enabled / disabled
 */
export function isPatched() {
  return _isOpenInferencePatched;
}

/**
 * Resolves the execution context for the current span
 * If tracing is suppressed, the span is dropped and the current context is returned
 * @param span
 */
function getExecContext(span: Span) {
  const activeContext = context.active();
  const suppressTracing = isTracingSuppressed(activeContext);
  const execContext = suppressTracing ? trace.setSpan(context.active(), span) : activeContext;
  // Drop the span from the context
  if (suppressTracing) {
    trace.deleteSpan(activeContext);
  }
  return execContext;
}

/**
 * An auto instrumentation class for Anthropic that creates {@link https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md|OpenInference} Compliant spans for the Anthropic API
 * @param instrumentationConfig The config for the instrumentation @see {@link InstrumentationConfig}
 * @param traceConfig The OpenInference trace configuration. Can be used to mask or redact sensitive information on spans. @see {@link TraceConfigOptions}
 */
export class AnthropicInstrumentation extends InstrumentationBase<typeof Anthropic> {
  private oiTracer: OITracer;
  private tracerProvider?: TracerProvider;
  private traceConfig?: TraceConfigOptions;

  constructor({
    instrumentationConfig,
    traceConfig,
    tracerProvider,
  }: {
    /**
     * The config for the instrumentation
     * @see {@link InstrumentationConfig}
     */
    instrumentationConfig?: InstrumentationConfig;
    /**
     * The OpenInference trace configuration. Can be used to mask or redact sensitive information on spans.
     * @see {@link TraceConfigOptions}
     */
    traceConfig?: TraceConfigOptions;
    /**
     * An optional custom trace provider to be used for tracing. If not provided, a tracer will be created using the global tracer provider.
     * This is useful if you want to use a non-global tracer provider.
     *
     * @see {@link TracerProvider}
     */
    tracerProvider?: TracerProvider;
  } = {}) {
    super(INSTRUMENTATION_NAME, VERSION, Object.assign({}, instrumentationConfig));
    this.tracerProvider = tracerProvider;
    this.traceConfig = traceConfig;
    this.oiTracer = new OITracer({
      tracer: this.tracerProvider?.getTracer(INSTRUMENTATION_NAME, VERSION) ?? this.tracer,
      traceConfig,
    });
  }

  protected init(): InstrumentationModuleDefinition<typeof Anthropic> {
    const module = new InstrumentationNodeModuleDefinition<typeof Anthropic>(
      "@anthropic-ai/sdk",
      ["*"], // Try accepting any version
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return module;
  }

  /**
   * Manually instruments the Anthropic module. This is needed when the module is not loaded via require (commonjs)
   * @param {Anthropic} module
   */
  manuallyInstrument(module: typeof Anthropic) {
    diag.debug(`Manually instrumenting ${MODULE_NAME}`);
    this.patch(module);
  }

  get tracer(): Tracer {
    if (this.tracerProvider) {
      return this.tracerProvider.getTracer(this.instrumentationName, this.instrumentationVersion);
    }
    return super.tracer;
  }

  setTracerProvider(tracerProvider: TracerProvider): void {
    super.setTracerProvider(tracerProvider);
    this.tracerProvider = tracerProvider;
    this.oiTracer = new OITracer({
      tracer: this.tracer,
      traceConfig: this.traceConfig,
    });
  }

  /**
   * Patches the Anthropic module
   */
  private patch(
    module: typeof Anthropic & { openInferencePatched?: boolean },
    moduleVersion?: string,
  ) {
    diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);

    if (module?.openInferencePatched) {
      return module;
    }

    const { anthropicModule, betaMessages } = resolveAnthropicModule(module);

    if (anthropicModule && _patchedModules.has(anthropicModule)) {
      return module;
    }

    if (!anthropicModule?.Messages?.prototype?.create) {
      diag.warn(`Cannot find Messages.prototype.create in ${MODULE_NAME}@${moduleVersion}`);
      return module;
    }

    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const instrumentation: AnthropicInstrumentation = this;

    // Patch stable and beta messages.create using the same span lifecycle.
    type MessagesCreateType = typeof anthropicModule.Messages.prototype.create;
    type BetaMessagesCreateType = typeof anthropicModule.Beta.Messages.prototype.create;
    type AnyMessagesCreateType = MessagesCreateType | BetaMessagesCreateType;

    const patchCreate =
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      <CreateType extends AnyMessagesCreateType>(original: CreateType): any => {
        return function patchedCreate(this: unknown, ...args: Parameters<CreateType>) {
          const body = args[0];
          const { messages: _messages, ...invocationParameters } = body;
          const span = instrumentation.oiTracer.startSpan(`Anthropic Messages`, {
            kind: SpanKind.INTERNAL,
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
              [SemanticConventions.LLM_MODEL_NAME]: body.model,
              [SemanticConventions.LLM_REQUEST_MODEL_NAME]: body.model,
              [SemanticConventions.INPUT_VALUE]: JSON.stringify(body),
              [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
              [SemanticConventions.LLM_INVOCATION_PARAMETERS]: JSON.stringify(invocationParameters),
              [SemanticConventions.LLM_SYSTEM]: LLMSystem.ANTHROPIC,
              [SemanticConventions.LLM_PROVIDER]: LLMProvider.ANTHROPIC,
              ...getAnthropicInputMessagesAttributes(body),
              ...getAnthropicToolsJSONSchema(body),
            },
          });
          const execContext = getExecContext(span);
          const execPromise = safeExecuteInTheMiddle(
            () => {
              return context.with(trace.setSpan(execContext, span), () => {
                // oxlint-disable-next-line typescript/no-unsafe-type-assertion
                return Reflect.apply(original, this, args) as ReturnType<CreateType>;
              });
            },
            (error: Error | undefined) => {
              // Push the error to the span
              if (error) {
                span.recordException(error);
                span.setStatus({
                  code: SpanStatusCode.ERROR,
                  message: error.message,
                });
                span.end();
              }
            },
          ) as APIPromise<Message | Stream<RawMessageStreamEvent>>;

          // The span can be ended by the parse path, the asResponse() override, or
          // an error, so guard against ending it more than once.
          let spanEnded = false;
          const endSpan = () => {
            if (spanEnded) {
              return;
            }
            spanEnded = true;
            span.end();
          };

          const recordError = (error: Error) => {
            if (spanEnded) {
              return;
            }
            span.recordException(error);
            span.setStatus({
              code: SpanStatusCode.ERROR,
              message: error.message,
            });
            endSpan();
          };

          const wrappedPromiseThen = (result: Message | Stream<RawMessageStreamEvent>) => {
            if (isAnthropicMessageResponse(result)) {
              // Record the results
              span.setAttributes({
                [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(result),
                [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
                // Override the model from the value sent by the server
                [SemanticConventions.LLM_MODEL_NAME]: result.model,
                [SemanticConventions.LLM_RESPONSE_MODEL_NAME]: result.model,
                ...getAnthropicFinishReasonAttributes(result.stop_reason),
                ...getAnthropicOutputMessagesAttributes(result),
                ...getAnthropicUsageAttributes(result.usage),
              });
              span.setStatus({ code: SpanStatusCode.OK });
              endSpan();
            } else if (isAnthropicStream(result)) {
              // This is a streaming response
              // handle the chunks and add them to the span
              // First split the stream via tee
              const [leftStream, rightStream] = result.tee();
              void consumeAnthropicStreamChunks(rightStream, span);
              result = leftStream;
            }

            return result;
          };

          // Use _thenUnwrap so the result stays an APIPromise and keeps
          // withResponse()/asResponse(). Plain .then() would drop them and break
          // client.messages.stream().
          if (hasThenUnwrap(execPromise)) {
            const wrappedPromise = execPromise._thenUnwrap(wrappedPromiseThen);
            const rawResponse = wrappedPromise.asResponse.bind(wrappedPromise);

            // Record request failures without triggering parse (which would consume
            // the body). Covers the await/withResponse and asResponse paths.
            rawResponse().catch(recordError);

            // Wrap asResponse() itself so the span is finalized only when the caller
            // actually chooses the raw-response path. Those callers bypass parsing,
            // so no parsed output attributes are available.
            wrappedPromise.asResponse = async () => {
              const response = await rawResponse();
              span.setStatus({ code: SpanStatusCode.OK });
              endSpan();
              return response;
            };

            // withResponse() calls this.asResponse() internally; reimplement it
            // against the raw response so the override above doesn't end the span
            // before wrappedPromiseThen records the output.
            wrappedPromise.withResponse = async () => {
              const [data, response] = await Promise.all([
                wrappedPromise.then((value) => value),
                rawResponse(),
              ]);
              return {
                data,
                response,
                request_id: response.headers.get("request-id"),
                workspace_id: response.headers.get("anthropic-workspace-id"),
              };
            };

            return context.bind(execContext, wrappedPromise);
          }

          const wrappedPromise = execPromise.then(wrappedPromiseThen).catch((error: Error) => {
            recordError(error);
            throw error;
          });
          return context.bind(execContext, wrappedPromise);
        };
      };

    this._wrap(anthropicModule.Messages.prototype, "create", patchCreate);
    if (betaMessages?.prototype?.create) {
      this._wrap(betaMessages.prototype, "create", patchCreate);
    } else {
      // Beta instrumentation is optional: the stable Messages patch still applies.
      diag.debug(`Cannot find Beta.Messages.prototype.create in ${MODULE_NAME}@${moduleVersion}`);
    }

    _isOpenInferencePatched = true;
    _patchedModules.add(anthropicModule);
    try {
      // This can fail if the module is made immutable via the runtime or bundler
      module.openInferencePatched = true;
    } catch (e) {
      diag.debug(`Failed to set ${MODULE_NAME} patched flag on the module`, e);
    }

    return module;
  }

  /**
   * Un-patches the Anthropic module's messages API
   */
  private unpatch(
    moduleExports: typeof Anthropic & { openInferencePatched?: boolean },
    moduleVersion?: string,
  ) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);
    const { anthropicModule, betaMessages } = resolveAnthropicModule(moduleExports);
    this._unwrap(anthropicModule.Messages.prototype, "create");
    if (betaMessages?.prototype?.create) {
      this._unwrap(betaMessages.prototype, "create");
    }

    _isOpenInferencePatched = false;
    // Keyed the same way patch() keys it, so a re-patch is possible after.
    _patchedModules.delete(
      (moduleExports as typeof Anthropic & { default?: typeof Anthropic }).default || moduleExports,
    );
    try {
      // This can fail if the module is made immutable via the runtime or bundler
      moduleExports.openInferencePatched = false;
    } catch (e) {
      diag.warn(`Failed to unset ${MODULE_NAME} patched flag on the module`, e);
    }
  }
}

/**
 * True when create() returned an APIPromise we can transform with _thenUnwrap.
 */
function hasThenUnwrap<T>(promise: PromiseLike<T>): promise is APIPromise<T> {
  return "_thenUnwrap" in promise && typeof promise._thenUnwrap === "function";
}

/**
 * type-guard that checks if the response is an Anthropic message response
 */
function isAnthropicMessageResponse(response: unknown): response is Message {
  return (
    response != null && typeof response === "object" && "content" in response && "role" in response
  );
}

/**
 * type-guard that checks if the response is an Anthropic stream
 */
function isAnthropicStream(response: unknown): response is Stream<RawMessageStreamEvent> {
  return response != null && typeof response === "object" && "tee" in response;
}

/**
 * Records the reason the model stopped generating tokens. `"refusal"` is the
 * signal that a safety classifier declined the request.
 *
 * @see https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback
 */
function getAnthropicFinishReasonAttributes(stopReason: string | null | undefined): Attributes {
  if (stopReason == null) {
    return {};
  }
  return { [SemanticConventions.LLM_FINISH_REASON]: stopReason };
}

/**
 * Summarizes a server-side fallback handoff so the boundary is visible on the
 * span. Without this the block would occupy an index in the flattened
 * `message_contents` list while contributing no attributes, leaving a hole in
 * the list and dropping which model declined and why.
 *
 * @see https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback#server-side-fallback
 */
function getAnthropicFallbackContentAttributes(
  prefix: string,
  block: Anthropic.Beta.Messages.BetaFallbackBlock | Anthropic.Beta.Messages.BetaFallbackBlockParam,
): Attributes {
  // The response block types `trigger` as a refusal trigger, but the param
  // variant echoed back on a later turn declares it `unknown` — the server
  // accepts and ignores any object there — so it has to be narrowed.
  const trigger: unknown = block.trigger;
  const category =
    typeof trigger === "object" && trigger !== null && "category" in trigger
      ? trigger.category
      : undefined;
  const reason = typeof category === "string" ? ` (refusal: ${category})` : "";
  return {
    [`${prefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`]: "fallback",
    [`${prefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`]: `${block.from.model} -> ${block.to.model}${reason}`,
  };
}

/**
 * Converts the body of an Anthropic messages request to LLM input messages
 */
function getAnthropicInputMessagesAttributes(body: MessageCreateParams): Attributes {
  return body.messages.reduce<Attributes>((acc, message, index) => {
    const messageAttributes = getAnthropicInputMessageAttributes(message);
    const indexPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.`;
    // Flatten the attributes on the index prefix
    for (const [key, value] of Object.entries(messageAttributes)) {
      acc[`${indexPrefix}${key}`] = value;
    }
    return acc;
  }, {});
}

/**
 * Converts each tool definition into a json schema
 */
function getAnthropicToolsJSONSchema(body: MessageCreateParams): Attributes {
  if (!body.tools) {
    // If tools is undefined, return an empty object
    return {};
  }
  return body.tools.reduce((acc: Attributes, tool, index) => {
    const toolJsonSchema = safelyJSONStringify(tool);
    const key = `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`;
    if (toolJsonSchema) {
      acc[key] = toolJsonSchema;
    }
    return acc;
  }, {});
}

function getAnthropicInputMessageAttributes(message: MessageParam): Attributes {
  const role = message.role;
  const attributes: Attributes = {
    [SemanticConventions.MESSAGE_ROLE]: role,
  };

  // Add the content based on type
  if (typeof message.content === "string") {
    attributes[SemanticConventions.MESSAGE_CONTENT] = message.content;
  } else if (Array.isArray(message.content)) {
    let toolIndex = 0;
    message.content.forEach((part, index) => {
      const contentsIndexPrefix = `${SemanticConventions.MESSAGE_CONTENTS}.${index}.`;
      if (part.type === "text") {
        attributes[`${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "text";
        attributes[`${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = part.text;
      } else if (part.type === "image") {
        attributes[`${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "image";
        if (part.source.type === "base64") {
          // For base64 images, we don't store the actual data but indicate it's base64
          attributes[`${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_IMAGE}.type`] =
            "base64";
          attributes[
            `${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_IMAGE}.media_type`
          ] = part.source.media_type;
        }
      } else if (part.type === "tool_use") {
        const toolCallIndexPrefix = `${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolIndex}.`;
        attributes[`${toolCallIndexPrefix}${SemanticConventions.TOOL_CALL_ID}`] = part.id;
        attributes[`${toolCallIndexPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`] =
          part.name;
        attributes[
          `${toolCallIndexPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
        ] = JSON.stringify(part.input);
        attributes[`${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] =
          "tool_use";
        attributes[`${contentsIndexPrefix}${SemanticConventions.TOOL_CALL_ID}`] = part.id;
        attributes[`${contentsIndexPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`] =
          part.name;
        attributes[
          `${contentsIndexPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
        ] = JSON.stringify(part.input);
        toolIndex++;
      } else if (part.type === "tool_result") {
        attributes[`${SemanticConventions.MESSAGE_TOOL_CALL_ID}`] = part.tool_use_id;
        if (typeof part.content === "string") {
          attributes[SemanticConventions.MESSAGE_CONTENT] = part.content;
        } else if (Array.isArray(part.content)) {
          // Handle complex tool result content
          attributes[SemanticConventions.MESSAGE_CONTENT] = JSON.stringify(part.content);
        }
      } else if (part.type === "thinking") {
        attributes[`${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] =
          "reasoning";
        attributes[`${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] =
          part.thinking;
        if (part.signature) {
          attributes[`${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_SIGNATURE}`] =
            part.signature;
        }
      } else if (part.type === "redacted_thinking") {
        attributes[`${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] =
          "reasoning";
        attributes[`${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_DATA}`] = part.data;
      } else if (part.type === "fallback") {
        // A prior turn's fallback boundary echoed back to the API
        Object.assign(attributes, getAnthropicFallbackContentAttributes(contentsIndexPrefix, part));
      }
    });
  }

  return attributes;
}

/**
 * Converts the Anthropic message result to LLM output attributes
 */
function getAnthropicOutputMessagesAttributes(message: Message): Attributes {
  const attributes: Attributes = {};
  const indexPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.`;

  attributes[`${indexPrefix}${SemanticConventions.MESSAGE_ROLE}`] = message.role;
  let toolIndex = 0;
  // Handle content array
  message.content.forEach((content, contentIndex) => {
    const contentPrefix = `${indexPrefix}${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.`;

    if (content.type === "text") {
      attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "text";
      attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = content.text;
    } else if (content.type === "tool_use") {
      const toolCallPrefix = `${indexPrefix}${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolIndex}.`;
      attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_ID}`] = content.id;
      attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`] = content.name;
      attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`] =
        JSON.stringify(content.input);

      attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "tool_use";
      attributes[`${contentPrefix}${SemanticConventions.TOOL_CALL_ID}`] = content.id;
      attributes[`${contentPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`] = content.name;
      attributes[`${contentPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`] =
        JSON.stringify(content.input);
      toolIndex++;
    } else if (content.type === "thinking") {
      attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "reasoning";
      attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = content.thinking;
      if (content.signature) {
        attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_SIGNATURE}`] =
          content.signature;
      }
    } else if (content.type === "redacted_thinking") {
      attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "reasoning";
      attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_DATA}`] = content.data;
    } else if (content.type === "fallback") {
      Object.assign(attributes, getAnthropicFallbackContentAttributes(contentPrefix, content));
    }
  });

  return attributes;
}

/**
 * Get usage attributes from Anthropic response
 */
function getAnthropicUsageAttributes(usage: MessageUsage): Attributes {
  const attributes: Attributes = {};
  if (usage.input_tokens != null) {
    attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] = usage.input_tokens;
  }
  if (usage.output_tokens != null) {
    attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] = usage.output_tokens;
  }
  if (usage.input_tokens != null && usage.output_tokens != null) {
    attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL] =
      usage.input_tokens + usage.output_tokens;
  }
  return attributes;
}

/**
 * Mutable state accumulated while consuming an Anthropic message stream.
 *
 * Usage is reported per attempt: message_start describes the first attempt,
 * which on a server-side fallback stream is the one that declined. Each source
 * is captured on its own so the precedence between them is stated once, where
 * they are merged in {@link getAnthropicStreamAttributes}.
 */
interface AnthropicStreamState {
  streamResponse: string;
  toolCallAttributes: Attributes;
  contentAttributes: Attributes;
  startUsageAttributes: Attributes;
  deltaUsageAttributes: Attributes;
  servingUsageAttributes: Attributes;
  responseModel?: string;
  finishReason?: string;
  toolIndex: number;
}

/**
 * Applies a `message_delta` event, capturing usage, finish reason and any
 * server-side fallback hop.
 */
function applyAnthropicMessageDelta({
  chunk,
  state,
}: {
  chunk: Extract<RawMessageStreamEvent, { type: "message_delta" }>;
  state: AnthropicStreamState;
}) {
  state.deltaUsageAttributes = getAnthropicUsageAttributes(chunk.usage);
  if (chunk.delta.stop_reason != null) {
    state.finishReason = chunk.delta.stop_reason;
  }
  if (!("iterations" in chunk.usage) || chunk.usage.iterations == null) {
    return;
  }
  for (const iteration of chunk.usage.iterations) {
    if (iteration.type === "fallback_message") {
      state.responseModel = iteration.model;
      state.servingUsageAttributes = getAnthropicUsageAttributes(iteration);
    }
  }
}

/**
 * Applies a `content_block_start` event, recording the content block's type and
 * any tool call it starts.
 */
function applyAnthropicContentBlockStart({
  chunk,
  state,
}: {
  chunk: Extract<RawMessageStreamEvent, { type: "content_block_start" }>;
  state: AnthropicStreamState;
}) {
  const contentBlock = chunk.content_block;
  const contentPrefix = `${SemanticConventions.MESSAGE_CONTENTS}.${chunk.index}.`;
  const { contentAttributes, toolCallAttributes } = state;

  if (contentBlock.type === "tool_use") {
    state.toolIndex++;
    const toolCallPrefix = `${SemanticConventions.MESSAGE_TOOL_CALLS}.${state.toolIndex}.`;
    toolCallAttributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_ID}`] = contentBlock.id;
    toolCallAttributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`] =
      contentBlock.name;
    contentAttributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "tool_use";
    contentAttributes[`${contentPrefix}${SemanticConventions.TOOL_CALL_ID}`] = contentBlock.id;
    contentAttributes[`${contentPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`] =
      contentBlock.name;
  } else if (contentBlock.type === "text") {
    contentAttributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "text";
    contentAttributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] =
      contentBlock.text;
  } else if (contentBlock.type === "thinking") {
    contentAttributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "reasoning";
    contentAttributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] =
      contentBlock.thinking;
  } else if (contentBlock.type === "redacted_thinking") {
    contentAttributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "reasoning";
    contentAttributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_DATA}`] =
      contentBlock.data;
  } else if (contentBlock.type === "fallback") {
    state.responseModel = contentBlock.to.model;
    Object.assign(
      contentAttributes,
      getAnthropicFallbackContentAttributes(contentPrefix, contentBlock),
    );
  }
}

/**
 * Applies a `content_block_delta` event, accumulating streamed text, reasoning
 * and tool call arguments.
 */
function applyAnthropicContentBlockDelta({
  chunk,
  state,
}: {
  chunk: Extract<RawMessageStreamEvent, { type: "content_block_delta" }>;
  state: AnthropicStreamState;
}) {
  const contentPrefix = `${SemanticConventions.MESSAGE_CONTENTS}.${chunk.index}.`;
  const { contentAttributes, toolCallAttributes } = state;
  const textKey = `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`;

  if (chunk.delta.type === "text_delta") {
    state.streamResponse += chunk.delta.text;
    contentAttributes[textKey] = (contentAttributes[textKey] || "") + chunk.delta.text;
  } else if (chunk.delta.type === "thinking_delta") {
    contentAttributes[textKey] = (contentAttributes[textKey] || "") + chunk.delta.thinking;
  } else if (chunk.delta.type === "signature_delta") {
    contentAttributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_SIGNATURE}`] =
      chunk.delta.signature;
  } else if (chunk.delta.type === "input_json_delta") {
    const toolCallPrefix = `${SemanticConventions.MESSAGE_TOOL_CALLS}.${state.toolIndex}.`;
    const argumentsKey = `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`;
    const updatedArgs = (toolCallAttributes[argumentsKey] || "") + chunk.delta.partial_json;
    toolCallAttributes[argumentsKey] = updatedArgs;
    contentAttributes[`${contentPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`] =
      updatedArgs;
  }
}

/**
 * Builds the span attributes for a fully consumed Anthropic message stream.
 */
function getAnthropicStreamAttributes(state: AnthropicStreamState): Attributes {
  const messageIndexPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.`;

  const attributes: Attributes = {
    [SemanticConventions.OUTPUT_VALUE]: state.streamResponse,
    [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
    [`${messageIndexPrefix}${SemanticConventions.MESSAGE_ROLE}`]: "assistant",
  };

  if (state.responseModel != null) {
    // Override the model from the value sent by the server
    attributes[SemanticConventions.LLM_MODEL_NAME] = state.responseModel;
    attributes[SemanticConventions.LLM_RESPONSE_MODEL_NAME] = state.responseModel;
  }

  if (state.finishReason != null) {
    attributes[SemanticConventions.LLM_FINISH_REASON] = state.finishReason;
  }

  // Add the content block attributes
  for (const [key, value] of Object.entries(state.contentAttributes)) {
    attributes[`${messageIndexPrefix}${key}`] = value;
  }

  // Add the tool call attributes
  for (const [key, value] of Object.entries(state.toolCallAttributes)) {
    attributes[`${messageIndexPrefix}${key}`] = value;
  }

  // Later sources win: on a server-side fallback stream the serving hop's
  // counts displace the declined attempt's counts from message_start, and the
  // final message_delta wins over both, so prompt and completion describe the
  // same model.
  const usageAttributes: Attributes = {
    ...state.startUsageAttributes,
    ...state.servingUsageAttributes,
    ...state.deltaUsageAttributes,
  };

  // Recompute the total in case prompt and completion counts came from
  // different sources.
  const promptTokens = usageAttributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT];
  const completionTokens = usageAttributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION];
  if (typeof promptTokens === "number" && typeof completionTokens === "number") {
    usageAttributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL] = promptTokens + completionTokens;
  }
  Object.assign(attributes, usageAttributes);

  return attributes;
}

/**
 * Consumes the stream chunks and adds them to the span
 */
async function consumeAnthropicStreamChunks(stream: Stream<RawMessageStreamEvent>, span: Span) {
  const state: AnthropicStreamState = {
    streamResponse: "",
    toolCallAttributes: {},
    contentAttributes: {},
    startUsageAttributes: {},
    deltaUsageAttributes: {},
    servingUsageAttributes: {},
    toolIndex: -1,
  };

  for await (const chunk of stream) {
    if (chunk.type === "message_start") {
      state.responseModel = chunk.message.model;
      state.startUsageAttributes = getAnthropicUsageAttributes(chunk.message.usage);
    } else if (chunk.type === "message_delta") {
      applyAnthropicMessageDelta({ chunk, state });
    } else if (chunk.type === "content_block_start") {
      applyAnthropicContentBlockStart({ chunk, state });
    } else if (chunk.type === "content_block_delta") {
      applyAnthropicContentBlockDelta({ chunk, state });
    }
  }

  span.setAttributes(getAnthropicStreamAttributes(state));
  span.setStatus({ code: SpanStatusCode.OK });
  span.end();
}
