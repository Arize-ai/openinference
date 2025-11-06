import {
  OITracer,
  safelyJSONStringify,
  TraceConfigOptions,
} from "@arizeai/openinference-core";
import {
  LLMProvider,
  LLMSystem,
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import {
  Attributes,
  context,
  diag,
  Span,
  SpanKind,
  SpanStatusCode,
  trace,
  Tracer,
  TracerProvider,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
  safeExecuteInTheMiddle,
} from "@opentelemetry/instrumentation";

// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore - No version file until build
import { VERSION } from "./version";

import type Anthropic from "@anthropic-ai/sdk";
import type { Stream } from "@anthropic-ai/sdk/streaming";

const MODULE_NAME = "@anthropic-ai/sdk";

const INSTRUMENTATION_NAME = "@arizeai/openinference-instrumentation-anthropic";

/**
 * Flag to check if the anthropic module has been patched
 * Note: This is a fallback in case the module is made immutable (e.x. Deno, webpack, etc.)
 */
let _isOpenInferencePatched = false;

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
  const execContext = suppressTracing
    ? trace.setSpan(context.active(), span)
    : activeContext;
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
export class AnthropicInstrumentation extends InstrumentationBase<
  typeof Anthropic
> {
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
    super(
      INSTRUMENTATION_NAME,
      VERSION,
      Object.assign({}, instrumentationConfig),
    );
    this.tracerProvider = tracerProvider;
    this.traceConfig = traceConfig;
    this.oiTracer = new OITracer({
      tracer:
        this.tracerProvider?.getTracer(INSTRUMENTATION_NAME, VERSION) ??
        this.tracer,
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
      return this.tracerProvider.getTracer(
        this.instrumentationName,
        this.instrumentationVersion,
      );
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

    if (module?.openInferencePatched || _isOpenInferencePatched) {
      return module;
    }

    // Handle ES module default export structure
    const anthropicModule =
      (module as typeof Anthropic & { default?: typeof Anthropic }).default ||
      module;

    if (!anthropicModule?.Messages?.prototype?.create) {
      diag.warn(
        `Cannot find Messages.prototype.create in ${MODULE_NAME}@${moduleVersion}`,
      );
      return module;
    }

    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const instrumentation: AnthropicInstrumentation = this;

    // Patch messages.create
    type MessagesCreateType = typeof anthropicModule.Messages.prototype.create;

    this._wrap(
      anthropicModule.Messages.prototype,
      "create",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (original: MessagesCreateType): any => {
        return function patchedCreate(
          this: unknown,
          ...args: Parameters<MessagesCreateType>
        ) {
          const body = args[0] as Anthropic.Messages.MessageCreateParams;
          const { messages: _messages, ...invocationParameters } = body;
          const span = instrumentation.oiTracer.startSpan(
            `Anthropic Messages`,
            {
              kind: SpanKind.INTERNAL,
              attributes: {
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.LLM,
                [SemanticConventions.LLM_MODEL_NAME]: body.model,
                [SemanticConventions.INPUT_VALUE]: JSON.stringify(body),
                [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
                [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
                  JSON.stringify(invocationParameters),
                [SemanticConventions.LLM_SYSTEM]: LLMSystem.ANTHROPIC,
                [SemanticConventions.LLM_PROVIDER]: LLMProvider.ANTHROPIC,
                ...getAnthropicInputMessagesAttributes(body),
                ...getAnthropicToolsJSONSchema(body),
              },
            },
          );
          const execContext = getExecContext(span);
          const execPromise = safeExecuteInTheMiddle(
            () => {
              return context.with(trace.setSpan(execContext, span), () => {
                return original.apply(this, args);
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
          );

          const wrappedPromiseThen = (
            result:
              | Anthropic.Messages.Message
              | Stream<Anthropic.Messages.RawMessageStreamEvent>,
          ) => {
            if (isAnthropicMessageResponse(result)) {
              // Record the results
              span.setAttributes({
                [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(result),
                [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
                // Override the model from the value sent by the server
                [SemanticConventions.LLM_MODEL_NAME]: result.model,
                ...getAnthropicOutputMessagesAttributes(result),
                ...getAnthropicUsageAttributes(result),
              });
              span.setStatus({ code: SpanStatusCode.OK });
              span.end();
            } else if (isAnthropicStream(result)) {
              // This is a streaming response
              // handle the chunks and add them to the span
              // First split the stream via tee
              const [leftStream, rightStream] = result.tee();
              consumeAnthropicStreamChunks(rightStream, span);
              result = leftStream;
            }

            return result;
          };

          const wrappedPromise = execPromise
            .then(wrappedPromiseThen)
            .catch((error: Error) => {
              span.recordException(error);
              span.setStatus({
                code: SpanStatusCode.ERROR,
                message: error.message,
              });
              span.end();
              throw error;
            });
          return context.bind(execContext, wrappedPromise);
        };
      },
    );

    _isOpenInferencePatched = true;
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
    this._unwrap(moduleExports.Messages.prototype, "create");

    _isOpenInferencePatched = false;
    try {
      // This can fail if the module is made immutable via the runtime or bundler
      moduleExports.openInferencePatched = false;
    } catch (e) {
      diag.warn(`Failed to unset ${MODULE_NAME} patched flag on the module`, e);
    }
  }
}

/**
 * type-guard that checks if the response is an Anthropic message response
 */
function isAnthropicMessageResponse(
  response: unknown,
): response is Anthropic.Messages.Message {
  return (
    response != null &&
    typeof response === "object" &&
    "content" in response &&
    "role" in response
  );
}

/**
 * type-guard that checks if the response is an Anthropic stream
 */
function isAnthropicStream(
  response: unknown,
): response is Stream<Anthropic.Messages.RawMessageStreamEvent> {
  return response != null && typeof response === "object" && "tee" in response;
}

/**
 * Converts the body of an Anthropic messages request to LLM input messages
 */
function getAnthropicInputMessagesAttributes(
  body: Anthropic.Messages.MessageCreateParams,
): Attributes {
  return body.messages.reduce((acc, message, index) => {
    const messageAttributes = getAnthropicInputMessageAttributes(message);
    const indexPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.`;
    // Flatten the attributes on the index prefix
    for (const [key, value] of Object.entries(messageAttributes)) {
      acc[`${indexPrefix}${key}`] = value;
    }
    return acc;
  }, {} as Attributes);
}

/**
 * Converts each tool definition into a json schema
 */
function getAnthropicToolsJSONSchema(
  body: Anthropic.Messages.MessageCreateParams,
): Attributes {
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

function getAnthropicInputMessageAttributes(
  message: Anthropic.Messages.MessageParam,
): Attributes {
  const role = message.role;
  const attributes: Attributes = {
    [SemanticConventions.MESSAGE_ROLE]: role,
  };

  // Add the content based on type
  if (typeof message.content === "string") {
    attributes[SemanticConventions.MESSAGE_CONTENT] = message.content;
  } else if (Array.isArray(message.content)) {
    message.content.forEach((part, index) => {
      const contentsIndexPrefix = `${SemanticConventions.MESSAGE_CONTENTS}.${index}.`;
      if (part.type === "text") {
        attributes[
          `${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`
        ] = "text";
        attributes[
          `${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`
        ] = part.text;
      } else if (part.type === "image") {
        attributes[
          `${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`
        ] = "image";
        if (part.source.type === "base64") {
          // For base64 images, we don't store the actual data but indicate it's base64
          attributes[
            `${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_IMAGE}.type`
          ] = "base64";
          attributes[
            `${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_IMAGE}.media_type`
          ] = part.source.media_type;
        }
      } else if (part.type === "tool_use") {
        const toolCallIndexPrefix = `${SemanticConventions.MESSAGE_TOOL_CALLS}.${index}.`;
        attributes[
          `${toolCallIndexPrefix}${SemanticConventions.TOOL_CALL_ID}`
        ] = part.id;
        attributes[
          `${toolCallIndexPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
        ] = part.name;
        attributes[
          `${toolCallIndexPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
        ] = JSON.stringify(part.input);
      } else if (part.type === "tool_result") {
        attributes[`${SemanticConventions.MESSAGE_TOOL_CALL_ID}`] =
          part.tool_use_id;
        if (typeof part.content === "string") {
          attributes[SemanticConventions.MESSAGE_CONTENT] = part.content;
        } else if (Array.isArray(part.content)) {
          // Handle complex tool result content
          attributes[SemanticConventions.MESSAGE_CONTENT] = JSON.stringify(
            part.content,
          );
        }
      }
    });
  }

  return attributes;
}

/**
 * Converts the Anthropic message result to LLM output attributes
 */
function getAnthropicOutputMessagesAttributes(
  message: Anthropic.Messages.Message,
): Attributes {
  const attributes: Attributes = {};
  const indexPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.`;

  attributes[`${indexPrefix}${SemanticConventions.MESSAGE_ROLE}`] =
    message.role;

  // Handle content array
  message.content.forEach((content, contentIndex) => {
    const contentPrefix = `${indexPrefix}${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.`;

    if (content.type === "text") {
      attributes[
        `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`
      ] = "text";
      attributes[
        `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`
      ] = content.text;
    } else if (content.type === "tool_use") {
      const toolCallPrefix = `${indexPrefix}${SemanticConventions.MESSAGE_TOOL_CALLS}.${contentIndex}.`;
      attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_ID}`] =
        content.id;
      attributes[
        `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
      ] = content.name;
      attributes[
        `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
      ] = JSON.stringify(content.input);
    }
  });

  return attributes;
}

/**
 * Get usage attributes from Anthropic response
 */
function getAnthropicUsageAttributes(
  message: Anthropic.Messages.Message,
): Attributes {
  if (message.usage) {
    return {
      [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]:
        message.usage.output_tokens,
      [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: message.usage.input_tokens,
      [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]:
        message.usage.input_tokens + message.usage.output_tokens,
    };
  }
  return {};
}

/**
 * Consumes the stream chunks and adds them to the span
 */
async function consumeAnthropicStreamChunks(
  stream: Stream<Anthropic.Messages.RawMessageStreamEvent>,
  span: Span,
) {
  let streamResponse = "";
  const toolCallAttributes: Attributes = {};

  for await (const chunk of stream) {
    if (
      chunk.type === "content_block_delta" &&
      chunk.delta.type === "text_delta"
    ) {
      streamResponse += chunk.delta.text;
    } else if (
      chunk.type === "content_block_start" &&
      chunk.content_block.type === "tool_use"
    ) {
      const toolCall = chunk.content_block;
      const toolCallIndex = chunk.index;
      const toolCallPrefix = `${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.`;

      toolCallAttributes[
        `${toolCallPrefix}${SemanticConventions.TOOL_CALL_ID}`
      ] = toolCall.id;
      toolCallAttributes[
        `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
      ] = toolCall.name;
    } else if (
      chunk.type === "content_block_delta" &&
      chunk.delta.type === "input_json_delta"
    ) {
      const toolCallIndex = chunk.index;
      const toolCallPrefix = `${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.`;
      const existingArgs =
        toolCallAttributes[
          `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
        ] || "";
      toolCallAttributes[
        `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
      ] = existingArgs + chunk.delta.partial_json;
    }
  }

  const messageIndexPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.`;

  // Append the attributes to the span as a message
  const attributes: Attributes = {
    [SemanticConventions.OUTPUT_VALUE]: streamResponse,
    [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
    [`${messageIndexPrefix}${SemanticConventions.MESSAGE_CONTENT}`]:
      streamResponse,
    [`${messageIndexPrefix}${SemanticConventions.MESSAGE_ROLE}`]: "assistant",
  };

  // Add the tool call attributes
  for (const [key, value] of Object.entries(toolCallAttributes)) {
    attributes[`${messageIndexPrefix}${key}`] = value;
  }

  span.setAttributes(attributes);
  span.setStatus({ code: SpanStatusCode.OK });
  span.end();
}
