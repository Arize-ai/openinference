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

const MODULE_NAME = "@google/genai";

const INSTRUMENTATION_NAME =
  "@arizeai/openinference-instrumentation-google-genai";

/**
 * Flag to check if the google-genai module has been patched
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
 */
function getExecContext(span: Span) {
  const activeContext = context.active();
  const suppressTracing = isTracingSuppressed(activeContext);
  const execContext = suppressTracing
    ? trace.setSpan(context.active(), span)
    : activeContext;
  if (suppressTracing) {
    trace.deleteSpan(activeContext);
  }
  return execContext;
}

/**
 * An auto instrumentation class for Google Gen AI SDK that creates OpenInference compliant spans
 */
export class GoogleGenAIInstrumentation extends InstrumentationBase {
  private oiTracer: OITracer;
  private tracerProvider?: TracerProvider;
  private traceConfig?: TraceConfigOptions;

  constructor({
    instrumentationConfig,
    traceConfig,
    tracerProvider,
  }: {
    instrumentationConfig?: InstrumentationConfig;
    traceConfig?: TraceConfigOptions;
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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  protected init(): InstrumentationModuleDefinition<any> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const module = new InstrumentationNodeModuleDefinition<any>(
      "@google/genai",
      ["*"], // Support all versions
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return module;
  }

  /**
   * Manually instruments the Google Gen AI module
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  manuallyInstrument(module: any) {
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
   * Patches a GoogleGenAI instance by instrumenting its models and chats
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  public instrumentInstance(instance: any): void {
    // Patch methods on the models property
    if (instance.models) {
      this.patchModelsInstance(instance.models, this);
    }

    // Patch methods on the chats property
    if (instance.chats && instance.chats.create) {
      this.patchChatsInstance(instance.chats, this);
    }

    // Patch methods on the batches property
    if (instance.batches) {
      this.patchBatchesInstance(instance.batches, this);
    }
  }

  /**
   * Patches the Google Gen AI module - Note: Due to ESM limitations,
   * this only works for CommonJS. For ESM, use instrumentInstance()
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private patch(module: any, moduleVersion?: string) {
    diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);

    if (module?.openInferencePatched || _isOpenInferencePatched) {
      return module;
    }

    _isOpenInferencePatched = true;
    try {
      module.openInferencePatched = true;
    } catch (e) {
      diag.debug(`Failed to set ${MODULE_NAME} patched flag on the module`, e);
    }

    diag.warn(
      `${MODULE_NAME} auto-instrumentation has limited support with ESM. Use instrumentInstance() or the provided helper functions for full functionality.`,
    );

    return module;
  }

  /**
   * Patches methods on a Models instance
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private patchModelsInstance(
    models: any,
    instrumentation: GoogleGenAIInstrumentation,
  ) {
    // Wrap generateContent
    if (models.generateContent) {
      const originalGenerateContent = models.generateContent.bind(models);
      models.generateContent = async function patchedGenerateContent(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ...genArgs: any[]
      ) {
        const params = genArgs[0];
        const modelName = params?.model || "unknown";

        const span = instrumentation.oiTracer.startSpan(
          `Google GenAI Generate Content`,
          {
            kind: SpanKind.INTERNAL,
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.LLM,
              [SemanticConventions.LLM_MODEL_NAME]: modelName,
              [SemanticConventions.INPUT_VALUE]:
                safelyJSONStringify(params) || "",
              [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
              [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
                safelyJSONStringify(params?.config) || "{}",
              [SemanticConventions.LLM_SYSTEM]: LLMSystem.VERTEXAI,
              [SemanticConventions.LLM_PROVIDER]: LLMProvider.GOOGLE,
              ...getInputMessagesAttributes(params),
              ...getToolsJSONSchema(params),
            },
          },
        );

        const execContext = getExecContext(span);
        const execPromise = safeExecuteInTheMiddle(
          () => {
            return context.with(trace.setSpan(execContext, span), () => {
              return originalGenerateContent(...genArgs);
            });
          },
          (error: Error | undefined) => {
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

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return execPromise
          .then((result: any) => {
            span.setAttributes({
              [SemanticConventions.OUTPUT_VALUE]:
                safelyJSONStringify(result) || "",
              [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
              ...getOutputMessagesAttributes(result),
              ...getUsageAttributes(result),
            });
            span.setStatus({ code: SpanStatusCode.OK });
            span.end();
            return result;
          })
          .catch((error: Error) => {
            // Span is already ended in safeExecuteInTheMiddle error callback
            // Just re-throw the error
            throw error;
          });
      };
    }

    // Wrap generateContentStream
    if (models.generateContentStream) {
      const originalGenerateContentStream =
        models.generateContentStream.bind(models);
      models.generateContentStream =
        async function patchedGenerateContentStream(
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          ...genArgs: any[]
        ) {
          const params = genArgs[0];
          const modelName = params?.model || "unknown";

          const span = instrumentation.oiTracer.startSpan(
            `Google GenAI Generate Content Stream`,
            {
              kind: SpanKind.INTERNAL,
              attributes: {
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.LLM,
                [SemanticConventions.LLM_MODEL_NAME]: modelName,
                [SemanticConventions.INPUT_VALUE]:
                  safelyJSONStringify(params) || "",
                [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
                [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
                  safelyJSONStringify(params?.config) || "{}",
                [SemanticConventions.LLM_SYSTEM]: LLMSystem.VERTEXAI,
                [SemanticConventions.LLM_PROVIDER]: LLMProvider.GOOGLE,
                ...getInputMessagesAttributes(params),
                ...getToolsJSONSchema(params),
              },
            },
          );

          const execContext = getExecContext(span);
          const execPromise = safeExecuteInTheMiddle(
            () => {
              return context.with(trace.setSpan(execContext, span), () => {
                return originalGenerateContentStream(...genArgs);
              });
            },
            (error: Error | undefined) => {
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

          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          return execPromise.then(async (stream: any) => {
            const chunks = [];
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            let lastChunk: any;
            for await (const chunk of stream) {
              chunks.push(chunk);
              lastChunk = chunk;
            }

            if (lastChunk) {
              span.setAttributes({
                [SemanticConventions.OUTPUT_VALUE]:
                  safelyJSONStringify(lastChunk) || "",
                [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
                ...getOutputMessagesAttributes(lastChunk),
                ...getUsageAttributes(lastChunk),
              });
            }

            span.setStatus({ code: SpanStatusCode.OK });
            span.end();

            return (async function* () {
              for (const chunk of chunks) {
                yield chunk;
              }
            })();
          });
        };
    }

    // Wrap generateImages if it exists
    if (models.generateImages) {
      const originalGenerateImages = models.generateImages.bind(models);
      models.generateImages = async function patchedGenerateImages(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ...genArgs: any[]
      ) {
        const params = genArgs[0];
        const modelName = params?.model || "unknown";

        const span = instrumentation.oiTracer.startSpan(
          `Google GenAI Generate Images`,
          {
            kind: SpanKind.INTERNAL,
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.LLM,
              [SemanticConventions.LLM_MODEL_NAME]: modelName,
              [SemanticConventions.INPUT_VALUE]:
                safelyJSONStringify(params) || "",
              [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
              [SemanticConventions.LLM_SYSTEM]: LLMSystem.VERTEXAI,
              [SemanticConventions.LLM_PROVIDER]: LLMProvider.GOOGLE,
            },
          },
        );

        const execContext = getExecContext(span);
        const execPromise = safeExecuteInTheMiddle(
          () => {
            return context.with(trace.setSpan(execContext, span), () => {
              return originalGenerateImages(...genArgs);
            });
          },
          (error: Error | undefined) => {
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

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return execPromise.then((result: any) => {
          span.setAttributes({
            [SemanticConventions.OUTPUT_VALUE]:
              safelyJSONStringify(result) || "",
            [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
          });
          span.setStatus({ code: SpanStatusCode.OK });
          span.end();
          return result;
        });
      };
    }
  }

  /**
   * Patches methods on a Chats instance and its created Chat sessions
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private patchChatsInstance(
    chats: any,
    instrumentation: GoogleGenAIInstrumentation,
  ) {
    // Wrap the create method to patch Chat instances
    if (chats.create) {
      const originalCreate = chats.create.bind(chats);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      chats.create = function patchedCreate(...args: any[]) {
        const chatSession = originalCreate(...args);

        // Patch the chat session methods
        instrumentation.patchChatInstance(chatSession, instrumentation);

        return chatSession;
      };
    }
  }

  /**
   * Patches methods on a Chat instance
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private patchChatInstance(
    chat: any,
    instrumentation: GoogleGenAIInstrumentation,
  ) {
    // Wrap sendMessage
    if (chat.sendMessage) {
      const originalSendMessage = chat.sendMessage.bind(chat);
      chat.sendMessage = async function patchedSendMessage(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ...sendArgs: any[]
      ) {
        const params = sendArgs[0];

        const span = instrumentation.oiTracer.startSpan(
          `Google GenAI Chat Send Message`,
          {
            kind: SpanKind.INTERNAL,
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.CHAIN,
              [SemanticConventions.INPUT_VALUE]:
                safelyJSONStringify(params?.message || params) || "",
              [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
            },
          },
        );

        const execContext = getExecContext(span);
        const execPromise = safeExecuteInTheMiddle(
          () => {
            return context.with(trace.setSpan(execContext, span), () => {
              return originalSendMessage(...sendArgs);
            });
          },
          (error: Error | undefined) => {
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

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return execPromise.then((result: any) => {
          span.setAttributes({
            [SemanticConventions.OUTPUT_VALUE]:
              safelyJSONStringify(result) || "",
            [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
          });
          span.setStatus({ code: SpanStatusCode.OK });
          span.end();
          return result;
        });
      };
    }

    // Wrap sendMessageStream
    if (chat.sendMessageStream) {
      const originalSendMessageStream = chat.sendMessageStream.bind(chat);
      chat.sendMessageStream = async function patchedSendMessageStream(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ...sendArgs: any[]
      ) {
        const params = sendArgs[0];

        const span = instrumentation.oiTracer.startSpan(
          `Google GenAI Chat Send Message Stream`,
          {
            kind: SpanKind.INTERNAL,
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.CHAIN,
              [SemanticConventions.INPUT_VALUE]:
                safelyJSONStringify(params?.message || params) || "",
              [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
            },
          },
        );

        const execContext = getExecContext(span);
        const execPromise = safeExecuteInTheMiddle(
          () => {
            return context.with(trace.setSpan(execContext, span), () => {
              return originalSendMessageStream(...sendArgs);
            });
          },
          (error: Error | undefined) => {
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

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return execPromise.then(async (stream: any) => {
          const chunks = [];
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          let lastChunk: any;
          for await (const chunk of stream) {
            chunks.push(chunk);
            lastChunk = chunk;
          }

          if (lastChunk) {
            span.setAttributes({
              [SemanticConventions.OUTPUT_VALUE]:
                safelyJSONStringify(lastChunk) || "",
              [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
            });
          }

          span.setStatus({ code: SpanStatusCode.OK });
          span.end();

          return (async function* () {
            for (const chunk of chunks) {
              yield chunk;
            }
          })();
        });
      };
    }
  }

  /**
   * Patches methods on a Batches instance
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private patchBatchesInstance(
    batches: any,
    instrumentation: GoogleGenAIInstrumentation,
  ) {
    // Wrap createEmbeddings
    if (batches.createEmbeddings) {
      const originalCreateEmbeddings = batches.createEmbeddings.bind(batches);
      batches.createEmbeddings = async function patchedCreateEmbeddings(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ...args: any[]
      ) {
        const params = args[0];
        const modelName = params?.model || "unknown";

        const span = instrumentation.oiTracer.startSpan(
          `Google GenAI Batch Create Embeddings`,
          {
            kind: SpanKind.INTERNAL,
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.EMBEDDING,
              [SemanticConventions.EMBEDDING_MODEL_NAME]: modelName,
              [SemanticConventions.INPUT_VALUE]:
                safelyJSONStringify(params) || "",
              [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
              [SemanticConventions.LLM_SYSTEM]: LLMSystem.VERTEXAI,
              [SemanticConventions.LLM_PROVIDER]: LLMProvider.GOOGLE,
            },
          },
        );

        const execContext = getExecContext(span);
        const execPromise = safeExecuteInTheMiddle(
          () => {
            return context.with(trace.setSpan(execContext, span), () => {
              return originalCreateEmbeddings(...args);
            });
          },
          (error: Error | undefined) => {
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

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return execPromise.then((result: any) => {
          span.setAttributes({
            [SemanticConventions.OUTPUT_VALUE]:
              safelyJSONStringify(result) || "",
            [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
          });
          span.setStatus({ code: SpanStatusCode.OK });
          span.end();
          return result;
        });
      };
    }
  }

  /**
   * Un-patches the Google Gen AI module
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private unpatch(moduleExports: any, moduleVersion?: string) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);

    // Note: We directly replace constructors, so unpatch is not fully supported
    // In practice, instrumentation is rarely disabled after being enabled

    _isOpenInferencePatched = false;
    try {
      moduleExports.openInferencePatched = false;
    } catch (e) {
      diag.warn(`Failed to unset ${MODULE_NAME} patched flag on the module`, e);
    }
  }
}

/**
 * Converts the request to input messages attributes
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getInputMessagesAttributes(params: any): Attributes {
  const attributes: Attributes = {};

  if (!params?.contents) {
    return attributes;
  }

  const contents = Array.isArray(params.contents)
    ? params.contents
    : [params.contents];

  contents.forEach(
    (
      content: // eslint-disable-next-line @typescript-eslint/no-explicit-any
      any,
      index: number,
    ) => {
      const indexPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.`;

      if (typeof content === "string") {
        attributes[`${indexPrefix}${SemanticConventions.MESSAGE_ROLE}`] =
          "user";
        attributes[`${indexPrefix}${SemanticConventions.MESSAGE_CONTENT}`] =
          content;
      } else if (content.role) {
        attributes[`${indexPrefix}${SemanticConventions.MESSAGE_ROLE}`] =
          content.role;
        if (content.parts) {
          content.parts.forEach(
            (
              part: // eslint-disable-next-line @typescript-eslint/no-explicit-any
              any,
              partIndex: number,
            ) => {
              const partPrefix = `${indexPrefix}${SemanticConventions.MESSAGE_CONTENTS}.${partIndex}.`;
              if (part.text) {
                attributes[
                  `${partPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`
                ] = "text";
                attributes[
                  `${partPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`
                ] = part.text;
              }
            },
          );
        }
      }
    },
  );

  return attributes;
}

/**
 * Converts tool definitions to JSON schema attributes
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getToolsJSONSchema(params: any): Attributes {
  const attributes: Attributes = {};

  if (params?.config?.tools && Array.isArray(params.config.tools)) {
    params.config.tools.forEach(
      (
        tool: // eslint-disable-next-line @typescript-eslint/no-explicit-any
        any,
        index: number,
      ) => {
        const toolJsonSchema = safelyJSONStringify(tool);
        if (toolJsonSchema) {
          attributes[
            `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`
          ] = toolJsonSchema;
        }
      },
    );
  }

  return attributes;
}

/**
 * Converts the response to output messages attributes
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getOutputMessagesAttributes(response: any): Attributes {
  const attributes: Attributes = {};

  if (response?.candidates && response.candidates.length > 0) {
    const candidate = response.candidates[0];
    const indexPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.`;

    if (candidate.content) {
      attributes[`${indexPrefix}${SemanticConventions.MESSAGE_ROLE}`] =
        candidate.content.role || "model";

      if (candidate.content.parts) {
        candidate.content.parts.forEach(
          (
            part: // eslint-disable-next-line @typescript-eslint/no-explicit-any
            any,
            partIndex: number,
          ) => {
            const partPrefix = `${indexPrefix}${SemanticConventions.MESSAGE_CONTENTS}.${partIndex}.`;

            if (part.text) {
              attributes[
                `${partPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`
              ] = "text";
              attributes[
                `${partPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`
              ] = part.text;
            }

            if (part.functionCall) {
              const toolCallPrefix = `${indexPrefix}${SemanticConventions.MESSAGE_TOOL_CALLS}.${partIndex}.`;
              attributes[
                `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
              ] = part.functionCall.name;
              attributes[
                `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
              ] = safelyJSONStringify(part.functionCall.args) || "{}";
            }
          },
        );
      }
    }
  }

  return attributes;
}

/**
 * Get usage attributes from response
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getUsageAttributes(response: any): Attributes {
  if (response?.usageMetadata) {
    const usage = response.usageMetadata;
    return {
      [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: usage.promptTokenCount || 0,
      [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]:
        usage.candidatesTokenCount || 0,
      [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: usage.totalTokenCount || 0,
      [SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]:
        usage.cachedContentTokenCount || undefined,
    };
  }

  return {};
}
