import type {
  Batches,
  Chat,
  Chats,
  Content,
  GenerateContentParameters,
  GenerateContentResponse,
  GoogleGenAI,
  GoogleGenAIOptions,
  Models,
  Part,
} from "@google/genai";
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

// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore - No version file until build
import { VERSION } from "./version";

const MODULE_NAME = "@google/genai";

const INSTRUMENTATION_NAME = "@arizeai/openinference-instrumentation-google-genai";

type GenAIModule = typeof import("@google/genai");

type PatchableModule = GenAIModule & { openInferencePatched?: boolean };

type GoogleGenAICtor = new (options: GoogleGenAIOptions) => GoogleGenAI;

let _isOpenInferencePatched = false;

/**
 * function to check if instrumentation is enabled / disabled
 */
export function isPatched() {
  return _isOpenInferencePatched;
}

/**
 * Resolves the execution context for the current span.
 * If tracing is suppressed, the span is dropped from the active context.
 */
function getExecContext(span: Span) {
  const activeContext = context.active();
  const suppressTracing = isTracingSuppressed(activeContext);
  const execContext = suppressTracing ? trace.setSpan(context.active(), span) : activeContext;
  if (suppressTracing) {
    trace.deleteSpan(activeContext);
  }
  return execContext;
}

/**
 * An auto instrumentation class for the Google Gen AI SDK that creates
 * {@link https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md|OpenInference}
 * compliant spans for Google Gemini API calls.
 */
export class GoogleGenAIInstrumentation extends InstrumentationBase<GenAIModule> {
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
    super(INSTRUMENTATION_NAME, VERSION, Object.assign({}, instrumentationConfig));
    this.tracerProvider = tracerProvider;
    this.traceConfig = traceConfig;
    this.oiTracer = new OITracer({
      tracer: this.tracerProvider?.getTracer(INSTRUMENTATION_NAME, VERSION) ?? this.tracer,
      traceConfig,
    });
  }

  protected init(): InstrumentationModuleDefinition<GenAIModule> {
    return new InstrumentationNodeModuleDefinition<GenAIModule>(
      MODULE_NAME,
      ["^1.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
  }

  /**
   * Manually instruments the Google Gen AI module.
   * This is needed when the module is not loaded via require (commonjs).
   */
  manuallyInstrument(module: GenAIModule) {
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
   * Patches a `GoogleGenAI` instance directly. Useful when running under ESM
   * where module hooks are not available, or when the user already has a
   * `GoogleGenAI` instance and wants to instrument it after construction
   * (e.g. inside a serverless handler).
   */
  public instrumentInstance(instance: GoogleGenAI): void {
    if (instance.models) {
      this.patchModelsInstance(instance.models);
    }
    if (instance.chats) {
      this.patchChatsInstance(instance.chats);
    }
    if (instance.batches) {
      this.patchBatchesInstance(instance.batches);
    }
  }

  /**
   * Patches the Google Gen AI module by wrapping the `GoogleGenAI`
   * constructor so that every newly constructed instance is instrumented
   * automatically.
   */
  private patch(module: GenAIModule, moduleVersion?: string): GenAIModule {
    diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);

    const patchable = module as PatchableModule;
    if (patchable.openInferencePatched || _isOpenInferencePatched) {
      return module;
    }

    const OriginalGoogleGenAI = module.GoogleGenAI as GoogleGenAICtor;
    if (!OriginalGoogleGenAI) {
      diag.warn(`Cannot find GoogleGenAI in ${MODULE_NAME}@${moduleVersion}`);
      return module;
    }

    const instrumentation = this;

    function PatchedGoogleGenAI(this: GoogleGenAI, options: GoogleGenAIOptions) {
      const instance = new OriginalGoogleGenAI(options);
      try {
        instrumentation.instrumentInstance(instance);
      } catch (e) {
        diag.warn(`Failed to instrument GoogleGenAI instance`, e);
      }
      return instance;
    }
    PatchedGoogleGenAI.prototype = OriginalGoogleGenAI.prototype;

    try {
      (module as { GoogleGenAI: GoogleGenAICtor }).GoogleGenAI =
        PatchedGoogleGenAI as unknown as GoogleGenAICtor;
      patchable.openInferencePatched = true;
    } catch (e) {
      diag.warn(`Failed to patch GoogleGenAI on ${MODULE_NAME}`, e);
    }

    _isOpenInferencePatched = true;
    return module;
  }

  /**
   * Unpatches the Google Gen AI module.
   * Note: instances already constructed remain instrumented; we only restore
   * the module-level `GoogleGenAI` export.
   */
  private unpatch(module: GenAIModule, moduleVersion?: string): GenAIModule {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);
    const patchable = module as PatchableModule;
    _isOpenInferencePatched = false;
    try {
      patchable.openInferencePatched = false;
    } catch (e) {
      diag.warn(`Failed to unset ${MODULE_NAME} patched flag on the module`, e);
    }
    return module;
  }

  /**
   * Patches request methods on a `Models` instance.
   */
  private patchModelsInstance(models: Models): void {
    if (typeof models.generateContent === "function") {
      const original = models.generateContent.bind(models) as Models["generateContent"];
      models.generateContent = this.wrapPromiseMethod(
        "Google GenAI Generate Content",
        original,
        (params) => getGenerateContentInputAttributes(params),
        (response) => getGenerateContentOutputAttributes(response),
      );
    }

    if (typeof models.generateContentStream === "function") {
      const original = models.generateContentStream.bind(models) as Models["generateContentStream"];
      models.generateContentStream = this.wrapStreamMethod(
        "Google GenAI Generate Content Stream",
        original,
        (params) => getGenerateContentInputAttributes(params),
        (chunk) => getGenerateContentOutputAttributes(chunk),
      );
    }

    if (typeof models.generateImages === "function") {
      const original = models.generateImages.bind(models) as Models["generateImages"];
      models.generateImages = this.wrapPromiseMethod(
        "Google GenAI Generate Images",
        original,
        (params) => getImageGenerationInputAttributes(params),
        () => ({}),
      );
    }
  }

  /**
   * Patches `Chats.create` so that every chat session returned is instrumented.
   */
  private patchChatsInstance(chats: Chats): void {
    if (typeof chats.create !== "function") {
      return;
    }
    const originalCreate = chats.create.bind(chats) as Chats["create"];
    const instrumentation = this;
    chats.create = function patchedCreate(
      ...args: Parameters<Chats["create"]>
    ): ReturnType<Chats["create"]> {
      const chatSession = originalCreate(...args);
      instrumentation.patchChatInstance(chatSession);
      return chatSession;
    } as Chats["create"];
  }

  /**
   * Patches request methods on a `Chat` instance.
   */
  private patchChatInstance(chat: Chat): void {
    if (typeof chat.sendMessage === "function") {
      const original = chat.sendMessage.bind(chat) as Chat["sendMessage"];
      chat.sendMessage = this.wrapPromiseMethod(
        "Google GenAI Chat Send Message",
        original,
        (params) => getSendMessageInputAttributes(params),
        (response) => getGenerateContentOutputAttributes(response),
        OpenInferenceSpanKind.LLM,
      );
    }

    if (typeof chat.sendMessageStream === "function") {
      const original = chat.sendMessageStream.bind(chat) as Chat["sendMessageStream"];
      chat.sendMessageStream = this.wrapStreamMethod(
        "Google GenAI Chat Send Message Stream",
        original,
        (params) => getSendMessageInputAttributes(params),
        (chunk) => getGenerateContentOutputAttributes(chunk),
        OpenInferenceSpanKind.LLM,
      );
    }
  }

  /**
   * Patches request methods on a `Batches` instance.
   */
  private patchBatchesInstance(batches: Batches): void {
    const batchesAny = batches as unknown as {
      createEmbeddings?: (params: unknown) => Promise<unknown>;
    };
    if (typeof batchesAny.createEmbeddings === "function") {
      const original = batchesAny.createEmbeddings.bind(batchesAny);
      batchesAny.createEmbeddings = this.wrapPromiseMethod(
        "Google GenAI Batch Create Embeddings",
        original,
        (params) => getEmbeddingsInputAttributes(params),
        () => ({}),
        OpenInferenceSpanKind.EMBEDDING,
      );
    }
  }

  /**
   * Wraps a method that returns a Promise<TResult> with span tracking.
   */
  private wrapPromiseMethod<TArgs extends unknown[], TResult, TParams = TArgs[0]>(
    spanName: string,
    original: (...args: TArgs) => Promise<TResult>,
    getInputAttributes: (params: TParams) => Attributes,
    getOutputAttributes: (result: TResult) => Attributes,
    spanKind: OpenInferenceSpanKind = OpenInferenceSpanKind.LLM,
  ): (...args: TArgs) => Promise<TResult> {
    const instrumentation = this;
    return async function patched(...args: TArgs): Promise<TResult> {
      const params = args[0] as TParams;
      const span = instrumentation.startRequestSpan(spanName, spanKind, params, getInputAttributes);
      const execContext = getExecContext(span);
      const execPromise = safeExecuteInTheMiddle<Promise<TResult>>(
        () => {
          return context.with(trace.setSpan(execContext, span), () => {
            return original(...args);
          });
        },
        (error) => {
          if (error) {
            endSpanWithError(span, error);
          }
        },
      );
      return execPromise.then(
        (result) => {
          span.setAttributes({
            [SemanticConventions.OUTPUT_VALUE]: safelyJSONStringify(result) || "",
            [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
            ...getOutputAttributes(result),
            ...getUsageAttributes(result),
          });
          span.setStatus({ code: SpanStatusCode.OK });
          span.end();
          return result;
        },
        (error: Error) => {
          endSpanWithError(span, error);
          throw error;
        },
      );
    };
  }

  /**
   * Wraps a method that returns a Promise<AsyncGenerator<TChunk>> with
   * span tracking that lasts for the lifetime of the stream consumption.
   */
  private wrapStreamMethod<TArgs extends unknown[], TChunk, TParams = TArgs[0]>(
    spanName: string,
    original: (...args: TArgs) => Promise<AsyncGenerator<TChunk>>,
    getInputAttributes: (params: TParams) => Attributes,
    getChunkAttributes: (chunk: TChunk) => Attributes,
    spanKind: OpenInferenceSpanKind = OpenInferenceSpanKind.LLM,
  ): (...args: TArgs) => Promise<AsyncGenerator<TChunk>> {
    const instrumentation = this;
    return async function patched(...args: TArgs): Promise<AsyncGenerator<TChunk>> {
      const params = args[0] as TParams;
      const span = instrumentation.startRequestSpan(spanName, spanKind, params, getInputAttributes);
      const execContext = getExecContext(span);
      const execPromise = safeExecuteInTheMiddle<Promise<AsyncGenerator<TChunk>>>(
        () => {
          return context.with(trace.setSpan(execContext, span), () => {
            return original(...args);
          });
        },
        (error) => {
          if (error) {
            endSpanWithError(span, error);
          }
        },
      );

      let stream: AsyncGenerator<TChunk>;
      try {
        stream = await execPromise;
      } catch (error) {
        endSpanWithError(span, error as Error);
        throw error;
      }

      return (async function* instrumentedStream(): AsyncGenerator<TChunk> {
        let lastChunk: TChunk | undefined;
        let errored = false;
        try {
          for await (const chunk of stream) {
            lastChunk = chunk;
            yield chunk;
          }
        } catch (error) {
          errored = true;
          endSpanWithError(span, error as Error);
          throw error;
        } finally {
          if (!errored) {
            if (lastChunk !== undefined) {
              span.setAttributes({
                [SemanticConventions.OUTPUT_VALUE]: safelyJSONStringify(lastChunk) || "",
                [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
                ...getChunkAttributes(lastChunk),
                ...getUsageAttributes(lastChunk),
              });
            }
            span.setStatus({ code: SpanStatusCode.OK });
            span.end();
          }
        }
      })();
    };
  }

  /**
   * Starts a request span with the common base attributes.
   */
  private startRequestSpan<TParams>(
    spanName: string,
    spanKind: OpenInferenceSpanKind,
    params: TParams,
    getInputAttributes: (params: TParams) => Attributes,
  ): Span {
    const baseAttributes: Attributes = {
      [SemanticConventions.OPENINFERENCE_SPAN_KIND]: spanKind,
      [SemanticConventions.INPUT_VALUE]: safelyJSONStringify(params) || "",
      [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
      [SemanticConventions.LLM_SYSTEM]: LLMSystem.VERTEXAI,
      [SemanticConventions.LLM_PROVIDER]: LLMProvider.GOOGLE,
    };
    return this.oiTracer.startSpan(spanName, {
      kind: SpanKind.INTERNAL,
      attributes: { ...baseAttributes, ...getInputAttributes(params) },
    });
  }
}

function endSpanWithError(span: Span, error: Error): void {
  span.recordException(error);
  span.setStatus({
    code: SpanStatusCode.ERROR,
    message: error.message,
  });
  span.end();
}

/**
 * Builds attributes for a `generateContent` / `generateContentStream` request.
 */
function getGenerateContentInputAttributes(
  params: GenerateContentParameters | undefined,
): Attributes {
  const attributes: Attributes = {};
  if (!params) return attributes;

  if (params.model) {
    attributes[SemanticConventions.LLM_MODEL_NAME] = params.model;
  }
  if (params.config !== undefined) {
    attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
      safelyJSONStringify(params.config) || "{}";
  }
  Object.assign(attributes, getInputMessagesAttributes(params.contents));
  Object.assign(attributes, getToolsJSONSchema(params));
  return attributes;
}

function getSendMessageInputAttributes(params: unknown): Attributes {
  const attributes: Attributes = {};
  if (!params || typeof params !== "object") return attributes;
  const p = params as { message?: unknown; config?: unknown };
  if (p.config !== undefined) {
    attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
      safelyJSONStringify(p.config) || "{}";
  }
  if (p.message !== undefined) {
    Object.assign(attributes, getInputMessagesAttributes(p.message));
  }
  return attributes;
}

function getImageGenerationInputAttributes(params: unknown): Attributes {
  const attributes: Attributes = {};
  if (!params || typeof params !== "object") return attributes;
  const p = params as { model?: string };
  if (p.model) {
    attributes[SemanticConventions.LLM_MODEL_NAME] = p.model;
  }
  return attributes;
}

function getEmbeddingsInputAttributes(params: unknown): Attributes {
  const attributes: Attributes = {};
  if (!params || typeof params !== "object") return attributes;
  const p = params as { model?: string };
  if (p.model) {
    attributes[SemanticConventions.EMBEDDING_MODEL_NAME] = p.model;
  }
  return attributes;
}

/**
 * Converts a `contents` payload to OpenInference input message attributes.
 *
 * Handles the various shapes accepted by `generateContent`:
 * - a single string
 * - a single Part / Content
 * - an array of strings, Parts, or Contents
 *
 * Within a Content, function calls and function responses are emitted as
 * separate messages so multi-turn tool conversations render correctly.
 */
function getInputMessagesAttributes(contents: unknown): Attributes {
  const attributes: Attributes = {};
  if (contents == null) return attributes;

  const list = Array.isArray(contents) ? contents : [contents];
  let messageIndex = 0;
  for (const item of list) {
    messageIndex = appendInputMessage(attributes, item, messageIndex);
  }
  return attributes;
}

/**
 * Appends one or more messages for a single `contents` entry. Returns the
 * next message index. A single Content with both a `functionCall` and
 * `functionResponse` is split into two messages so the trace shows the
 * tool call and tool result distinctly.
 */
function appendInputMessage(attributes: Attributes, item: unknown, startIndex: number): number {
  if (item == null) return startIndex;

  if (typeof item === "string") {
    setMessage(attributes, startIndex, "user", [{ text: item }]);
    return startIndex + 1;
  }

  if (typeof item !== "object") return startIndex;

  // Bare Part (no role / no parts wrapper)
  if (looksLikePart(item)) {
    return appendPartsAsMessages(attributes, "user", [item as Part], startIndex);
  }

  const content = item as Content;
  const role = content.role || "user";
  const parts = content.parts ?? [];
  return appendPartsAsMessages(attributes, role, parts, startIndex);
}

function looksLikePart(value: object): boolean {
  const v = value as Partial<Part> & { role?: unknown; parts?: unknown };
  if (v.role !== undefined || v.parts !== undefined) return false;
  return (
    "text" in v ||
    "functionCall" in v ||
    "functionResponse" in v ||
    "inlineData" in v ||
    "fileData" in v
  );
}

/**
 * Splits a list of Parts into messages such that function responses become
 * their own `tool` role message (matching OpenAI-style trace expectations).
 */
function appendPartsAsMessages(
  attributes: Attributes,
  role: string,
  parts: Part[],
  startIndex: number,
): number {
  let index = startIndex;
  let buffer: Part[] = [];

  const flush = (flushRole: string) => {
    if (buffer.length === 0) return;
    setMessage(attributes, index, flushRole, buffer);
    index += 1;
    buffer = [];
  };

  for (const part of parts) {
    if (part.functionResponse) {
      flush(role);
      setToolMessage(attributes, index, part);
      index += 1;
      continue;
    }
    buffer.push(part);
  }
  flush(role);

  return index;
}

function setMessage(
  attributes: Attributes,
  messageIndex: number,
  role: string,
  parts: Part[],
): void {
  const prefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.`;
  attributes[`${prefix}${SemanticConventions.MESSAGE_ROLE}`] = role;

  // Emit a flat content string when the message is plain text.
  const textParts = parts.filter((p) => typeof p.text === "string");
  if (textParts.length > 0 && parts.every((p) => typeof p.text === "string")) {
    attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT}`] = textParts
      .map((p) => p.text)
      .join("");
  }

  parts.forEach((part, partIndex) => {
    if (typeof part.text === "string") {
      const partPrefix = `${prefix}${SemanticConventions.MESSAGE_CONTENTS}.${partIndex}.`;
      attributes[`${partPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "text";
      attributes[`${partPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = part.text;
    }
    if (part.functionCall) {
      const toolCallPrefix = `${prefix}${SemanticConventions.MESSAGE_TOOL_CALLS}.${partIndex}.`;
      if (part.functionCall.name) {
        attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`] =
          part.functionCall.name;
      }
      attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`] =
        safelyJSONStringify(part.functionCall.args ?? {}) || "{}";
      if (part.functionCall.id) {
        attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_ID}`] = part.functionCall.id;
      }
    }
  });
}

function setToolMessage(attributes: Attributes, messageIndex: number, part: Part): void {
  const prefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.`;
  attributes[`${prefix}${SemanticConventions.MESSAGE_ROLE}`] = "tool";
  const fnResponse = part.functionResponse;
  if (fnResponse?.name) {
    attributes[`${prefix}${SemanticConventions.MESSAGE_NAME}`] = fnResponse.name;
  }
  if (fnResponse?.id) {
    attributes[`${prefix}${SemanticConventions.MESSAGE_TOOL_CALL_ID}`] = fnResponse.id;
  }
  if (fnResponse?.response !== undefined) {
    attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT}`] =
      safelyJSONStringify(fnResponse.response) || "";
  }
}

function getToolsJSONSchema(params: GenerateContentParameters): Attributes {
  const attributes: Attributes = {};
  const tools = params.config?.tools;
  if (!Array.isArray(tools)) return attributes;
  tools.forEach((tool, index) => {
    const toolJsonSchema = safelyJSONStringify(tool);
    if (toolJsonSchema) {
      attributes[
        `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`
      ] = toolJsonSchema;
    }
  });
  return attributes;
}

/**
 * Builds output message attributes from a generateContent response (or
 * the latest chunk of a stream).
 */
function getGenerateContentOutputAttributes(
  response: GenerateContentResponse | undefined,
): Attributes {
  const attributes: Attributes = {};
  const candidates = response?.candidates;
  if (!candidates || candidates.length === 0) return attributes;

  const candidate = candidates[0];
  const content = candidate.content;
  if (!content) return attributes;

  const prefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.`;
  attributes[`${prefix}${SemanticConventions.MESSAGE_ROLE}`] = content.role || "model";

  const parts = content.parts ?? [];
  const textParts = parts.filter((p) => typeof p.text === "string");
  if (textParts.length > 0 && parts.every((p) => typeof p.text === "string")) {
    attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT}`] = textParts
      .map((p) => p.text)
      .join("");
  }

  parts.forEach((part, partIndex) => {
    if (typeof part.text === "string") {
      const partPrefix = `${prefix}${SemanticConventions.MESSAGE_CONTENTS}.${partIndex}.`;
      attributes[`${partPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "text";
      attributes[`${partPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = part.text;
    }
    if (part.functionCall) {
      const toolCallPrefix = `${prefix}${SemanticConventions.MESSAGE_TOOL_CALLS}.${partIndex}.`;
      if (part.functionCall.name) {
        attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`] =
          part.functionCall.name;
      }
      attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`] =
        safelyJSONStringify(part.functionCall.args ?? {}) || "{}";
      if (part.functionCall.id) {
        attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_ID}`] = part.functionCall.id;
      }
    }
  });
  return attributes;
}

function getUsageAttributes(response: unknown): Attributes {
  if (!response || typeof response !== "object") return {};
  const usage = (response as { usageMetadata?: Record<string, number> }).usageMetadata;
  if (!usage) return {};
  const attributes: Attributes = {
    [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: usage.promptTokenCount || 0,
    [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: usage.candidatesTokenCount || 0,
    [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: usage.totalTokenCount || 0,
  };
  if (usage.cachedContentTokenCount) {
    attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] =
      usage.cachedContentTokenCount;
  }
  return attributes;
}
