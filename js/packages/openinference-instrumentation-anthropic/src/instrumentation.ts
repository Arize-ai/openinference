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

import type { Metadata, SpanOutput, TraceConfigOptions } from "@arizeai/openinference-core";
import {
  getInputAttributes,
  getMetadata,
  getMetadataAttributes,
  getOutputAttributes,
  OITracer,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
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

/**
 * The name of the span of a `beta.messages.toolRunner()` run, the same as the
 * Python instrumentation's for `beta.messages.tool_runner()`.
 */
const TOOL_RUNNER_SPAN_NAME = "beta.messages.tool_runner";

type MessageCreateParams =
  | Parameters<typeof Anthropic.Messages.prototype.create>[0]
  | Parameters<typeof Anthropic.Beta.Messages.prototype.create>[0];
/** The params of a request with messages: a messages create or a token count */
type MessageRequestParams =
  | MessageCreateParams
  | Parameters<typeof Anthropic.Messages.prototype.countTokens>[0]
  | Parameters<typeof Anthropic.Beta.Messages.prototype.countTokens>[0];
type TokensCount =
  | Anthropic.Messages.MessageTokensCount
  | Anthropic.Beta.Messages.BetaMessageTokensCount;
type ToolRunnerParams = Parameters<typeof Anthropic.Beta.Messages.prototype.toolRunner>[0];
type ToolRunner = ReturnType<typeof Anthropic.Beta.Messages.prototype.toolRunner>;
type RunnableTool = Extract<ToolRunnerParams["tools"][number], { parse: unknown }>;
type MessageParam = Anthropic.Messages.MessageParam | Anthropic.Beta.Messages.BetaMessageParam;
type Message = Anthropic.Messages.Message | Anthropic.Beta.Messages.BetaMessage;
type ContentBlock = Anthropic.Messages.ContentBlock | Anthropic.Beta.Messages.BetaContentBlock;
type ContentBlockParam =
  | Anthropic.Messages.ContentBlockParam
  | Anthropic.Beta.Messages.BetaContentBlockParam;
type ToolUseContentBlock = Extract<
  ContentBlock | ContentBlockParam,
  { type: "tool_use" | "server_tool_use" | "mcp_tool_use" }
>;
type ImageBlockParam =
  | Anthropic.Messages.ImageBlockParam
  | Anthropic.Beta.Messages.BetaImageBlockParam;
type ToolDefinition = NonNullable<MessageRequestParams["tools"]>[number];
/**
 * The parts of a message the output attributes are read from, shared by a
 * parsed message and one rebuilt from a stream.
 */
type OutputMessage = Pick<Message, "role"> & { content: ContentBlock[] };
type RawMessageStreamEvent =
  | Anthropic.Messages.RawMessageStreamEvent
  | Anthropic.Beta.Messages.BetaRawMessageStreamEvent;
type MessageUsage =
  | Anthropic.Messages.Usage
  | Anthropic.Messages.MessageDeltaUsage
  | Anthropic.Beta.Messages.BetaUsage
  | Anthropic.Beta.Messages.BetaMessageDeltaUsage
  | Anthropic.Beta.Messages.BetaFallbackMessageIterationUsage;
type MessageUsageCounts = Partial<
  Pick<
    MessageUsage,
    "input_tokens" | "output_tokens" | "cache_creation_input_tokens" | "cache_read_input_tokens"
  >
>;
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
 * patched first would block the other one forever (#3557). A Set is
 * scoped to the object, and needs no write to the module, so it also keeps
 * the double-patch guard working when the module is immutable (e.g. Deno,
 * webpack) and the `openInferencePatched` property cannot be set. Entries
 * are removed when their wrappers are removed, so the Set does not retain
 * unpatched SDK builds.
 */
const _patchedModules = new Set<object>();

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
  private readonly patchedModuleExports = new Map<
    typeof Anthropic,
    typeof Anthropic & { openInferencePatched?: boolean }
  >();

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

  disable(): void {
    super.disable();
    for (const moduleExports of [...this.patchedModuleExports.values()]) {
      this.unpatch(moduleExports);
    }
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
          const contextMetadata = getMetadata(context.active());
          const span = instrumentation.oiTracer.startSpan(`Anthropic Messages`, {
            kind: SpanKind.INTERNAL,
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
              [SemanticConventions.LLM_SYSTEM]: LLMSystem.ANTHROPIC,
              [SemanticConventions.LLM_PROVIDER]: getAnthropicProvider(this),
              ...safelyGetAttributes(() => getAnthropicRequestAttributes(body)),
            },
          });
          return traceAnthropicAPIPromise<Message | Stream<RawMessageStreamEvent>>({
            span,
            // oxlint-disable-next-line typescript/no-unsafe-type-assertion
            invoke: () => Reflect.apply(original, this, args) as ReturnType<CreateType>,
            onResult: (result, { endSpan, recordError }) => {
              if (isAnthropicMessageResponse(result)) {
                // Record the results
                span.setAttributes(
                  safelyGetAttributes(() =>
                    getAnthropicResponseAttributes(result, contextMetadata),
                  ),
                );
                span.setStatus({ code: SpanStatusCode.OK });
                endSpan();
              } else if (isAnthropicStream(result)) {
                // Observe the events as the caller reads them, so the caller's
                // stream keeps its own iterator (break / abort still cancel the
                // request) and nothing is read ahead or buffered on its behalf.
                instrumentAnthropicStream({
                  stream: result,
                  span,
                  endSpan,
                  recordError,
                  contextMetadata,
                });
              } else {
                endSpan();
              }
              return result;
            },
          });
        };
      };

    this._wrap(anthropicModule.Messages.prototype, "create", patchCreate);
    if (betaMessages?.prototype?.create) {
      this._wrap(betaMessages.prototype, "create", patchCreate);
    } else {
      // Beta instrumentation is optional: the stable Messages patch still applies.
      diag.debug(`Cannot find Beta.Messages.prototype.create in ${MODULE_NAME}@${moduleVersion}`);
    }

    // Patch stable and beta messages.countTokens: an LLM span with the
    // request and the counted tokens as its output.
    type CountTokensType =
      | typeof anthropicModule.Messages.prototype.countTokens
      | typeof anthropicModule.Beta.Messages.prototype.countTokens;
    const patchCountTokens =
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      <CountType extends CountTokensType>(original: CountType): any => {
        return function patchedCountTokens(this: unknown, ...args: Parameters<CountType>) {
          const body = args[0];
          const span = instrumentation.oiTracer.startSpan(`Anthropic Count Tokens`, {
            kind: SpanKind.INTERNAL,
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
              [SemanticConventions.LLM_SYSTEM]: LLMSystem.ANTHROPIC,
              [SemanticConventions.LLM_PROVIDER]: getAnthropicProvider(this),
              ...safelyGetAttributes(() => getAnthropicRequestAttributes(body)),
            },
          });
          return traceAnthropicAPIPromise<TokensCount>({
            span,
            // oxlint-disable-next-line typescript/no-unsafe-type-assertion
            invoke: () => Reflect.apply(original, this, args) as PromiseLike<TokensCount>,
            onResult: (result, { endSpan }) => {
              span.setAttributes(
                safelyGetAttributes(() => getOutputAttributes(toJSONOutput(result))),
              );
              span.setStatus({ code: SpanStatusCode.OK });
              endSpan();
              return result;
            },
          });
        };
      };
    if (typeof anthropicModule.Messages.prototype.countTokens === "function") {
      this._wrap(anthropicModule.Messages.prototype, "countTokens", patchCountTokens);
    }
    if (betaMessages?.prototype?.countTokens) {
      this._wrap(betaMessages.prototype, "countTokens", patchCountTokens);
    }

    // Patch beta.messages.toolRunner: a CHAIN span for the run, parenting the
    // LLM span of each request and a TOOL span for each tool it runs.
    type ToolRunnerType = typeof anthropicModule.Beta.Messages.prototype.toolRunner;
    const patchToolRunner =
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (original: ToolRunnerType): any => {
        return function patchedToolRunner(this: unknown, ...args: Parameters<ToolRunnerType>) {
          const [params, ...rest] = args;
          const span = instrumentation.oiTracer.startSpan(TOOL_RUNNER_SPAN_NAME, {
            kind: SpanKind.INTERNAL,
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
              [SemanticConventions.LLM_SYSTEM]: LLMSystem.ANTHROPIC,
              [SemanticConventions.LLM_PROVIDER]: getAnthropicProvider(this),
              ...safelyGetAttributes(() => getInputAttributes(toJSONOutput(params))),
            },
          });
          const runContext = trace.setSpan(getExecContext(span), span);
          const tracedParams = traceRunnableTools({
            params,
            oiTracer: instrumentation.oiTracer,
          });
          let runner: ReturnType<ToolRunnerType>;
          try {
            runner = context.with(runContext, () =>
              Reflect.apply(original, this, [tracedParams, ...rest]),
            );
          } catch (error) {
            recordSpanError(span, error);
            span.end();
            throw error;
          }
          instrumentToolRunner({ runner, span, runContext });
          return runner;
        };
      };
    if (betaMessages?.prototype?.toolRunner) {
      this._wrap(betaMessages.prototype, "toolRunner", patchToolRunner);
    }

    _isOpenInferencePatched = true;
    _patchedModules.add(anthropicModule);
    this.patchedModuleExports.set(anthropicModule, module);
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
    if (typeof anthropicModule.Messages.prototype.countTokens === "function") {
      this._unwrap(anthropicModule.Messages.prototype, "countTokens");
    }
    if (betaMessages?.prototype?.create) {
      this._unwrap(betaMessages.prototype, "create");
    }
    if (betaMessages?.prototype?.countTokens) {
      this._unwrap(betaMessages.prototype, "countTokens");
    }
    if (betaMessages?.prototype?.toolRunner) {
      this._unwrap(betaMessages.prototype, "toolRunner");
    }

    // Keyed the same way patch() keys it, so a re-patch is possible after.
    _patchedModules.delete(anthropicModule);
    this.patchedModuleExports.delete(anthropicModule);
    _isOpenInferencePatched = _patchedModules.size > 0;
    try {
      // This can fail if the module is made immutable via the runtime or bundler
      moduleExports.openInferencePatched = false;
    } catch (e) {
      diag.warn(`Failed to unset ${MODULE_NAME} patched flag on the module`, e);
    }
  }
}

/**
 * Ends a span at most once, and records an error on it before it ends.
 */
interface SpanLifecycle {
  endSpan: () => void;
  recordError: (error: unknown) => void;
}

/**
 * Calls an SDK method returning an `APIPromise` in the span's context, and
 * hands its parsed result to `onResult`, which records it and ends the span.
 *
 * The result stays an `APIPromise` (through `_thenUnwrap`) so `withResponse()`
 * and `asResponse()` keep working, which `client.messages.stream()` relies on.
 * A request failure is recorded without triggering the parse, and a caller
 * that takes the raw response ends the span, as nothing is parsed for it.
 */
function traceAnthropicAPIPromise<T>({
  span,
  invoke,
  onResult,
}: {
  span: Span;
  invoke: () => PromiseLike<T>;
  onResult: (result: T, lifecycle: SpanLifecycle) => T;
}) {
  const execContext = getExecContext(span);
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
  const recordError = (error: unknown) => {
    if (spanEnded) {
      return;
    }
    recordSpanError(span, error);
    endSpan();
  };
  const lifecycle: SpanLifecycle = { endSpan, recordError };

  const execPromise = safeExecuteInTheMiddle(
    () => context.with(trace.setSpan(execContext, span), invoke),
    (error: Error | undefined) => {
      // Push the error to the span
      if (error) {
        recordError(error);
      }
    },
  );
  const handleResult = (result: T) => onResult(result, lifecycle);

  if (hasThenUnwrap(execPromise)) {
    const wrappedPromise = execPromise._thenUnwrap(handleResult);
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
    // before the result is recorded.
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

  const wrappedPromise = Promise.resolve(execPromise)
    .then(handleResult)
    .catch((error: unknown) => {
      recordError(error);
      throw error;
    });
  return context.bind(execContext, wrappedPromise);
}

/**
 * True when create() returned an APIPromise we can transform with _thenUnwrap.
 */
function hasThenUnwrap<T>(promise: PromiseLike<T>): promise is APIPromise<T> {
  return "_thenUnwrap" in promise && typeof promise._thenUnwrap === "function";
}

/**
 * Identifies the cloud serving a request from the client the patched resource
 * belongs to: the Bedrock (including Mantle), Vertex and Foundry SDK clients
 * reuse this SDK's `Messages` resource, so the same patch traces them.
 * Detected by class name (AWS / Google Cloud named clients included), and by
 * the fields the Bedrock and Vertex clients carry for when a bundler renames
 * classes.
 */
function getAnthropicProvider(resource: unknown): LLMProvider {
  try {
    const client: unknown =
      resource != null && typeof resource === "object" ? Reflect.get(resource, "_client") : null;
    if (client == null || typeof client !== "object") {
      return LLMProvider.ANTHROPIC;
    }
    const name = client.constructor?.name ?? "";
    if (/Bedrock|AWS/.test(name) || "awsRegion" in client) {
      return LLMProvider.AWS;
    }
    if (/Vertex|GoogleCloud/.test(name) || "projectId" in client) {
      return LLMProvider.GOOGLE;
    }
    if (/Foundry|Azure/.test(name)) {
      return LLMProvider.AZURE;
    }
  } catch (error) {
    diag.debug(`Failed to identify the ${MODULE_NAME} client's provider`, error);
  }
  return LLMProvider.ANTHROPIC;
}

/**
 * Runs an attribute extractor so that a malformed request or response can never
 * make the instrumentation throw into the caller's code. A failure costs the
 * attributes it would have produced, and is reported through `diag`.
 */
function safelyGetAttributes(getAttributes: () => Attributes): Attributes {
  try {
    return getAttributes();
  } catch (error) {
    diag.warn(`Failed to extract ${MODULE_NAME} span attributes`, error);
    return {};
  }
}

/**
 * Records an error on the span and sets its status, without ending it.
 */
function recordSpanError(span: Span, error: unknown) {
  if (error instanceof Error) {
    span.recordException(error);
    span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
  } else {
    span.recordException(String(error));
    span.setStatus({ code: SpanStatusCode.ERROR, message: String(error) });
  }
}

/**
 * Builds the request attributes of an Anthropic messages request.
 */
function getAnthropicRequestAttributes(body: MessageRequestParams): Attributes {
  const { messages: _messages, ...invocationParameters } = body;
  const attributes: Attributes = {
    [SemanticConventions.LLM_MODEL_NAME]: body.model,
    [SemanticConventions.LLM_REQUEST_MODEL_NAME]: body.model,
    ...getAnthropicInputMessagesAttributes(body),
    ...getAnthropicToolsJSONSchema(body),
  };
  const input = safelyJSONStringify(body);
  if (input != null) {
    attributes[SemanticConventions.INPUT_VALUE] = input;
    attributes[SemanticConventions.INPUT_MIME_TYPE] = MimeType.JSON;
  }
  const invocationParametersJSON = safelyJSONStringify(invocationParameters);
  if (invocationParametersJSON != null) {
    attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS] = invocationParametersJSON;
  }
  return attributes;
}

/**
 * Builds the response attributes of a non-streaming Anthropic message.
 */
function getAnthropicResponseAttributes(result: Message, contextMetadata?: Metadata): Attributes {
  const attributes: Attributes = {
    // Override the model from the value sent by the server
    [SemanticConventions.LLM_MODEL_NAME]: result.model,
    [SemanticConventions.LLM_RESPONSE_MODEL_NAME]: result.model,
    ...getAnthropicFinishReasonAttributes(result.stop_reason),
    ...getAnthropicOutputMessagesAttributes(result),
    ...getAnthropicUsageAttributes(result.usage ?? {}),
    ...getAnthropicReasoningTokenAttributes(result.usage ?? {}),
    ...getAnthropicMetadataAttributes({
      metadata: getAnthropicResponseMetadata(result),
      contextMetadata,
    }),
  };
  const output = safelyJSONStringify(result);
  if (output != null) {
    attributes[SemanticConventions.OUTPUT_VALUE] = output;
    attributes[SemanticConventions.OUTPUT_MIME_TYPE] = MimeType.JSON;
  }
  return attributes;
}

/**
 * Observes an Anthropic stream through the iterator its caller reads, and ends
 * the span when the caller is done with it:
 *
 * - read to the end: the span ends with status OK;
 * - left early (`break`, `return()`, `toReadableStream()` cancel) or aborted
 *   through `stream.controller`: the span ends with what was read so far and
 *   the status left unset, as the request did not run to completion;
 * - failed (a network error or an `error` event): the error is recorded on
 *   the span, which ends with status ERROR, and the error still reaches the
 *   caller.
 *
 * The SDK `Stream` builds every iterator it hands out (`for await`, `tee()`,
 * `toReadableStream()`) from its `iterator` factory, so wrapping that factory
 * observes all of them while leaving the stream's own controller, early-exit
 * abort and helpers untouched.
 */
function instrumentAnthropicStream({
  stream,
  span,
  endSpan,
  recordError,
  contextMetadata,
}: {
  stream: Stream<RawMessageStreamEvent>;
  span: Span;
  endSpan: () => void;
  recordError: (error: unknown) => void;
  contextMetadata?: Metadata;
}) {
  const state = createAnthropicStreamState();
  const recordStreamAttributes = () => {
    span.setAttributes(
      safelyGetAttributes(() => getAnthropicStreamAttributes(state, contextMetadata)),
    );
  };
  const observer: AnthropicStreamObserver = {
    onEvent: (event) => {
      try {
        applyAnthropicStreamEvent({ chunk: event, state });
      } catch (error) {
        diag.warn(`Failed to record a ${MODULE_NAME} stream event`, error);
      }
    },
    onDone: () => {
      recordStreamAttributes();
      if (!stream.controller.signal.aborted) {
        span.setStatus({ code: SpanStatusCode.OK });
      }
      endSpan();
    },
    onReturn: () => {
      recordStreamAttributes();
      endSpan();
    },
    onError: (error) => {
      recordStreamAttributes();
      recordError(error);
    },
  };

  const iteratorFactory: unknown = Reflect.get(stream, "iterator");
  if (typeof iteratorFactory === "function") {
    Reflect.set(stream, "iterator", () =>
      wrapAnthropicStreamIterator({
        iterator: getAsyncIterator(Reflect.apply(iteratorFactory, stream, [])),
        observer,
      }),
    );
    return;
  }
  // Fall back to the async iterator protocol should the SDK stop building its
  // iterators from `iterator`.
  const asyncIterator = stream[Symbol.asyncIterator].bind(stream);
  Object.defineProperty(stream, Symbol.asyncIterator, {
    configurable: true,
    writable: true,
    value: () => wrapAnthropicStreamIterator({ iterator: asyncIterator(), observer }),
  });
}

/**
 * Narrows the value returned by the SDK stream's iterator factory.
 */
function getAsyncIterator(value: unknown): AsyncIterator<RawMessageStreamEvent> {
  if (isAsyncIterator(value)) {
    return value;
  }
  throw new TypeError(`${MODULE_NAME} stream iterator factory did not return an iterator`);
}

function isAsyncIterator(value: unknown): value is AsyncIterator<RawMessageStreamEvent> {
  return (
    value != null &&
    typeof value === "object" &&
    "next" in value &&
    typeof value.next === "function"
  );
}

interface AnthropicStreamObserver {
  onEvent: (event: RawMessageStreamEvent) => void;
  onDone: () => void;
  onReturn: () => void;
  onError: (error: unknown) => void;
}

/**
 * Wraps a stream iterator so that each event, the end of the stream, an early
 * return and a failure are reported to the observer. Calls are forwarded to the
 * SDK iterator unchanged, so its `return()` still aborts the HTTP request when
 * the caller leaves the loop early.
 */
function wrapAnthropicStreamIterator({
  iterator,
  observer,
}: {
  iterator: AsyncIterator<RawMessageStreamEvent>;
  observer: AnthropicStreamObserver;
}): AsyncIterator<RawMessageStreamEvent> {
  return {
    async next() {
      let result: IteratorResult<RawMessageStreamEvent>;
      try {
        result = await iterator.next();
      } catch (error) {
        observer.onError(error);
        throw error;
      }
      if (result.done) {
        observer.onDone();
      } else {
        observer.onEvent(result.value);
      }
      return result;
    },
    async return(value?: unknown) {
      try {
        const result: IteratorResult<RawMessageStreamEvent> = iterator.return
          ? await iterator.return(value)
          : { done: true, value };
        observer.onReturn();
        return result;
      } catch (error) {
        observer.onError(error);
        throw error;
      }
    },
  };
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
    [`${prefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`]: `${block.from?.model} -> ${block.to?.model}${reason}`,
  };
}

/**
 * Converts the body of an Anthropic messages request to LLM input messages
 */
function getAnthropicInputMessagesAttributes(body: MessageRequestParams): Attributes {
  // The top-level system prompt is the first input message, ahead of the
  // conversation, so the prompt the model saw is recorded in order.
  const messagesAttributes = [
    ...(body.system != null ? [getAnthropicSystemMessageAttributes(body.system)] : []),
    ...(body.messages ?? []).map((message) => getAnthropicInputMessageAttributes(message)),
  ];
  return messagesAttributes.reduce<Attributes>((acc, messageAttributes, index) => {
    const indexPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.`;
    // Flatten the attributes on the index prefix
    for (const [key, value] of Object.entries(messageAttributes)) {
      acc[`${indexPrefix}${key}`] = value;
    }
    return acc;
  }, {});
}

/**
 * Converts the top-level `system` request param, a string or a list of text
 * blocks, to a message with the system role.
 */
function getAnthropicSystemMessageAttributes(
  system: NonNullable<MessageRequestParams["system"]>,
): Attributes {
  const attributes: Attributes = { [SemanticConventions.MESSAGE_ROLE]: "system" };
  if (typeof system === "string") {
    attributes[SemanticConventions.MESSAGE_CONTENT] = system;
    return attributes;
  }
  system.forEach((block, index) => {
    const contentsIndexPrefix = `${SemanticConventions.MESSAGE_CONTENTS}.${index}.`;
    attributes[`${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "text";
    attributes[`${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = block.text;
  });
  return attributes;
}

/**
 * Lists the tool definitions of a request: the `tools` param, followed by the
 * tools a `tool_addition` block of a beta request defines inline.
 */
function getAnthropicToolDefinitions(body: MessageRequestParams): ToolDefinition[] {
  const tools: ToolDefinition[] = [...(body.tools ?? [])];
  for (const message of body.messages ?? []) {
    if (typeof message.content === "string") {
      continue;
    }
    for (const part of message.content) {
      if (part.type === "tool_addition" && part.tool.type === "tool_definition") {
        tools.push(part.tool.definition);
      }
    }
  }
  return tools;
}

/**
 * Converts each tool definition into the `llm.tools` attributes: its JSON
 * schema, plus the name and description the spec defines next to it.
 */
function getAnthropicToolsJSONSchema(body: MessageRequestParams): Attributes {
  return getAnthropicToolDefinitions(body).reduce((acc: Attributes, tool, index) => {
    const toolPrefix = `${SemanticConventions.LLM_TOOLS}.${index}.`;
    const toolJsonSchema = safelyJSONStringify(tool);
    if (toolJsonSchema) {
      acc[`${toolPrefix}${SemanticConventions.TOOL_JSON_SCHEMA}`] = toolJsonSchema;
    }
    if ("name" in tool && typeof tool.name === "string" && tool.name) {
      acc[`${toolPrefix}${SemanticConventions.TOOL_NAME}`] = tool.name;
    }
    if ("description" in tool && typeof tool.description === "string" && tool.description) {
      acc[`${toolPrefix}${SemanticConventions.TOOL_DESCRIPTION}`] = tool.description;
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
        Object.assign(attributes, getAnthropicImageContentAttributes(contentsIndexPrefix, part));
      } else if (isToolUseContentBlock(part)) {
        Object.assign(
          attributes,
          getAnthropicToolUseContentAttributes({
            block: part,
            contentPrefix: contentsIndexPrefix,
            toolCallPrefix: `${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolIndex}.`,
          }),
        );
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
      } else {
        Object.assign(attributes, getAnthropicOtherContentAttributes(contentsIndexPrefix, part));
      }
    });
  }

  return attributes;
}

/**
 * Converts the Anthropic message result to LLM output attributes
 */
function getAnthropicOutputMessagesAttributes(message: OutputMessage): Attributes {
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
    } else if (isToolUseContentBlock(content)) {
      // Server and MCP tool calls are recorded as tool calls too, so every
      // call the model made is visible, whoever ran it.
      Object.assign(
        attributes,
        getAnthropicToolUseContentAttributes({
          block: content,
          contentPrefix,
          toolCallPrefix: `${indexPrefix}${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolIndex}.`,
        }),
      );
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
    } else {
      Object.assign(attributes, getAnthropicOtherContentAttributes(contentPrefix, content));
    }
  });

  return attributes;
}

/**
 * Collects the response details that have no semantic convention of their
 * own, for the span's `metadata`:
 *
 * - `stop_details`: why the model stopped, e.g. the category of a refusal;
 * - `cache_miss_reason`: why the prompt cache was not read;
 * - `container_id`: the container a code execution tool ran in;
 * - `context_management`: the context edits applied to the request;
 * - `input_transformations`: the input blocks the API dropped or let through;
 * - `usage_iterations`: the usage of each sampling iteration. A compaction
 *   iteration's tokens are not part of the top-level usage, so they are not
 *   added to the token counts either;
 * - `server_tool_use`: the number of server tool requests by tool;
 * - `citations`: the citations of each text block, by content index;
 * - `mcp_server_names`: the MCP server of each `mcp_tool_use` call, by id.
 */
function getAnthropicResponseMetadata(message: AnthropicResponseDetails): Metadata {
  const metadata: Metadata = {
    ...getAnthropicResponseDetailsMetadata(message),
    ...getAnthropicUsageMetadata(message.usage),
  };
  const { content } = message;
  const citations: Record<string, unknown> = {};
  const mcpServerNames: Record<string, string> = {};
  content.forEach((block, index) => {
    if (block.type === "text" && block.citations != null && block.citations.length > 0) {
      citations[String(index)] = block.citations;
    } else if (block.type === "mcp_tool_use") {
      mcpServerNames[block.id] = block.server_name;
    }
  });
  if (Object.keys(citations).length > 0) {
    metadata.citations = citations;
  }
  if (Object.keys(mcpServerNames).length > 0) {
    metadata.mcp_server_names = mcpServerNames;
  }
  return metadata;
}

/**
 * The response fields metadata is read from, shared by a parsed message and
 * one rebuilt from a stream, whose fields are only known once they arrive.
 */
type AnthropicResponseDetails = { content: ContentBlock[]; usage?: unknown } & Partial<
  Record<
    "stop_details" | "diagnostics" | "container" | "context_management" | "input_transformations",
    unknown
  >
>;

function getAnthropicResponseDetailsMetadata(message: AnthropicResponseDetails): Metadata {
  const metadata: Metadata = {};
  if (message.stop_details != null) {
    metadata.stop_details = message.stop_details;
  }
  const diagnostics = message.diagnostics;
  if (
    diagnostics != null &&
    typeof diagnostics === "object" &&
    "cache_miss_reason" in diagnostics &&
    diagnostics.cache_miss_reason != null
  ) {
    metadata.cache_miss_reason = diagnostics.cache_miss_reason;
  }
  const container = message.container;
  if (
    container != null &&
    typeof container === "object" &&
    "id" in container &&
    typeof container.id === "string"
  ) {
    metadata.container_id = container.id;
  }
  if (message.context_management != null) {
    metadata.context_management = message.context_management;
  }
  if (Array.isArray(message.input_transformations) && message.input_transformations.length > 0) {
    metadata.input_transformations = message.input_transformations;
  }
  return metadata;
}

function getAnthropicUsageMetadata(usage: unknown): Metadata {
  const metadata: Metadata = {};
  if (usage == null || typeof usage !== "object") {
    return metadata;
  }
  if ("iterations" in usage && usage.iterations != null) {
    metadata.usage_iterations = usage.iterations;
  }
  if ("server_tool_use" in usage && usage.server_tool_use != null) {
    metadata.server_tool_use = usage.server_tool_use;
  }
  return metadata;
}

/**
 * Records response metadata merged into the metadata set on the context, so
 * setting `metadata` again does not drop it. Keys the caller set win.
 */
function getAnthropicMetadataAttributes({
  metadata,
  contextMetadata,
}: {
  metadata: Metadata;
  contextMetadata?: Metadata;
}): Attributes {
  if (Object.keys(metadata).length === 0) {
    return {};
  }
  return getMetadataAttributes({ ...metadata, ...contextMetadata });
}

/**
 * type-guard for the content blocks that call a tool: a client tool
 * (`tool_use`), a server tool (`server_tool_use`) or an MCP tool
 * (`mcp_tool_use`).
 */
function isToolUseContentBlock(
  block: ContentBlock | ContentBlockParam,
): block is ToolUseContentBlock {
  return (
    block.type === "tool_use" || block.type === "server_tool_use" || block.type === "mcp_tool_use"
  );
}

/**
 * Records a tool call both as a `tool_use` content item and as an entry of
 * `message.tool_calls`. Streamed input is kept as the raw JSON received when
 * it could not be parsed, as for a stream the caller left early.
 */
function getAnthropicToolUseContentAttributes({
  block,
  contentPrefix,
  toolCallPrefix,
}: {
  block: ToolUseContentBlock;
  contentPrefix: string;
  toolCallPrefix: string;
}): Attributes {
  const input: unknown = block.input;
  const argumentsJSON = typeof input === "string" ? input : (safelyJSONStringify(input) ?? "");
  return {
    [`${toolCallPrefix}${SemanticConventions.TOOL_CALL_ID}`]: block.id,
    [`${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]: block.name,
    [`${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]: argumentsJSON,
    [`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`]: "tool_use",
    [`${contentPrefix}${SemanticConventions.TOOL_CALL_ID}`]: block.id,
    [`${contentPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]: block.name,
    [`${contentPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]: argumentsJSON,
  };
}

/**
 * Records an image input. A base64 image becomes a data URI (which trace
 * config can truncate or hide), a URL image its URL, and a Files API image the
 * file id, as there is no URL for it.
 */
function getAnthropicImageContentAttributes(prefix: string, block: ImageBlockParam): Attributes {
  const imagePrefix = `${prefix}${SemanticConventions.MESSAGE_CONTENT_IMAGE}.`;
  const attributes: Attributes = {
    [`${prefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`]: "image",
  };
  const source = block.source;
  if (source.type === "base64") {
    attributes[`${imagePrefix}type`] = "base64";
    attributes[`${imagePrefix}media_type`] = source.media_type;
    attributes[`${imagePrefix}${SemanticConventions.IMAGE_URL}`] =
      `data:${source.media_type};base64,${source.data}`;
  } else if (source.type === "url") {
    attributes[`${imagePrefix}${SemanticConventions.IMAGE_URL}`] = source.url;
  } else if (source.type === "file") {
    attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT_ID}`] = source.file_id;
  }
  return attributes;
}

/**
 * Records a content block without a dedicated mapping. A compaction summary
 * is text the model reads in place of the conversation it replaced. Any other
 * block (server tool results, MCP results and listings, container uploads,
 * documents, tool changes, and types this version does not know) is recorded
 * as its type and its JSON, so it still occupies its index in the contents
 * list rather than leaving a hole.
 */
function getAnthropicOtherContentAttributes(
  prefix: string,
  block: ContentBlock | ContentBlockParam,
): Attributes {
  if (block.type === "compaction") {
    const attributes: Attributes = {
      [`${prefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`]: "text",
    };
    if (block.content != null) {
      attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = block.content;
    }
    if (block.signature != null) {
      attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT_SIGNATURE}`] = block.signature;
    }
    return attributes;
  }
  const attributes: Attributes = {
    [`${prefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`]: String(block.type),
  };
  const blockJSON = safelyJSONStringify(block);
  if (blockJSON != null) {
    attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = blockJSON;
  }
  return attributes;
}

/**
 * Get usage attributes from Anthropic response.
 *
 * Anthropic reports `input_tokens` exclusive of cached tokens, so the cache
 * read and write counts are added back into the prompt count. The OpenInference
 * spec treats `prompt_details.*` as sub-counts of `llm.token_count.prompt`, so
 * the prompt count has to include them.
 * https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#tracking-cache-performance
 */
function getAnthropicUsageAttributes(usage: MessageUsageCounts): Attributes {
  const attributes: Attributes = {};
  const cacheWriteTokens = usage.cache_creation_input_tokens ?? 0;
  const cacheReadTokens = usage.cache_read_input_tokens ?? 0;
  const promptTokens =
    usage.input_tokens != null
      ? usage.input_tokens + cacheWriteTokens + cacheReadTokens
      : undefined;

  if (promptTokens != null) {
    attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] = promptTokens;
  }
  if (usage.output_tokens != null) {
    attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] = usage.output_tokens;
  }
  if (promptTokens != null && usage.output_tokens != null) {
    attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL] = promptTokens + usage.output_tokens;
  }
  // Anthropic reports these as 0 when caching did not run, so zeros are left
  // out rather than tagging every span with them.
  if (cacheWriteTokens > 0) {
    attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] = cacheWriteTokens;
  }
  if (cacheReadTokens > 0) {
    attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] = cacheReadTokens;
  }
  return attributes;
}

/**
 * Records the output tokens spent on extended thinking, which Anthropic
 * reports as a breakdown of (and so included in) `output_tokens`.
 */
function getAnthropicReasoningTokenAttributes(usage: object): Attributes {
  const details: unknown =
    "output_tokens_details" in usage ? usage.output_tokens_details : undefined;
  if (
    details != null &&
    typeof details === "object" &&
    "thinking_tokens" in details &&
    typeof details.thinking_tokens === "number"
  ) {
    return {
      [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING]: details.thinking_tokens,
    };
  }
  return {};
}

/**
 * Merges two usage reports field by field, keeping the later value whenever it
 * is present. Stream usage counts are cumulative, so a later event that omits a
 * field keeps the earlier value, and a later 0 replaces an earlier count.
 */
function mergeAnthropicUsage(
  earlier: MessageUsageCounts,
  later: MessageUsageCounts,
): MessageUsageCounts {
  return {
    input_tokens: later.input_tokens ?? earlier.input_tokens,
    output_tokens: later.output_tokens ?? earlier.output_tokens,
    cache_creation_input_tokens:
      later.cache_creation_input_tokens ?? earlier.cache_creation_input_tokens,
    cache_read_input_tokens: later.cache_read_input_tokens ?? earlier.cache_read_input_tokens,
  };
}

/**
 * Mutable state accumulated while consuming an Anthropic message stream.
 *
 * The message is rebuilt from the events as the SDK's own `MessageStream`
 * does, so a stream is recorded with the same attributes and the same JSON
 * output as the message a non-streaming request returns.
 *
 * Usage is reported per attempt: message_start describes the first attempt,
 * which on a server-side fallback stream is the one that declined. Each source
 * is captured on its own so the precedence between them is stated once, where
 * they are merged in {@link getAnthropicStreamAttributes}.
 */
interface AnthropicStreamState {
  /** The message of `message_start`, without its content */
  message?: Omit<Message, "content">;
  /** The content blocks, by their stream `index` */
  content: ContentBlock[];
  /** The raw JSON streamed so far for the input of each tool call, by block index */
  partialInputs: Map<number, string>;
  /** The fields `message_delta` events set on the message */
  messageDelta: Record<string, unknown>;
  /** Every usage field reported by `message_delta` events, latest first */
  deltaUsageReport: Record<string, unknown>;
  startUsage: MessageUsageCounts;
  deltaUsage: MessageUsageCounts;
  servingUsage?: MessageUsageCounts;
  responseModel?: string;
  finishReason?: string;
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
  const usage = chunk.usage ?? {};
  state.deltaUsage = mergeAnthropicUsage(state.deltaUsage, usage);
  state.deltaUsageReport = { ...state.deltaUsageReport, ...withoutNullValues(usage) };
  state.messageDelta = { ...state.messageDelta, ...withoutNullValues(chunk.delta ?? {}) };
  if ("context_management" in chunk && chunk.context_management != null) {
    state.messageDelta.context_management = chunk.context_management;
  }
  if ("input_transformations" in chunk && chunk.input_transformations != null) {
    state.messageDelta.input_transformations = chunk.input_transformations;
  }
  if (chunk.delta?.stop_reason != null) {
    state.finishReason = chunk.delta.stop_reason;
  }
  if (!("iterations" in usage) || usage.iterations == null) {
    return;
  }
  for (const iteration of usage.iterations) {
    if (iteration.type === "fallback_message") {
      state.responseModel = iteration.model;
      state.servingUsage = iteration;
    }
  }
}

/**
 * Drops the fields of a delta that are null, which a stream sends for the
 * fields an event leaves unchanged.
 */
function withoutNullValues(record: object): Record<string, unknown> {
  return Object.fromEntries(Object.entries(record).filter(([, value]) => value != null));
}

/**
 * Applies a `content_block_start` event, adding a copy of the block, so the
 * events the caller receives are never modified.
 */
function applyAnthropicContentBlockStart({
  chunk,
  state,
}: {
  chunk: Extract<RawMessageStreamEvent, { type: "content_block_start" }>;
  state: AnthropicStreamState;
}) {
  const contentBlock = chunk.content_block;
  state.content[chunk.index] = { ...contentBlock };
  if (contentBlock.type === "fallback") {
    state.responseModel = contentBlock.to?.model ?? state.responseModel;
  }
}

/**
 * Applies a `content_block_delta` event to the block it belongs to,
 * accumulating streamed text, reasoning, citations, compaction summaries and
 * tool call arguments.
 */
function applyAnthropicContentBlockDelta({
  chunk,
  state,
}: {
  chunk: Extract<RawMessageStreamEvent, { type: "content_block_delta" }>;
  state: AnthropicStreamState;
}) {
  const block = state.content[chunk.index];
  const delta = chunk.delta;
  if (block == null) {
    return;
  }
  if (delta.type === "text_delta" && block.type === "text") {
    block.text += delta.text;
  } else if (delta.type === "citations_delta" && block.type === "text") {
    Object.assign(block, { citations: [...(block.citations ?? []), delta.citation] });
  } else if (delta.type === "thinking_delta" && block.type === "thinking") {
    block.thinking += delta.thinking;
  } else if (delta.type === "signature_delta") {
    Object.assign(block, { signature: delta.signature });
  } else if (delta.type === "compaction_delta" && block.type === "compaction") {
    applyAnthropicCompactionDelta(block, delta);
  } else if (delta.type === "input_json_delta" && isToolUseContentBlock(block)) {
    const partialInput = (state.partialInputs.get(chunk.index) ?? "") + delta.partial_json;
    state.partialInputs.set(chunk.index, partialInput);
    // Kept as the raw JSON until it parses, which it may never do when the
    // caller leaves the stream early.
    Object.assign(block, { input: parseToolInput(partialInput) });
  }
}

function applyAnthropicCompactionDelta(
  block: Extract<ContentBlock, { type: "compaction" }>,
  delta: Anthropic.Beta.Messages.BetaCompactionContentBlockDelta,
) {
  if (delta.content != null) {
    block.content = (block.content ?? "") + delta.content;
  }
  if (delta.encrypted_content != null) {
    block.encrypted_content = delta.encrypted_content;
  }
}

function parseToolInput(partialInput: string): unknown {
  try {
    return JSON.parse(partialInput);
  } catch {
    return partialInput;
  }
}

/**
 * Rebuilds the message a stream delivered, as far as it was read.
 */
function getAnthropicStreamMessage(state: AnthropicStreamState): Record<string, unknown> {
  // On a server-side fallback stream the serving hop's usage replaces
  // message_start's, which belongs to the attempt that declined.
  const startUsage = state.servingUsage ?? state.message?.usage ?? {};
  return {
    ...state.message,
    ...(state.responseModel != null && { model: state.responseModel }),
    content: state.content.filter((block) => block != null),
    ...state.messageDelta,
    usage: { ...startUsage, ...state.deltaUsageReport },
  };
}

/**
 * Builds the span attributes for a consumed Anthropic message stream.
 */
function getAnthropicStreamAttributes(
  state: AnthropicStreamState,
  contextMetadata?: Metadata,
): Attributes {
  const content = state.content.filter((block) => block != null);
  const message = getAnthropicStreamMessage(state);
  const attributes: Attributes = {
    ...getAnthropicOutputMessagesAttributes({ role: state.message?.role ?? "assistant", content }),
    ...getAnthropicMetadataAttributes({
      metadata: getAnthropicResponseMetadata({ ...message, content }),
      contextMetadata,
    }),
  };
  if (message.usage != null && typeof message.usage === "object") {
    Object.assign(attributes, getAnthropicReasoningTokenAttributes(message.usage));
  }

  const output = safelyJSONStringify(message);
  if (output != null) {
    attributes[SemanticConventions.OUTPUT_VALUE] = output;
    attributes[SemanticConventions.OUTPUT_MIME_TYPE] = MimeType.JSON;
  }

  if (state.responseModel != null) {
    // Override the model from the value sent by the server
    attributes[SemanticConventions.LLM_MODEL_NAME] = state.responseModel;
    attributes[SemanticConventions.LLM_RESPONSE_MODEL_NAME] = state.responseModel;
  }

  if (state.finishReason != null) {
    attributes[SemanticConventions.LLM_FINISH_REASON] = state.finishReason;
  }

  // On a server-side fallback stream the serving hop's usage replaces
  // message_start's, which belongs to the attempt that declined. The final
  // message_delta then wins field by field, so prompt, completion and cache
  // counts all describe the same model. Prompt and total are derived only
  // after merging so they stay consistent with the cache counts.
  const usage = mergeAnthropicUsage(state.servingUsage ?? state.startUsage, state.deltaUsage);
  Object.assign(attributes, getAnthropicUsageAttributes(usage));

  return attributes;
}

/**
 * Creates the empty state a stream's events are accumulated into.
 */
function createAnthropicStreamState(): AnthropicStreamState {
  return {
    content: [],
    partialInputs: new Map(),
    messageDelta: {},
    deltaUsageReport: {},
    startUsage: {},
    deltaUsage: {},
  };
}

/**
 * Applies one stream event to the accumulated state.
 */
function applyAnthropicStreamEvent({
  chunk,
  state,
}: {
  chunk: RawMessageStreamEvent;
  state: AnthropicStreamState;
}) {
  if (chunk.type === "message_start") {
    const { content: _content, ...message } = chunk.message;
    state.message = message;
    state.responseModel = chunk.message.model;
    state.startUsage = chunk.message.usage ?? {};
  } else if (chunk.type === "message_delta") {
    applyAnthropicMessageDelta({ chunk, state });
  } else if (chunk.type === "content_block_start") {
    applyAnthropicContentBlockStart({ chunk, state });
  } else if (chunk.type === "content_block_delta") {
    applyAnthropicContentBlockDelta({ chunk, state });
  }
}

/**
 * Serializes a value as a JSON input or output, or nothing when it cannot be.
 */
function toJSONOutput(value: unknown): SpanOutput | undefined {
  const json = safelyJSONStringify(value);
  return json == null ? undefined : { value: json, mimeType: MimeType.JSON };
}

/**
 * type-guard for a tool the tool runner runs on the client: a tool definition
 * with the `run` and `parse` the runner calls. Toolsets, which have a `run`
 * but no `parse`, are left alone.
 */
function isRunnableTool(tool: ToolRunnerParams["tools"][number]): tool is RunnableTool {
  return (
    tool != null &&
    typeof tool === "object" &&
    "run" in tool &&
    typeof tool.run === "function" &&
    "parse" in tool &&
    typeof tool.parse === "function"
  );
}

/**
 * Returns the tool runner params with each runnable tool traced. The caller's
 * params and tools are never modified: the tools are proxies that trace `run`
 * and pass everything else through.
 */
function traceRunnableTools({
  params,
  oiTracer,
}: {
  params: ToolRunnerParams;
  oiTracer: OITracer;
}): ToolRunnerParams {
  try {
    if (!Array.isArray(params?.tools)) {
      return params;
    }
    return {
      ...params,
      tools: params.tools.map((tool) =>
        isRunnableTool(tool) ? traceRunnableTool({ tool, oiTracer }) : tool,
      ),
    };
  } catch (error) {
    diag.warn(`Failed to trace the ${MODULE_NAME} tool runner's tools`, error);
    return params;
  }
}

/**
 * Wraps a runnable tool so each run is a TOOL span, a child of the span
 * active when the runner calls it.
 */
function traceRunnableTool({
  tool,
  oiTracer,
}: {
  tool: RunnableTool;
  oiTracer: OITracer;
}): RunnableTool {
  const run = tool.run.bind(tool);
  const tracedRun: RunnableTool["run"] = (args, runContext) => {
    const span = oiTracer.startSpan(tool.name, {
      kind: SpanKind.INTERNAL,
      attributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
        ...safelyGetAttributes(() => getRunnableToolAttributes({ tool, args })),
      },
    });
    let result: ReturnType<RunnableTool["run"]>;
    try {
      result = context.with(trace.setSpan(context.active(), span), () => run(args, runContext));
    } catch (error) {
      recordSpanError(span, error);
      span.end();
      throw error;
    }
    return Promise.resolve(result).then(
      (output) => {
        span.setAttributes(
          safelyGetAttributes(() =>
            getOutputAttributes(typeof output === "string" ? output : toJSONOutput(output)),
          ),
        );
        span.setStatus({ code: SpanStatusCode.OK });
        span.end();
        return output;
      },
      (error: unknown) => {
        recordSpanError(span, error);
        span.end();
        throw error;
      },
    );
  };
  return new Proxy(tool, {
    get(target, property, receiver) {
      return property === "run" ? tracedRun : Reflect.get(target, property, receiver);
    },
  });
}

/**
 * The attributes of a tool run: the tool, its parameters schema and the
 * arguments it was called with.
 */
function getRunnableToolAttributes({
  tool,
  args,
}: {
  tool: RunnableTool;
  args: unknown;
}): Attributes {
  const attributes: Attributes = {
    [SemanticConventions.TOOL_NAME]: tool.name,
    ...getInputAttributes(toJSONOutput(args)),
  };
  if ("description" in tool && typeof tool.description === "string" && tool.description) {
    attributes[SemanticConventions.TOOL_DESCRIPTION] = tool.description;
  }
  if ("input_schema" in tool) {
    const parameters = safelyJSONStringify(tool.input_schema);
    if (parameters != null) {
      attributes[SemanticConventions.TOOL_PARAMETERS] = parameters;
    }
  }
  return attributes;
}

/**
 * Runs each step of a tool runner in the context of its span, so the LLM span
 * of every request and the TOOL span of every tool it runs are children of it,
 * and ends the span when the run ends:
 *
 * - run to the end (`for await`, `runUntilDone()` or `await runner`): status
 *   OK, with the final message as the output;
 * - left early (`break`): with the status unset;
 * - failed: status ERROR, with the error recorded.
 *
 * Every way of running the runner iterates it, so wrapping its async iterator
 * covers them all.
 */
function instrumentToolRunner({
  runner,
  span,
  runContext,
}: {
  runner: ToolRunner;
  span: Span;
  runContext: ReturnType<typeof context.active>;
}) {
  let spanEnded = false;
  const endSpan = () => {
    if (!spanEnded) {
      spanEnded = true;
      span.end();
    }
  };
  const onError = (error: unknown) => {
    if (!spanEnded) {
      recordSpanError(span, error);
      endSpan();
    }
  };
  const onDone = async () => {
    if (spanEnded) {
      return;
    }
    const message = await runner.done().catch(() => undefined);
    if (message != null) {
      span.setAttributes(
        safelyGetAttributes(() => ({
          ...getAnthropicOutputMessagesAttributes(message),
          ...getOutputAttributes(toJSONOutput(message)),
        })),
      );
    }
    span.setStatus({ code: SpanStatusCode.OK });
    endSpan();
  };

  const iterate = runner[Symbol.asyncIterator].bind(runner);
  type Item = Awaited<ReturnType<ReturnType<typeof iterate>["next"]>>;
  Object.defineProperty(runner, Symbol.asyncIterator, {
    configurable: true,
    writable: true,
    value: () => {
      const iterator = context.with(runContext, iterate);
      const tracedIterator = {
        async next(): Promise<Item> {
          let result: Item;
          try {
            result = await context.with(runContext, () => iterator.next());
          } catch (error) {
            onError(error);
            throw error;
          }
          if (result.done) {
            await onDone();
          }
          return result;
        },
        async return(value?: unknown): Promise<Item> {
          try {
            const returned = await context.with(runContext, () => iterator.return?.(value));
            const result: Item = returned ?? { done: true, value: undefined };
            endSpan();
            return result;
          } catch (error) {
            onError(error);
            throw error;
          }
        },
        [Symbol.asyncIterator]() {
          return tracedIterator;
        },
      };
      return tracedIterator;
    },
  });
}
