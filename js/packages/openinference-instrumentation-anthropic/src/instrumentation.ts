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

import type { TraceConfig, TraceConfigOptions } from "@arizeai/openinference-core";
import {
  generateTraceConfig,
  getInputAttributes,
  getMetadata,
  getOutputAttributes,
  getToolAttributes,
  OITracer,
} from "@arizeai/openinference-core";
import {
  LLMProvider,
  LLMSystem,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { getAnthropicOutputMessagesAttributes, isToolUseContentBlock } from "./attributes/content";
import { getAnthropicRequestAttributes } from "./attributes/request";
import {
  getAnthropicResponseAttributes,
  getAnthropicResponseMetadata,
  safelyGetToolCallMetadata,
  getAnthropicMetadataAttributes,
  getAnthropicUsageAttributes,
  getAnthropicReasoningTokenAttributes,
  mergeAnthropicUsage,
} from "./attributes/response";
import { MODULE_NAME } from "./constants";
import type {
  TokensCount,
  ToolRunnerParams,
  ToolRunner,
  RunnableTool,
  Message,
  ContentBlock,
  RawMessageStreamEvent,
  MessageUsageCounts,
  MetadataSources,
} from "./types";
import { safelyGetAttributes, toJSONOutput } from "./utils";
// oxlint-disable-next-line typescript/prefer-ts-expect-error
// @ts-ignore - No version file until build
import { VERSION } from "./version";

const INSTRUMENTATION_NAME = "@arizeai/openinference-instrumentation-anthropic";

/**
 * The name of the span of a `beta.messages.toolRunner()` run. The Python
 * instrumentation has no tool runner span yet; this follows the naming
 * proposed for it in PR #3531 (`beta.messages.tool_runner`).
 */
const TOOL_RUNNER_SPAN_NAME = "beta.messages.tool_runner";

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
  /** The trace config with its defaults and environment variables applied */
  private resolvedTraceConfig: TraceConfig = generateTraceConfig({});
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
    this.resolvedTraceConfig = generateTraceConfig(traceConfig ?? {});
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
          const metadataSources: MetadataSources = {
            context: getMetadata(context.active()),
            requestToolCalls: safelyGetToolCallMetadata(body),
          };
          const span = instrumentation.oiTracer.startSpan(`Anthropic Messages`, {
            kind: SpanKind.INTERNAL,
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
              [SemanticConventions.LLM_SYSTEM]: LLMSystem.ANTHROPIC,
              [SemanticConventions.LLM_PROVIDER]: getAnthropicProvider(this),
              ...safelyGetAttributes(() =>
                getAnthropicRequestAttributes(body, instrumentation.resolvedTraceConfig),
              ),
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
                    getAnthropicResponseAttributes(
                      result,
                      metadataSources,
                      instrumentation.resolvedTraceConfig,
                    ),
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
                  metadataSources,
                  traceConfig: instrumentation.resolvedTraceConfig,
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
    const patchCountTokens = <CountType extends CountTokensType>(
      original: CountType,
    ): CountType => {
      const patched = function patchedCountTokens(this: unknown, ...args: Parameters<CountType>) {
        const body = args[0];
        const span = instrumentation.oiTracer.startSpan(`Anthropic Count Tokens`, {
          kind: SpanKind.INTERNAL,
          attributes: {
            [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
            [SemanticConventions.LLM_SYSTEM]: LLMSystem.ANTHROPIC,
            [SemanticConventions.LLM_PROVIDER]: getAnthropicProvider(this),
            ...safelyGetAttributes(() =>
              getAnthropicRequestAttributes(body, instrumentation.resolvedTraceConfig),
            ),
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
      return isPatchOf(original, patched) ? patched : original;
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
    const patchToolRunner = (original: ToolRunnerType): ToolRunnerType => {
      const patched = function patchedToolRunner(
        this: unknown,
        ...args: Parameters<ToolRunnerType>
      ) {
        const [params, ...rest] = args;
        const tools = new RunnableToolTracer(instrumentation.oiTracer);
        const runner: ReturnType<ToolRunnerType> = Reflect.apply(original, this, [
          tools.traceParams(params),
          ...rest,
        ]);
        const provider = getAnthropicProvider(this);
        try {
          instrumentToolRunnerTools({ runner, tools });
        } catch (error) {
          diag.warn(`Failed to instrument a ${MODULE_NAME} tool runner's tools`, error);
        }
        // Instrumented on its own, so the run keeps its CHAIN span even when
        // the runner's tool methods could not be wrapped.
        try {
          instrumentToolRunner({
            runner,
            traceConfig: instrumentation.resolvedTraceConfig,
            // Started for each run, with the params as they are then.
            startRunSpan: () =>
              instrumentation.oiTracer.startSpan(TOOL_RUNNER_SPAN_NAME, {
                kind: SpanKind.INTERNAL,
                attributes: {
                  [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
                  [SemanticConventions.LLM_SYSTEM]: LLMSystem.ANTHROPIC,
                  [SemanticConventions.LLM_PROVIDER]: provider,
                  ...safelyGetAttributes(() => getInputAttributes(toJSONOutput(runner.params))),
                },
              }),
          });
        } catch (error) {
          diag.warn(`Failed to instrument a ${MODULE_NAME} tool runner`, error);
        }
        return runner;
      };
      return isPatchOf(original, patched) ? patched : original;
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
 * Narrows a patched method to the type of the SDK method it replaces. A patch
 * takes the same arguments and returns what the original returns (through
 * `traceAnthropicAPIPromise` or the original call itself), which TypeScript
 * cannot check against the SDK's overloads.
 */
function isPatchOf<Method extends (...args: never[]) => unknown>(
  _original: Method,
  patched: unknown,
): patched is Method {
  return typeof patched === "function";
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
  metadataSources,
  traceConfig,
}: {
  stream: Stream<RawMessageStreamEvent>;
  span: Span;
  endSpan: () => void;
  recordError: (error: unknown) => void;
  metadataSources: MetadataSources;
  traceConfig: TraceConfig;
}) {
  const state = createAnthropicStreamState();
  const recordStreamAttributes = () => {
    span.setAttributes(
      safelyGetAttributes(() => getAnthropicStreamAttributes(state, metadataSources, traceConfig)),
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

  // Only the first iterator is observed: the response can be read once, so a
  // later one only gets the SDK's "consumed stream" error, which belongs to
  // that caller and must not end the span while the first is still reading.
  let observing = false;
  const observeFirst = (iterator: AsyncIterator<RawMessageStreamEvent>) => {
    if (observing) {
      return iterator;
    }
    observing = true;
    return wrapAnthropicStreamIterator({ iterator, observer });
  };

  const iteratorFactory: unknown = Reflect.get(stream, "iterator");
  if (typeof iteratorFactory === "function") {
    Reflect.set(stream, "iterator", () =>
      observeFirst(getAsyncIterator(Reflect.apply(iteratorFactory, stream, []))),
    );
    return;
  }
  // Fall back to the async iterator protocol should the SDK stop building its
  // iterators from `iterator`.
  const asyncIterator = stream[Symbol.asyncIterator].bind(stream);
  Object.defineProperty(stream, Symbol.asyncIterator, {
    configurable: true,
    writable: true,
    value: () => observeFirst(asyncIterator()),
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
  /**
   * The raw JSON chunks streamed so far for the input of each tool call not
   * yet stopped, by block index. They are joined and parsed once, when the
   * block stops or the stream ends, as parsing on every delta is quadratic.
   */
  partialInputs: Map<number, string[]>;
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
  } else if (delta.type === "signature_delta" && block.type === "thinking") {
    // As the SDK's message stream, only a thinking block takes a signature.
    block.signature = delta.signature;
  } else if (delta.type === "compaction_delta" && block.type === "compaction") {
    applyAnthropicCompactionDelta(block, delta);
  } else if (delta.type === "input_json_delta" && isToolUseContentBlock(block)) {
    const chunks = state.partialInputs.get(chunk.index);
    if (chunks == null) {
      state.partialInputs.set(chunk.index, [delta.partial_json]);
    } else {
      chunks.push(delta.partial_json);
    }
  }
}

/**
 * Sets the input of each tool call whose JSON is still being accumulated,
 * or just the one of `index`. Input that does not parse, as when the caller
 * left the stream early, is kept as the raw JSON received.
 */
function settleAnthropicToolInputs(state: AnthropicStreamState, index?: number) {
  for (const [blockIndex, chunks] of state.partialInputs) {
    if (index != null && blockIndex !== index) {
      continue;
    }
    const block = state.content[blockIndex];
    if (block != null) {
      // No input JSON is an empty input, as in the SDK's own accumulation.
      const json = chunks.join("");
      Object.assign(block, { input: json === "" ? {} : parseToolInput(json) });
    }
    state.partialInputs.delete(blockIndex);
  }
}

function applyAnthropicCompactionDelta(
  block: Extract<ContentBlock, { type: "compaction" }>,
  delta: Anthropic.Beta.Messages.BetaCompactionContentBlockDelta,
) {
  // As the SDK's message stream: the delta carries the block's final value
  // (null content is a failed compaction), so it replaces rather than appends,
  // and encrypted_content is copied only when the server sent it.
  block.content = delta.content;
  if ("encrypted_content" in delta) {
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
  // On a server-side fallback stream the serving hop's token counts replace
  // message_start's, which belong to the attempt that declined. Only the
  // counts are taken from the iteration, not its `type` and `model`, and the
  // other usage fields (service tier, cache creation...) are kept.
  const servingCounts =
    state.servingUsage != null
      ? withoutNullValues({
          input_tokens: state.servingUsage.input_tokens,
          output_tokens: state.servingUsage.output_tokens,
          cache_creation_input_tokens: state.servingUsage.cache_creation_input_tokens,
          cache_read_input_tokens: state.servingUsage.cache_read_input_tokens,
        })
      : {};
  const startUsage = { ...state.message?.usage, ...servingCounts };
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
  metadataSources: MetadataSources,
  traceConfig: TraceConfig,
): Attributes {
  settleAnthropicToolInputs(state);
  const content = state.content.filter((block) => block != null);
  const message = getAnthropicStreamMessage(state);
  const attributes: Attributes = {
    ...getAnthropicOutputMessagesAttributes(
      { role: state.message?.role ?? "assistant", content },
      traceConfig,
    ),
    ...getAnthropicMetadataAttributes({
      metadata: getAnthropicResponseMetadata(
        { ...message, content },
        traceConfig,
        metadataSources.requestToolCalls,
      ),
      contextMetadata: metadataSources.context,
    }),
  };
  if (message.usage != null && typeof message.usage === "object") {
    Object.assign(attributes, getAnthropicReasoningTokenAttributes(message.usage));
  }

  Object.assign(attributes, getOutputAttributes(toJSONOutput(message)));

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
    const { content, ...message } = chunk.message;
    state.message = message;
    // A message can start with content and a stop reason already in it: the
    // stream resuming a programmatic tool call sends its tool_use there, with
    // no block or message_delta events. The SDK's MessageStream keeps them, so
    // they seed the message here too.
    if (Array.isArray(content)) {
      content.forEach((block, index) => {
        if (block != null && typeof block === "object") {
          state.content[index] = { ...block };
        }
      });
    }
    if (chunk.message.stop_reason != null) {
      state.finishReason = chunk.message.stop_reason;
    }
    state.responseModel = chunk.message.model;
    state.startUsage = chunk.message.usage ?? {};
  } else if (chunk.type === "message_delta") {
    applyAnthropicMessageDelta({ chunk, state });
  } else if (chunk.type === "content_block_start") {
    applyAnthropicContentBlockStart({ chunk, state });
  } else if (chunk.type === "content_block_delta") {
    applyAnthropicContentBlockDelta({ chunk, state });
  } else if (chunk.type === "content_block_stop") {
    settleAnthropicToolInputs(state, chunk.index);
  }
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
 * Traces the runnable tools of one tool runner, while keeping the caller's own
 * tool objects everywhere the caller can see them.
 *
 * The runner calls `tool.run()` from module-private code, on the tools of its
 * private params, so the only place to trace a run from is the tool object
 * itself. Each runnable tool is therefore handed to the runner as a wrapper that
 * traces `run`, and the runner's public surface maps between the two:
 * `params` and the `setMessagesParams()` mutator show the caller's tools, and
 * the tools passed to `setMessagesParams()` and `addTools()` are traced too.
 */
class RunnableToolTracer {
  private readonly traced = new WeakMap<object, RunnableTool>();
  private readonly originals = new WeakMap<object, RunnableTool>();

  constructor(private readonly oiTracer: OITracer) {}

  /** The traced wrapper of a runnable tool, the same one each time */
  trace<Tool extends ToolRunnerParams["tools"][number]>(tool: Tool): Tool | RunnableTool {
    if (!isRunnableTool(tool) || this.originals.has(tool)) {
      return tool;
    }
    let wrapper = this.traced.get(tool);
    if (wrapper == null) {
      wrapper = traceRunnableTool({ tool, oiTracer: this.oiTracer });
      this.traced.set(tool, wrapper);
      this.originals.set(wrapper, tool);
    }
    return wrapper;
  }

  /** The caller's tool behind a traced wrapper */
  original<Tool extends ToolRunnerParams["tools"][number]>(tool: Tool): Tool | RunnableTool {
    return (tool != null && typeof tool === "object" && this.originals.get(tool)) || tool;
  }

  traceParams(params: ToolRunnerParams): ToolRunnerParams {
    try {
      if (!Array.isArray(params?.tools)) {
        return params;
      }
      return { ...params, tools: params.tools.map((tool) => this.trace(tool)) };
    } catch (error) {
      diag.warn(`Failed to trace the ${MODULE_NAME} tool runner's tools`, error);
      return params;
    }
  }

  originalParams(params: ToolRunnerParams): ToolRunnerParams {
    if (!Array.isArray(params?.tools)) {
      return params;
    }
    return { ...params, tools: params.tools.map((tool) => this.original(tool)) };
  }
}

/**
 * Wraps a runnable tool in an object whose `run` is a TOOL span, a child of the
 * span active when the runner calls it. Every other property is read from the
 * tool itself, with its methods bound to it, so a class tool keeps working with
 * private (`#`) fields, and a frozen tool can be wrapped too.
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
  // A plain object rather than a Proxy: a Proxy must return a frozen tool's
  // own `run` and methods unchanged, so it cannot trace a frozen tool.
  const wrapper: RunnableTool = Object.create(Object.getPrototypeOf(tool));
  const boundMethods = new Map<unknown, unknown>();
  for (const [property, descriptor] of getToolProperties(tool)) {
    Object.defineProperty(wrapper, property, {
      configurable: true,
      enumerable: descriptor.enumerable ?? false,
      get: () => {
        const value: unknown = Reflect.get(tool, property, tool);
        if (typeof value !== "function") {
          return value;
        }
        let bound = boundMethods.get(value);
        if (bound == null) {
          bound = value.bind(tool);
          boundMethods.set(value, bound);
        }
        return bound;
      },
    });
  }
  Object.defineProperty(wrapper, "run", {
    configurable: true,
    enumerable: Object.getOwnPropertyDescriptor(tool, "run")?.enumerable ?? false,
    writable: true,
    value: tracedRun,
  });
  return wrapper;
}

/**
 * The properties of a tool, its own and those of its classes, each with the
 * descriptor seen first, as `tool.property` resolves them.
 */
function getToolProperties(tool: object): Map<PropertyKey, PropertyDescriptor> {
  const properties = new Map<PropertyKey, PropertyDescriptor>();
  let current: object | null = tool;
  while (current != null && current !== Object.prototype) {
    for (const property of Reflect.ownKeys(current)) {
      const descriptor = Object.getOwnPropertyDescriptor(current, property);
      if (property !== "constructor" && descriptor != null && !properties.has(property)) {
        properties.set(property, descriptor);
      }
    }
    current = Object.getPrototypeOf(current);
  }
  return properties;
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
  const description =
    "description" in tool && typeof tool.description === "string" && tool.description
      ? tool.description
      : undefined;
  const inputSchema: unknown = "input_schema" in tool ? tool.input_schema : undefined;
  // A tool without a schema of its own (e.g. a typed client tool) has no
  // parameters to record, which the core helper always records.
  const toolAttributes: Attributes = isStringRecord(inputSchema)
    ? getToolAttributes({ name: tool.name, description, parameters: inputSchema })
    : {
        [SemanticConventions.TOOL_NAME]: tool.name,
        ...(description != null && { [SemanticConventions.TOOL_DESCRIPTION]: description }),
      };
  return { ...toolAttributes, ...getInputAttributes(toJSONOutput(args)) };
}

function isStringRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Maps the runner's public view of its params back to the caller's tools, and
 * traces the tools given to it later, see {@link RunnableToolTracer}.
 */
function instrumentToolRunnerTools({
  runner,
  tools,
}: {
  runner: ToolRunner;
  tools: RunnableToolTracer;
}) {
  const prototype: unknown = Object.getPrototypeOf(runner);
  if (prototype == null || typeof prototype !== "object") {
    return;
  }
  Object.defineProperty(runner, "params", {
    configurable: true,
    get: () => {
      const params: ToolRunnerParams = Reflect.get(prototype, "params", runner);
      return tools.originalParams(params);
    },
  });

  // setMessagesParams() and addTools() are wrapped only where the SDK has
  // them: SDKs before 0.133 have no addTools().
  if (typeof runner.setMessagesParams === "function") {
    instrumentSetMessagesParams({ runner, tools });
  }
  if (typeof runner.addTools === "function") {
    const addTools = runner.addTools.bind(runner);
    Object.defineProperty(runner, "addTools", {
      configurable: true,
      writable: true,
      value: (...added: Parameters<ToolRunner["addTools"]>) =>
        addTools(...added.map((tool) => tools.trace(tool))),
    });
  }
}

/**
 * Traces the tools of the params given to `setMessagesParams()`, and hands its
 * mutator the caller's tools.
 */
function instrumentSetMessagesParams({
  runner,
  tools,
}: {
  runner: ToolRunner;
  tools: RunnableToolTracer;
}) {
  const setMessagesParams = runner.setMessagesParams.bind(runner);
  Object.defineProperty(runner, "setMessagesParams", {
    configurable: true,
    writable: true,
    value: (
      paramsOrMutator: ToolRunnerParams | ((prevParams: ToolRunnerParams) => ToolRunnerParams),
    ) => {
      if (typeof paramsOrMutator === "function") {
        setMessagesParams((prevParams) =>
          tools.traceParams(paramsOrMutator(tools.originalParams(prevParams))),
        );
      } else {
        setMessagesParams(tools.traceParams(paramsOrMutator));
      }
    },
  });
}

/**
 * Runs each run of a tool runner under a CHAIN span of its own, so the LLM
 * span of every request and the TOOL span of every tool it runs are children
 * of it, and ends the span when the run ends:
 *
 * - run to the end (`for await`, `runUntilDone()` or `await runner`): status
 *   OK, with the final message as the output;
 * - left early (`break`): with the status unset;
 * - failed: status ERROR, with the error recorded.
 *
 * Every way of running the runner iterates it, so wrapping its async iterator
 * covers them all, and a run retried after a failure gets a new span.
 */
function instrumentToolRunner({
  runner,
  startRunSpan,
  traceConfig,
}: {
  runner: ToolRunner;
  startRunSpan: () => Span;
  traceConfig: TraceConfig;
}) {
  const iterate = runner[Symbol.asyncIterator].bind(runner);
  type Item = Awaited<ReturnType<ReturnType<typeof iterate>["next"]>>;
  Object.defineProperty(runner, Symbol.asyncIterator, {
    configurable: true,
    writable: true,
    value: () => {
      const span = startRunSpan();
      const runContext = trace.setSpan(getExecContext(span), span);
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
              ...getAnthropicOutputMessagesAttributes(message, traceConfig),
              ...getOutputAttributes(toJSONOutput(message)),
            })),
          );
        }
        span.setStatus({ code: SpanStatusCode.OK });
        endSpan();
      };

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
