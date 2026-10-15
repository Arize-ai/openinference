import type Anthropic from "@anthropic-ai/sdk";
import type { Stream } from "@anthropic-ai/sdk/streaming";
import type { Tracer, TracerProvider } from "@opentelemetry/api";
import { context, diag, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import type {
  InstrumentationConfig,
  InstrumentationModuleDefinition,
} from "@opentelemetry/instrumentation";
import {
  InstrumentationBase,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";

import type { TraceConfig, TraceConfigOptions } from "@arizeai/openinference-core";
import {
  generateTraceConfig,
  getInputAttributes,
  getMetadata,
  getOutputAttributes,
  OITracer,
} from "@arizeai/openinference-core";
import {
  LLMSystem,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { traceAnthropicAPIPromise } from "./apiPromise";
import { getAnthropicRequestAttributes } from "./attributes/request";
import { getAnthropicResponseAttributes, safelyGetToolCallMetadata } from "./attributes/response";
import { MODULE_NAME } from "./constants";
import { RunnableToolTracer, instrumentToolRunnerTools } from "./runnableTools";
import {
  resolveAnthropicModule,
  isPatchOf,
  getAnthropicProvider,
  isAnthropicMessageResponse,
  isAnthropicStream,
} from "./sdk";
import { instrumentAnthropicStream } from "./stream";
import { instrumentToolRunner } from "./toolRunner";
import type { TokensCount, Message, RawMessageStreamEvent, MetadataSources } from "./types";
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
