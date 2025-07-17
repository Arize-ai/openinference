import {
  InstrumentationBase,
  InstrumentationNodeModuleDefinition,
  InstrumentationConfig,
} from "@opentelemetry/instrumentation";
import { diag, Tracer, TracerProvider } from "@opentelemetry/api";
import { Version } from "beeai-framework";
import * as bee from "beeai-framework";

import { OITracer, TraceConfigOptions } from "@arizeai/openinference-core";
import { createTelemetryMiddleware } from "./middleware";
import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";
import { satisfies } from "semver";

const MODULE_NAME = "beeai-framework";

const INSTRUMENTATION_NAME = "@arizeai/openinference-instrumentation-beeai";

const INSTRUMENTS = [">=0.1.9 <0.1.14"];

/**
 * Flag to check if the openai module has been patched
 * Note: This is a fallback in case the module is made immutable (e.x. Deno, webpack, etc.)
 */
let _isOpenInferencePatched = false;

/**
 * function to check if instrumentation is enabled / disabled
 */
export function isPatched() {
  return _isOpenInferencePatched;
}

export class BeeAIInstrumentation extends InstrumentationBase {
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
      Version,
      Object.assign({}, instrumentationConfig),
    );
    this.tracerProvider = tracerProvider;
    this.traceConfig = traceConfig;
    this.oiTracer = new OITracer({
      tracer:
        this.tracerProvider?.getTracer(INSTRUMENTATION_NAME) ?? this.tracer,
      traceConfig,
    });
  }

  protected init() {
    const module = new InstrumentationNodeModuleDefinition(
      MODULE_NAME,
      INSTRUMENTS,
      this.patch.bind(this),
      this.unpatch.bind(this),
    );

    return module;
  }

  /**
   * Manually instruments the BeeAI module. This is needed when the module is not loaded via require (commonjs)
   * @param {openai} module
   */
  manuallyInstrument(module: typeof bee) {
    if (
      !INSTRUMENTS.some((instrument) => satisfies(module.Version, instrument))
    ) {
      const supportedVersionsString = `beeai-framework ["${INSTRUMENTS.join(',"')}"]`;
      diag.warn(
        `DependencyConflict: requested: '${supportedVersionsString}' but found: 'beeai-framework "${module.Version}"'`,
      );
      return;
    }

    diag.debug(`[BeeaiInstrumentation] Manually instrumenting ${MODULE_NAME}`);
    this.patch(module);
  }

  get tracer(): Tracer {
    if (this.tracerProvider) {
      return this.tracerProvider.getTracer(this.instrumentationName);
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

  private patch(module: typeof bee & { openInferencePatched?: boolean }) {
    if (!module?.BaseAgent) {
      diag.warn(
        "[BeeaiInstrumentation] BaseAgent not found, skipping instrumentation.",
      );
      return module;
    }

    if (module?.openInferencePatched || _isOpenInferencePatched) {
      return module;
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const createWrappedMethod = <T extends (...args: any[]) => any>(
      instrumentation: BeeAIInstrumentation,
      original: T,
      mainSpanKind: OpenInferenceSpanKind,
    ) => {
      return function wrappedMethod(this: unknown, ...args: unknown[]) {
        const returned = original.apply(this, args);
        if (returned?.middleware) {
          returned.middleware(
            createTelemetryMiddleware(instrumentation.oiTracer, mainSpanKind),
          );
        }
        return returned;
      };
    };
    diag.info("[BeeaiInstrumentation] Patching BaseAgent.run");

    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const instrumentation: BeeAIInstrumentation = this;

    this._wrap(module.BaseAgent.prototype, "run", (original) =>
      createWrappedMethod(
        instrumentation,
        original,
        OpenInferenceSpanKind.AGENT,
      ),
    );

    // tool instrumentation support
    if (module.Tool) {
      this._wrap(module.Tool.prototype, "run", (original) =>
        createWrappedMethod(
          instrumentation,
          original,
          OpenInferenceSpanKind.TOOL,
        ),
      );
    }
    // model instrumentation support
    if (module.ChatModel) {
      this._wrap(module.ChatModel.prototype, "create", (original) =>
        createWrappedMethod(
          instrumentation,
          original,
          OpenInferenceSpanKind.LLM,
        ),
      );
      this._wrap(module.ChatModel.prototype, "createStructure", (original) =>
        createWrappedMethod(
          instrumentation,
          original,
          OpenInferenceSpanKind.LLM,
        ),
      );
    }
    if (module.EmbeddingModel) {
      this._wrap(module.EmbeddingModel.prototype, "create", (original) =>
        createWrappedMethod(
          instrumentation,
          original,
          OpenInferenceSpanKind.EMBEDDING,
        ),
      );
    }

    return module;
  }

  /**
   * Un-patches the OpenAI module's chat completions API
   */
  private unpatch(
    moduleExports: typeof bee & { openInferencePatched?: boolean },
    moduleVersion?: string,
  ) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);
    this._unwrap(moduleExports.BaseAgent.prototype, "run");

    _isOpenInferencePatched = false;
    try {
      // This can fail if the module is made immutable via the runtime or bundler
      moduleExports.openInferencePatched = false;
    } catch (e) {
      diag.warn(`Failed to unset ${MODULE_NAME} patched flag on the module`, e);
    }
  }
}
