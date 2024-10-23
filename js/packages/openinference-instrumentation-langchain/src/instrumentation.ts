import type * as CallbackManagerModuleV02 from "@langchain/core/callbacks/manager";
import type * as CallbackManagerModuleV01 from "@langchain/coreV0.1/callbacks/manager";
import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
  isWrapped,
} from "@opentelemetry/instrumentation";
import { VERSION } from "./version";
import { diag } from "@opentelemetry/api";
import { addTracerToHandlers } from "./instrumentationUtils";
import { OITracer, TraceConfigOptions } from "@arizeai/openinference-core";

const MODULE_NAME = "@langchain/core/callbacks";

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

type CallbackManagerModule =
  | typeof CallbackManagerModuleV01
  | typeof CallbackManagerModuleV02;

/**
 * An auto instrumentation class for LangChain that creates {@link https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md|OpenInference} Compliant spans for LangChain
 * @param instrumentationConfig The config for the instrumentation @see {@link InstrumentationConfig}
 * @param traceConfig The OpenInference trace configuration. Can be used to mask or redact sensitive information on spans. @see {@link TraceConfigOptions}
 */
export class LangChainInstrumentation extends InstrumentationBase<CallbackManagerModule> {
  private oiTracer: OITracer;

  constructor({
    instrumentationConfig,
    traceConfig,
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
  } = {}) {
    super(
      "@arizeai/openinference-instrumentation-langchain",
      VERSION,
      Object.assign({}, instrumentationConfig),
    );
    this.oiTracer = new OITracer({
      tracer: this.tracer,
      traceConfig,
    });
  }

  manuallyInstrument(module: CallbackManagerModule) {
    diag.debug(`Manually instrumenting ${MODULE_NAME}`);
    this.patch(module);
  }

  protected init(): InstrumentationModuleDefinition<CallbackManagerModule> {
    const module =
      new InstrumentationNodeModuleDefinition<CallbackManagerModule>(
        "@langchain/core/dist/callbacks/manager.cjs",
        ["^0.1.0", "^0.2.0"],
        this.patch.bind(this),
        this.unpatch.bind(this),
      );
    return module;
  }

  private patch(
    module: CallbackManagerModule & {
      openInferencePatched?: boolean;
    },
    moduleVersion?: string,
  ) {
    diag.debug(
      `Applying patch for ${MODULE_NAME}${
        moduleVersion != null ? `@${moduleVersion}` : ""
      }`,
    );
    if (module?.openInferencePatched || _isOpenInferencePatched) {
      return module;
    }
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const instrumentation = this;

    /**
     * _configureSync is only available in v0.2.0 and above
     * It was added as a replacement to the configure method which is marked as soon to be deprecated
     * In v0.2.0 and above, the configure method is a wrapper around _configureSync
     * However, configure is not always called, where as _configureSync is always called
     * so we want to patch only configure sync if it's available
     * and only configure if _configureSync is not available so we don't get duplicate traces
     */
    if ("_configureSync" in module.CallbackManager) {
      this._wrap(module.CallbackManager, "_configureSync", (original) => {
        return function (
          this: typeof CallbackManagerModuleV02,
          ...args: Parameters<
            (typeof CallbackManagerModuleV02.CallbackManager)["_configureSync"]
          >
        ) {
          const inheritableHandlers = args[0];
          const newInheritableHandlers = addTracerToHandlers(
            instrumentation.oiTracer,
            inheritableHandlers,
          );
          args[0] = newInheritableHandlers;

          return original.apply(this, args);
        };
      });
    } else {
      this._wrap(module.CallbackManager, "configure", (original) => {
        return function (
          this: typeof CallbackManagerModuleV01,
          ...args: Parameters<
            (typeof CallbackManagerModuleV01.CallbackManager)["configure"]
          >
        ) {
          const handlers = args[0];
          const newHandlers = addTracerToHandlers(
            instrumentation.oiTracer,
            handlers,
          );
          args[0] = newHandlers;

          return original.apply(this, args);
        };
      });
    }
    _isOpenInferencePatched = true;
    try {
      // This can fail if the module is made immutable via the runtime or bundler
      module.openInferencePatched = true;
    } catch (e) {
      diag.debug(`Failed to set ${MODULE_NAME} patched flag on the module`, e);
    }

    return module;
  }

  private unpatch(
    module?: CallbackManagerModule & {
      openInferencePatched?: boolean;
    },
    moduleVersion?: string,
  ) {
    if (module == null) {
      return;
    }
    diag.debug(
      `Removing patch for ${MODULE_NAME}${
        moduleVersion != null ? `@${moduleVersion}` : ""
      }`,
    );
    if (isWrapped(module.CallbackManager.configure)) {
      this._unwrap(module.CallbackManager, "configure");
    }
    /**
     * _configureSync is only available in v0.2.0 and above
     * Thus we only want to unwrap it if it's available and has been wrapped
     */
    if (
      "_configureSync" in module.CallbackManager &&
      isWrapped(module.CallbackManager._configureSync)
    ) {
      this._unwrap(module.CallbackManager, "_configureSync");
    }
    _isOpenInferencePatched = false;
    try {
      // This can fail if the module is made immutable via the runtime or bundler
      module.openInferencePatched = false;
    } catch (e) {
      diag.warn(`Failed to unset ${MODULE_NAME} patched flag on the module`, e);
    }
    return module;
  }
}
