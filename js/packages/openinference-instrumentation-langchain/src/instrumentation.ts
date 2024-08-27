import type * as CallbackManagerModuleV2 from "@langchain/core/callbacks/manager";
import type * as CallbackManagerModuleV1 from "@langchain/coreV1/callbacks/manager";
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
  | typeof CallbackManagerModuleV1
  | typeof CallbackManagerModuleV2;

export class LangChainInstrumentation extends InstrumentationBase<CallbackManagerModule> {
  constructor(config?: InstrumentationConfig) {
    super(
      "@arizeai/openinference-instrumentation-langchain",
      VERSION,
      Object.assign({}, config),
    );
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

    if ("_configureSync" in module.CallbackManager) {
      this._wrap(module.CallbackManager, "_configureSync", (original) => {
        return function (
          this: typeof CallbackManagerModuleV2,
          ...args: Parameters<
            (typeof CallbackManagerModuleV2.CallbackManager)["_configureSync"]
          >
        ) {
          const inheritableHandlers = args[0];
          const newInheritableHandlers = addTracerToHandlers(
            instrumentation.tracer,
            inheritableHandlers,
          );
          args[0] = newInheritableHandlers;

          return original.apply(this, args);
        };
      });
    } else {
      this._wrap(module.CallbackManager, "configure", (original) => {
        return function (
          this: typeof CallbackManagerModuleV1,
          ...args: Parameters<
            (typeof CallbackManagerModuleV1.CallbackManager)["configure"]
          >
        ) {
          const handlers = args[0];
          const newHandlers = addTracerToHandlers(
            instrumentation.tracer,
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
      diag.warn(`Failed to set ${MODULE_NAME} patched flag on the module`, e);
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
