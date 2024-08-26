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
import { Tracer, diag } from "@opentelemetry/api";
import { LangChainTracer } from "./tracer";

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

    this._wrap(module.CallbackManager, "configure", (original) => {
      return function <
        T extends
          | (typeof CallbackManagerModuleV1)["CallbackManager"]
          | (typeof CallbackManagerModuleV2)["CallbackManager"],
      >(this: T, ...args: Parameters<T["configure"]>) {
        const inheritableHandlers = args[0];
        const newInheritableHandlers = addTracerToHandlers(
          instrumentation.tracer,
          inheritableHandlers,
        );
        args[0] = newInheritableHandlers;

        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore the types of the callback manager are slightly different between v1 and v2
        // Here, we will only be calling with one or the other so we know they will be compatible and can ignore the error
        return original.apply(this, args);
      };
    });
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

function addTracerToHandlers<
  T extends
    | CallbackManagerModuleV1.Callbacks
    | CallbackManagerModuleV2.Callbacks,
>(tracer: Tracer, handlers?: T) {
  if (handlers == null) {
    return [new LangChainTracer(tracer)];
  }
  if (Array.isArray(handlers)) {
    const newHandlers = handlers;
    const tracerAlreadyRegistered = newHandlers.some(
      (handler) => handler instanceof LangChainTracer,
    );
    if (!tracerAlreadyRegistered) {
      newHandlers.push(new LangChainTracer(tracer));
    }
    return newHandlers;
  }
  const tracerAlreadyRegistered = handlers.inheritableHandlers.some(
    (handler) => handler instanceof LangChainTracer,
  );
  if (tracerAlreadyRegistered) {
    return handlers;
  }
  // There are some slight differences in teh BaseCallbackHandler interface between v1 and v2
  // We support both versions and our tracer is compatible with either
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  handlers.addHandler(new LangChainTracer(tracer) as any, true);
  return handlers;
}
