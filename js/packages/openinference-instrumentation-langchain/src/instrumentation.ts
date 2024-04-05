import * as CallbackManagerModule from "@langchain/core/callbacks/manager";
import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
  isWrapped,
} from "@opentelemetry/instrumentation";
import { VERSION } from "./version";
import { diag } from "@opentelemetry/api";
import { LangChainTracer } from "./tracer";

const MODULE_NAME = "@langchain/core";

export class LangChainInstrumentation extends InstrumentationBase<
  typeof CallbackManagerModule
> {
  constructor(config?: InstrumentationConfig) {
    super(
      "@arizeai/openinference-instrumentation-langchain",
      VERSION,
      Object.assign({}, config),
    );
  }

  protected init(): InstrumentationModuleDefinition<
    typeof CallbackManagerModule
  > {
    const module = new InstrumentationNodeModuleDefinition(
      "@langchain/core/callbacks/manager",
      ["^0.1.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );

    return module;
  }

  private patch(
    module: typeof CallbackManagerModule & {
      openInferencePatched?: boolean;
    },
    moduleVersion?: string,
  ) {
    diag.debug(
      `Applying patch for ${MODULE_NAME}${
        moduleVersion != null ? `@${moduleVersion}` : ""
      }`,
    );
    if (module?.openInferencePatched) {
      return module;
    }

    this._wrap(module.CallbackManager, "configure", (original) => {
      return (...args: Parameters<typeof original>) => {
        const inheritableHandlers = args[0];
        const newInheritableHandlers =
          this._addTracerToHandlers(inheritableHandlers);
        args[0] = newInheritableHandlers;
        const manager = original.apply(this, args);
        return manager;
      };
    });
    module.openInferencePatched = true;
    return module;
  }

  private unpatch(
    module?: typeof CallbackManagerModule & { openInferencePatched?: boolean },
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
    delete module.openInferencePatched;
    return module;
  }

  private _addTracerToHandlers(handlers?: CallbackManagerModule.Callbacks) {
    if (handlers == null) {
      return [new LangChainTracer(this.tracer)];
    }
    if (handlers instanceof CallbackManagerModule.CallbackManager) {
      if (
        handlers.inheritableHandlers.some(
          (handler) => handler instanceof LangChainTracer,
        )
      ) {
        return handlers;
      }
      handlers.addHandler(new LangChainTracer(this.tracer), true);
      return handlers;
    }
    const newHandlers = handlers ?? [];
    if (!newHandlers.some((handler) => handler instanceof LangChainTracer)) {
      newHandlers.push(new LangChainTracer(this.tracer));
    }
    return newHandlers;
  }
}
