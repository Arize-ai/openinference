import type * as CallbackManagerModule from "@langchain/core/callbacks/manager";
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
    const module = new InstrumentationNodeModuleDefinition<
      typeof CallbackManagerModule
    >(
      "@langchain/core/dist/callbacks/manager.cjs",
      ["^0.0.0", "^0.1.0"],
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
    this.tracer;

    this._wrap(module.CallbackManager, "configure", (original) => {
      return (...args: Parameters<typeof original>) => {
        const inheritableHandlers = args[0];
        const newInheritableHandlers = addTracerToHandlers(
          this.tracer,
          inheritableHandlers,
        );
        args[0] = newInheritableHandlers;
        return original.apply(this, args);
      };
    });
    module.openInferencePatched = true;
    return module;
  }

  private unpatch(
    module?: typeof CallbackManagerModule & {
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
    delete module.openInferencePatched;
    return module;
  }
}

function addTracerToHandlers(
  tracer: Tracer,
  handlers?: CallbackManagerModule.Callbacks,
) {
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
  handlers.addHandler(new LangChainTracer(tracer), true);
  return handlers;
}
