import type * as CallbackManagerModuleV02 from "@langchain/core/callbacks/manager";
import type * as CallbackManagerModuleV01 from "@langchain/coreV0.1/callbacks/manager";
import { Tracer } from "@opentelemetry/api";
import { LangChainTracer } from "./tracer";

/**
 * Adds the {@link LangChainTracer} to the callback handlers if it is not already present
 * @param tracer the {@link tracer} to pass into the {@link LangChainTracer} when added to handlers
 * @param handlers the LangChain callback handlers which may be an array of handlers or a CallbackManager
 * @returns the callback handlers with the {@link LangChainTracer} added
 *
 * If the handlers are an array, we add the tracer to the array if it is not already present
 *
 * There are some slight differences in the CallbackHandler interface between V0.1 and v0.2
 * So we have to cast our tracer to any to avoid type errors
 * We support both versions and our tracer is compatible with either as it will extend the BaseTracer from the installed version which will be the same as the version of handlers passed in here
 */
export function addTracerToHandlers(
  tracer: Tracer,
  handlers?: CallbackManagerModuleV01.Callbacks,
): CallbackManagerModuleV01.Callbacks;
export function addTracerToHandlers(
  tracer: Tracer,
  handlers?: CallbackManagerModuleV02.Callbacks,
): CallbackManagerModuleV02.Callbacks;
export function addTracerToHandlers(
  tracer: Tracer,
  handlers?:
    | CallbackManagerModuleV01.Callbacks
    | CallbackManagerModuleV02.Callbacks,
): CallbackManagerModuleV01.Callbacks | CallbackManagerModuleV02.Callbacks {
  if (handlers == null) {
    return [new LangChainTracer(tracer)];
  }
  if (Array.isArray(handlers)) {
    const tracerAlreadyRegistered = handlers.some(
      (handler) => handler instanceof LangChainTracer,
    );
    if (!tracerAlreadyRegistered) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      handlers.push(new LangChainTracer(tracer) as any);
    }
    return handlers;
  }
  const tracerAlreadyRegistered =
    handlers.inheritableHandlers.some(
      (handler) => handler instanceof LangChainTracer,
    ) ||
    handlers.handlers.some((handler) => handler instanceof LangChainTracer);
  if (tracerAlreadyRegistered) {
    return handlers;
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  handlers.addHandler(new LangChainTracer(tracer) as any, true);
  return handlers;
}
