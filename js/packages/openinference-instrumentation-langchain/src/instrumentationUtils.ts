import { OITracer } from "@arizeai/openinference-core";

import { LangChainTracer } from "./tracer";

import type * as CallbackManagerModuleV02 from "@langchain/core/callbacks/manager";

/**
 * Adds the {@link LangChainTracer} to the callback handlers if it is not already present
 * @param tracer the {@link tracer} to pass into the {@link LangChainTracer} when added to handlers
 * @param handlers the LangChain callback handlers which may be an array of handlers or a CallbackManager
 * @returns the callback handlers with the {@link LangChainTracer} added
 *
 * If the handlers are an array, we add the tracer to the array if it is not already present
 */
export function addTracerToHandlers(
  tracer: OITracer,
  handlers?: CallbackManagerModuleV02.Callbacks,
): CallbackManagerModuleV02.Callbacks;
export function addTracerToHandlers(
  tracer: OITracer,
  handlers?: CallbackManagerModuleV02.Callbacks,
): CallbackManagerModuleV02.Callbacks {
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
