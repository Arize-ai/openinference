import type * as CallbackManagerModuleV2 from "@langchain/core/callbacks/manager";
import type * as CallbackManagerModuleV1 from "@langchain/coreV1/callbacks/manager";
import { Tracer } from "@opentelemetry/api";
import { LangChainTracer } from "./tracer";

export function addTracerToHandlers(
  tracer: Tracer,
  handlers?: CallbackManagerModuleV1.Callbacks,
): CallbackManagerModuleV1.Callbacks;
export function addTracerToHandlers(
  tracer: Tracer,
  handlers?: CallbackManagerModuleV2.Callbacks,
): CallbackManagerModuleV2.Callbacks;
export function addTracerToHandlers(
  tracer: Tracer,
  handlers?:
    | CallbackManagerModuleV1.Callbacks
    | CallbackManagerModuleV2.Callbacks,
): CallbackManagerModuleV1.Callbacks | CallbackManagerModuleV2.Callbacks {
  if (handlers == null) {
    // There are some slight differences in the Callbacks interface between v1 and v2
    return [new LangChainTracer(tracer)];
  }
  if (Array.isArray(handlers)) {
    const tracerAlreadyRegistered = handlers.some(
      (handler) => handler instanceof LangChainTracer,
    );
    if (!tracerAlreadyRegistered) {
      // There are some slight differences in the CallbackHandler interface between v1 and v2
      // We support both versions and our tracer is compatible with either as it will extend the class from the installed version
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      handlers.push(new LangChainTracer(tracer) as any);
    }
    return handlers;
  }
  const tracerAlreadyRegistered = handlers.inheritableHandlers.some(
    (handler) => handler instanceof LangChainTracer,
  );
  if (tracerAlreadyRegistered) {
    return handlers;
  }
  // There are some slight differences in the CallbackHandler interface between v1 and v2
  // We support both versions and our tracer is compatible with either as it will extend the class from the installed version
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  handlers.addHandler(new LangChainTracer(tracer) as any, true);
  return handlers;
}
