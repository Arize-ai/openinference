import { AsyncLocalStorage } from "node:async_hooks";

import { ROOT_CONTEXT, type Context, type ContextManager } from "@opentelemetry/api";

/**
 * OpenTelemetry context propagation for workerd. Workers expose Node's `AsyncLocalStorage`
 * under the `nodejs_compat` flag, which is all a context manager needs: the active context
 * follows `await` through the agent loop, so spans nest without threading a parent through
 * every call. This is the `context-async-hooks` manager without the Node-only imports.
 */
export class AsyncLocalStorageContextManager implements ContextManager {
  #store: AsyncLocalStorage<Context> | undefined = new AsyncLocalStorage<Context>();

  active(): Context {
    return this.#store?.getStore() ?? ROOT_CONTEXT;
  }

  with<A extends unknown[], F extends (...args: A) => ReturnType<F>>(
    context: Context,
    fn: F,
    thisArg?: ThisParameterType<F>,
    ...args: A
  ): ReturnType<F> {
    if (!this.#store) return fn.call(thisArg, ...args);
    return this.#store.run(context, () => fn.call(thisArg, ...args));
  }

  /** Bind a function to context while preserving its receiver and arguments. */
  bind<T>(context: Context, target: T): T {
    if (typeof target !== "function") return target;
    return new Proxy(target, {
      apply: (fn, receiver, args: unknown[]) =>
        this.with(context, () => Reflect.apply(fn, receiver, args)),
    });
  }

  enable(): this {
    this.#store ??= new AsyncLocalStorage<Context>();
    return this;
  }

  disable(): this {
    // workerd deliberately does not implement AsyncLocalStorage.disable().
    this.#store = undefined;
    return this;
  }
}
