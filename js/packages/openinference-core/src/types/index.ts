/**
 * Types shared across the package. Types that belong to one feature live next
 * to it, e.g. `trace/trace-config/types.ts`.
 */

/**
 * A function with the same parameters as `Fn` that returns null instead of throwing.
 */
export type SafeFunction<Fn extends (...args: never[]) => unknown> = (
  ...args: Parameters<Fn>
) => ReturnType<Fn> | null;

/** A record whose values have not been narrowed yet, e.g. span attributes or parsed JSON. */
export type UnknownRecord = Record<string, unknown>;
