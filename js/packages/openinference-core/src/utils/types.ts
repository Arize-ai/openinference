/** Any function, used as a constraint for function wrappers. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type GenericFunction = (...args: any[]) => any;

/** A function with the same parameters as `T` that returns null instead of throwing. */
export type SafeFunction<T extends GenericFunction> = (
  ...args: Parameters<T>
) => ReturnType<T> | null;
