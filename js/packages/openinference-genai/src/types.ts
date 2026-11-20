export type Mutable<T> = {
  -readonly [P in keyof T]: T[P];
};

export type DeeplyMutable<T> = {
  -readonly [P in keyof T]: T[P] extends object ? DeeplyMutable<T[P]> : T[P];
};

/**
 * A function with the same parameters as `Fn` that returns null instead of throwing.
 */
export type SafeFunction<Fn extends (...args: never[]) => unknown> = (
  ...args: Parameters<Fn>
) => ReturnType<Fn> | null;
