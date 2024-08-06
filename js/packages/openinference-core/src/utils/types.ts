// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type GenericFunction = (...args: any[]) => any;

export type SafeFunction<T extends GenericFunction> = (
  ...args: Parameters<T>
) => ReturnType<T> | null;
