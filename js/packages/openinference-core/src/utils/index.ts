import { GenericFunction, SafeFunction } from "./types";
export * from "./typeUtils";

/**
 * Wraps a function with a try-catch block to catch and log any errors.
 * @param fn - A function to wrap with a try-catch block.
 * @returns A function that returns null if an error is thrown.
 */
export function withSafety<T extends GenericFunction>({
  fn,
  onError,
}: {
  fn: T;
  onError?: (error: unknown) => void;
}): SafeFunction<T> {
  return (...args) => {
    try {
      return fn(...args);
    } catch (error) {
      if (onError) {
        onError(error);
      }
      return null;
    }
  };
}

export const safelyJSONStringify = withSafety({ fn: JSON.stringify });

export const safelyJSONParse = withSafety({ fn: JSON.parse });
