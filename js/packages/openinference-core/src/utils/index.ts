import type { SafeFunction } from "../types";
export * from "./typeUtils";

/**
 * Wraps a function with a try-catch block to catch and log any errors.
 * @param fn - A function to wrap with a try-catch block.
 * @returns A function that returns null if an error is thrown.
 */
export function withSafety<Fn extends (...args: never[]) => unknown>({
  fn,
  onError,
}: {
  fn: Fn;
  onError?: (error: unknown) => void;
}): SafeFunction<Fn> {
  return (...args) => {
    try {
      // Calling through the constraint yields `unknown`; the result is `ReturnType<Fn>` by definition.
      // oxlint-disable-next-line typescript/no-unsafe-type-assertion
      return fn(...args) as ReturnType<Fn>;
    } catch (error) {
      if (onError) {
        onError(error);
      }
      return null;
    }
  };
}

/** `JSON.parse`, typed to return `unknown` instead of `any`. */
const unknownJSONParse = (...args: Parameters<typeof JSON.parse>): unknown => {
  return JSON.parse(...args);
};

/**
 * `JSON.stringify` that returns null instead of throwing, e.g. for circular
 * structures or bigints.
 */
export const safelyJSONStringify = withSafety({ fn: JSON.stringify });

/** `JSON.parse` that returns null instead of throwing on invalid JSON. */
export const safelyJSONParse = withSafety({ fn: unknownJSONParse });
