import { GenericFunction, SafeFunction } from "./types";
import { diag } from "@opentelemetry/api";
export * from "./typeUtils";

/**
 * Wraps a function with a try-catch block to catch and log any errors.
 * @param fn - A function to wrap with a try-catch block.
 * @returns A function that returns null if an error is thrown.
 */
export function withSafety<T extends GenericFunction>(
  fn: T,
  fallbackMessage?: string,
): SafeFunction<T> {
  return (...args) => {
    try {
      return fn(...args);
    } catch (error) {
      if (fallbackMessage) {
        diag.error(`${fallbackMessage} ${error}`);
      }
      return null;
    }
  };
}

export const safelyJSONStringify = withSafety(JSON.stringify);

export const safelyJSONParse = withSafety(JSON.parse);
