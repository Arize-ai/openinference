import type { Attributes } from "@opentelemetry/api";
import { isAttributeValue } from "@opentelemetry/core";

import type { UnknownRecord } from "../types";

/**
 * Type guard to determine whether or not a value is a string.
 * @param value
 * @returns true if the value is a string, false otherwise.
 */
export function isString(value: unknown): value is string {
  return typeof value === "string";
}

/**
 * Type guard to determine whether or not a value is a number.
 * `NaN` and the infinities are numbers; use `Number.isFinite` to exclude them.
 * @param value
 * @returns true if the value is a number, false otherwise.
 */
export function isNumber(value: unknown): value is number {
  return typeof value === "number";
}

/**
 * Type guard to determine whether or not a value is a boolean.
 * @param value
 * @returns true if the value is a boolean, false otherwise.
 */
export function isBoolean(value: unknown): value is boolean {
  return typeof value === "boolean";
}

/**
 * Type guard to determine whether or not a value is a bigint.
 * @param value
 * @returns true if the value is a bigint, false otherwise.
 */
export function isBigInt(value: unknown): value is bigint {
  return typeof value === "bigint";
}

/**
 * Type guard to determine whether or not a value is undefined.
 * @param value
 * @returns true if the value is undefined, false otherwise.
 */
export function isUndefined(value: unknown): value is undefined {
  return value === undefined;
}

/**
 * Type guard to determine whether or not a value is anything other than
 * undefined. `null` is defined.
 * @example
 * ```typescript
 * const names = [toName(first), toName(second)].filter(isDefined);
 * ```
 * @param value
 * @returns true if the value is not undefined, false otherwise.
 */
export function isDefined<Value>(value: Value | undefined): value is Value {
  return value !== undefined;
}

/**
 * Type guard to determine whether or not a value is null or undefined.
 * @param value
 * @returns true if the value is null or undefined, false otherwise.
 */
export function isNullish(value: unknown): value is null | undefined {
  return value === null || value === undefined;
}

/**
 * Type guard to determine whether or not a value is present: neither null nor
 * undefined. Falsy values such as `0`, `""` and `false` are present.
 * @param value
 * @returns true if the value is neither null nor undefined, false otherwise.
 */
export function isPresent<Value>(value: Value): value is NonNullable<Value> {
  return !isNullish(value);
}

/**
 * Type guard to determine whether or not a value is an array.
 * Unlike `Array.isArray`, the items narrow to `unknown` instead of `any`.
 * @param value
 * @returns true if the value is an array, false otherwise.
 */
export function isArray(value: unknown): value is unknown[] {
  return Array.isArray(value);
}

/**
 * Type guard to determine whether or not a value is an array of strings.
 * @param value
 * @returns true if the value is an array of strings, false otherwise.
 */
export function isStringArray(value: unknown): value is string[] {
  return isArray(value) && value.every(isString);
}

/**
 * Type guard to determine whether or not a value is a record: an object that
 * is neither null nor an array.
 * @param value
 * @returns true if the value is a record, false otherwise.
 */
export function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !isArray(value);
}

/**
 * Type guard to determine whether or not a value is an object with string keys.
 * @param value
 * @returns true if the value is an object with string keys, false otherwise.
 */
export function isObjectWithStringKeys(value: unknown): value is Record<string, unknown> {
  return isRecord(value) && Object.keys(value).every(isString);
}

/**
 * Type guard for if a function is a Promise
 * @param value
 * @returns true if it is a Promise
 */
export function isPromise<T = unknown>(value: unknown): value is Promise<T> {
  return isRecord(value) && typeof value.then === "function" && typeof value.catch === "function";
}

/**
 * Type guard to determine whether or not a value is an object with string keys and attribute values.
 * @param value
 * @returns true if the value is an object with string keys and attribute values, false otherwise.
 */
export function isAttributes(value: unknown): value is Attributes {
  return (
    isRecord(value) &&
    Object.entries(value).every(([key, value]) => isAttributeValue(value) && isString(key))
  );
}

/**
 * A type check function to ensure that a switch or set of conditionals is exhaustive.
 * Typescript will throw an error if the switch or conditionals are not exhaustive.
 * @example
 *  ```typescript
 * type MyType = "a" | "b";
 * function myFunction(value: MyType) {
 *   switch (value) {
 *     case "a":
 *      return "A";
 *    case "b":
 *      return "B";
 *    default:
 *      assertUnreachable(value);
 *   }
 * }
 * ```
 * @param value - The value that should have been narrowed to `never`
 */
export function assertUnreachable(value: never): never {
  throw new Error(`Unreachable code reached with value: ${String(value)}`);
}
