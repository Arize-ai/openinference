import { Attributes } from "@opentelemetry/api";
import { isAttributeValue } from "@opentelemetry/core";

/**
 * Type guard to determine whether or not a value is an array of strings.
 * @param value
 * @returns true if the value is an array of strings, false otherwise.
 */
export function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((v) => typeof v === "string");
}

/**
 * Type guard to determine whether or not a value is an object.
 * @param value
 * @returns true if the value is an object, false otherwise.
 */
function isObject(
  value: unknown,
): value is Record<string | number | symbol, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Type guard to determine whether or not a value is an object with string keys.
 * @param value
 * @returns true if the value is an object with string keys, false otherwise.
 */
export function isObjectWithStringKeys(
  value: unknown,
): value is Record<string, unknown> {
  return (
    isObject(value) &&
    Object.keys(value).every((key) => typeof key === "string")
  );
}

/**
 * Type guard to determine whether or not a value is an object with string keys and attribute values.
 * @param value
 * @returns true if the value is an object with string keys and attribute values, false otherwise.
 */
export function isAttributes(value: unknown): value is Attributes {
  return (
    isObject(value) &&
    Object.entries(value).every(
      ([key, value]) => isAttributeValue(value) && typeof key === "string",
    )
  );
}
