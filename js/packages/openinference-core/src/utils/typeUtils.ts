import { AttributeValue } from "@opentelemetry/api";

/**
 * Type guard to determine whether or not a value is an array of nullable numbers.
 * @param value
 * @returns true if the value is an array of nullable numbers, false otherwise.
 */
function isSparseNumericArray(
  value: unknown,
): value is (number | null | undefined)[] {
  return (
    Array.isArray(value) &&
    value.every((v) => v == null || typeof v === "number")
  );
}

/**
 * Type guard to determine whether or not a value is an array of nullable booleans.
 * @param value
 * @returns true if the value is an array of nullable booleans, false otherwise.
 */
function isSparseBooleanArray(
  value: unknown,
): value is (boolean | null | undefined)[] {
  return (
    Array.isArray(value) &&
    value.every((v) => v == null || typeof v === "boolean")
  );
}

/**
 * Type guard to determine whether or not a value is an array of nullable strings.
 * @param value
 * @returns true if the value is an array of nullable strings, false otherwise.
 */
function isSparseStringArray(
  value: unknown,
): value is (string | null | undefined)[] {
  return (
    Array.isArray(value) &&
    value.every((v) => v == null || typeof v === "string")
  );
}

/**
 * Type guard to determine whether or not a value is an attribute value (i.e., a primitive value or an array of strings, numbers or booleans).
 * @param value
 * @returns true if the value is an attribute value, false otherwise.
 */
export function isAttributeValue(value: unknown): value is AttributeValue {
  return (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean" ||
    isSparseNumericArray(value) ||
    isSparseBooleanArray(value) ||
    isSparseStringArray(value)
  );
}

/**
 * Type guard to determine whether or not a value is an array of strings.
 * @param value
 * @returns true if the value is an array of strings, false otherwise.
 */
export function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((v) => typeof v === "string");
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
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value) &&
    Object.keys(value).every((key) => typeof key === "string")
  );
}
