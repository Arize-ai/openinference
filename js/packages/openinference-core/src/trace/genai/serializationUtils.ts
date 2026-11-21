import type { UnknownRecord } from "../../types";
import {
  isArray,
  isBigInt,
  isBoolean,
  isNullish,
  isNumber,
  isRecord,
  isString,
  safelyJSONStringify,
} from "../../utils";

/** Matches an optionally signed integer written as a string. */
const INTEGER_PATTERN = /^[+-]?\d+$/;

/**
 * Parses a string as JSON, returning the original value when it is not a
 * string or is not valid JSON.
 * @param value - The value to parse
 * @returns The parsed JSON, or the original value
 */
export function parseJSONIfString(value: unknown): unknown {
  if (!isString(value)) {
    return value;
  }
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}

/**
 * Parses a value that is a record or a JSON encoded record.
 * @param value - A record, or a string holding a JSON object
 * @returns The record, or an empty record when the value is anything else
 */
export function parseJSONRecord(value: unknown): UnknownRecord {
  const parsedValue = parseJSONIfString(value);
  return isRecord(parsedValue) ? parsedValue : {};
}

/**
 * Parses a value that is an array or a JSON encoded array.
 * @param value - An array, or a string holding a JSON array
 * @returns The array, or an empty array when the value is anything else
 */
export function parseJSONArray(value: unknown): unknown[] {
  const parsedValue = parseJSONIfString(value);
  return isArray(parsedValue) ? parsedValue : [];
}

/**
 * JSON encodes a value without throwing.
 * @param value - The value to encode
 * @returns The JSON string, or undefined when the value cannot be encoded
 */
export function stringifyJSON(value: unknown): string | undefined {
  return safelyJSONStringify(value) ?? undefined;
}

/**
 * Converts a value to a string, preserving empty strings.
 * Returns undefined for null and undefined.
 * @param value - The value to convert
 * @returns The string, or undefined for null and undefined
 */
export function toOptionalString(value: unknown): string | undefined {
  if (isNullish(value)) {
    return undefined;
  }
  if (isString(value)) {
    return value;
  }
  if (isNumber(value) || isBoolean(value) || isBigInt(value)) {
    return String(value);
  }
  return stringifyJSON(value);
}

/**
 * Converts a value to a string, returning undefined for null, undefined, and
 * empty strings.
 * @param value - The value to convert
 * @returns The string, or undefined when it would be empty
 */
export function toNonEmptyString(value: unknown): string | undefined {
  return toOptionalString(value) || undefined;
}

/**
 * Serializes a value that may already be a JSON string. A JSON encoded string
 * is unwrapped, any other string is returned as-is, and everything else is
 * JSON encoded.
 *
 * A string that holds JSON is never re-encoded: a parse and stringify round
 * trip is lossy in JavaScript, e.g. for integers above
 * `Number.MAX_SAFE_INTEGER`.
 * @param value - The value to serialize
 * @returns The serialized value, or undefined when it cannot be encoded
 */
export function serializeJSONLikeValue(value: unknown): string | undefined {
  const parsedValue = parseJSONIfString(value);
  if (isString(parsedValue)) {
    return parsedValue;
  }
  return isString(value) ? value : stringifyJSON(parsedValue);
}

/**
 * Whether a value is empty in the sense that it carries no information:
 * null, undefined, false, 0, "", [] or {}.
 * @param value - The value to check
 */
export function isEmptyValue(value: unknown): boolean {
  if (isArray(value)) {
    return value.length === 0;
  }
  if (isRecord(value)) {
    return Object.keys(value).length === 0;
  }
  return !value;
}

/**
 * Reads a finite number from a number or a numeric string.
 * @param value - The value to read
 * @returns The number, or undefined when the value is not a finite number
 */
export function coerceNumber(value: unknown): number | undefined {
  if (isNumber(value)) {
    return Number.isFinite(value) ? value : undefined;
  }
  if (isString(value) && value.trim() !== "") {
    const parsedNumber = Number(value);
    return Number.isFinite(parsedNumber) ? parsedNumber : undefined;
  }
  return undefined;
}

/**
 * Reads an integer from a number or an integer string.
 * @param value - The value to read
 * @returns The integer, or undefined when the value is not an integer
 */
export function coerceInteger(value: unknown): number | undefined {
  if (isNumber(value)) {
    return Number.isInteger(value) ? value : undefined;
  }
  if (isString(value) && INTEGER_PATTERN.test(value.trim())) {
    return Number(value);
  }
  return undefined;
}

/**
 * Reads a boolean from a boolean or the strings `"true"` and `"false"`, ignoring case.
 * @param value - The value to read
 * @returns The boolean, or undefined when the value is not a boolean
 */
export function coerceBoolean(value: unknown): boolean | undefined {
  if (isBoolean(value)) {
    return value;
  }
  if (isString(value)) {
    const loweredValue = value.toLowerCase();
    if (loweredValue === "true") {
      return true;
    }
    if (loweredValue === "false") {
      return false;
    }
  }
  return undefined;
}

/**
 * Reads a list of strings. A single string becomes a one-item list, and
 * non-string items are converted to strings.
 * @param value - A string or an array
 * @returns The strings, or undefined when the value is neither
 */
export function coerceStringArray(value: unknown): string[] | undefined {
  if (isString(value)) {
    return [value];
  }
  if (isArray(value)) {
    return value.map((item) => toOptionalString(item) ?? String(item));
  }
  return undefined;
}
