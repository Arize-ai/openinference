/**
 * Utility functions for safe and loose JSON parsing.
 */

import { isObjectWithStringKeys } from "@arizeai/openinference-core";
import { StringKeyedObject } from "../types";
import { isArrayOfObjectWithStringKeys } from "./typeUtils";

/**
 * Safely parse JSON string, returns the parsed json as unknown.
 * If the parsing fails, attempts to sanitize the json and parse again.
 * If the sanitization fails, returns undefined.
 *
 * @param {string} text - The JSON string to parse
 * @returns {unknown} The parsed JSON object, or undefined if parsing fails
 * @example
 * ```typescript
 * const result = parseSanitizedJson('{"key": "value"}');
 * // Returns: { key: "value" }
 *
 * const badJson = parseSanitizedJson('{"key": "value with \ backslash"}');
 * // Attempts to sanitize and parse, may return parsed object or undefined
 * ```
 */
export function parseSanitizedJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    try {
      return JSON.parse(sanitizeJsonInput(text));
    } catch {
      return undefined;
    }
  }
}

/**
 * Cleans a JSON string by escaping invalid backslashes.
 * This function fixes common JSON parsing issues by properly escaping backslashes
 * that are not part of valid JSON escape sequences.
 *
 * @param {string} badJsonStr - The potentially malformed JSON string to sanitize
 * @returns {string} The sanitized JSON string with properly escaped backslashes
 * @internal
 * @example
 * ```typescript
 * const malformed = '{"path": "C:\folder\file.txt"}';
 * const sanitized = sanitizeJsonInput(malformed);
 * // Returns: '{"path": "C:\\folder\\file.txt"}'
 * ```
 */
function sanitizeJsonInput(badJsonStr: string): string {
  // Regex: match backslashes not followed by valid escape chars
  return badJsonStr.replace(/\\(?!["\\/bfnrtu])/g, "\\\\");
}

/**
 * Attempts to fix and parse loose JSON strings. Returns array of parsed objects or original string.
 * Tries to extract objects from loose JSON, similar to Python's fix_loose_json_string.
 *
 * This function handles various malformed JSON formats by:
 * 1. First attempting to parse as valid JSON
 * 2. Extracting object-like substrings using regex
 * 3. Fixing common formatting issues (unquoted keys, single quotes, etc.)
 * 4. Returning successfully parsed objects or the original content
 *
 * @param {string} content - The potentially malformed JSON string to fix and parse
 * @returns {StringKeyedObject[] | [string]} Array of parsed objects, or array containing original string if parsing fails
 * @example
 * ```typescript
 * // Valid JSON array
 * fixLooseJsonString('[{"key": "value"}]');
 * // Returns: [{ key: "value" }]
 *
 * // Loose format with unquoted keys
 * fixLooseJsonString('{key=value, other="test"}');
 * // Returns: [{ key: "value", other: "test" }]
 *
 * // Multiple objects in string
 * fixLooseJsonString('text {key1=val1} more text {key2=val2}');
 * // Returns: [{ key1: "val1" }, { key2: "val2" }]
 *
 * // Unparseable content
 * fixLooseJsonString('just plain text');
 * // Returns: ["just plain text"]
 * ```
 *
 * Note: This upon review with @Parker-Stafford and @mikeldking does seem to fail
 * in places, notably when there are commas in the string.
 */
export function fixLooseJsonString(
  content: string,
): StringKeyedObject[] | [string] {
  if (!content) return [];
  const trimmed = content.trim();

  // If it's a valid JSON array or object, parse directly
  const parsed = parseSanitizedJson(trimmed);
  if (isArrayOfObjectWithStringKeys(parsed)) {
    return parsed;
  }
  if (isObjectWithStringKeys(parsed)) {
    return [parsed];
  }

  // Try to extract object-like substrings using regex
  const objStrings = trimmed.match(/\{[\s\S]*?}/g) || [];
  const fixedObjects: StringKeyedObject[] = [];
  for (const objStr of objStrings) {
    let objFixed = objStr.replace(/(\w+)=/g, '"$1":');
    objFixed = objFixed.replace(/:\s*([^"{},[]]+)/g, ': "$1"');
    objFixed = objFixed.replace(/'/g, '"');
    const parsedObj = parseSanitizedJson(objFixed);
    if (isObjectWithStringKeys(parsedObj)) {
      fixedObjects.push(parsedObj);
    }
  }
  // If we found objects, return them; else, return the original string
  return fixedObjects.length > 0 ? fixedObjects : [content];
}

/**
 * Attempts to safely extract an object from unknown data using a specified key.
 * This utility function provides type-safe access to nested object properties
 * when the data structure is unknown at compile time.
 *
 * @param {Object} params - The parameter object
 * @param {unknown} params.data - The data object to extract from (must be an object with string keys)
 * @param {string} params.key - The property key to extract the object from
 * @returns {StringKeyedObject | null} The extracted object if it exists and is valid, otherwise null
 * @example
 * ```typescript
 * const unknownData: unknown = {
 *   user: { name: "John", age: 30 },
 *   settings: { theme: "dark" }
 * };
 *
 * const userObj = getObjectDataFromUnknown({ data: unknownData, key: "user" });
 * // Returns: { name: "John", age: 30 }
 *
 * const invalidObj = getObjectDataFromUnknown({ data: "not an object", key: "user" });
 * // Returns: null
 *
 * const missingObj = getObjectDataFromUnknown({ data: unknownData, key: "nonexistent" });
 * // Returns: null
 * ```
 */
export function getObjectDataFromUnknown({
  data,
  key,
}: {
  data: unknown;
  key: string;
}): StringKeyedObject | null {
  if (!isObjectWithStringKeys(data)) {
    return null;
  }
  const maybeData = data[key] ?? null;
  if (isObjectWithStringKeys(maybeData)) {
    return maybeData;
  }
  return null;
}
