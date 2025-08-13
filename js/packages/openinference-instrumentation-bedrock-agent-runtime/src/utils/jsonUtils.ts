/**
 * Utility functions for safe and loose JSON parsing.
 */

import { isObjectWithStringKeys } from "@arizeai/openinference-core";
import { StringKeyedObject } from "../types";
import { isArrayOfObjectWithStringKeys } from "./typeUtils";

/**
 * Safely parse JSON string, returns the parsed json as unknown.
 * If the parsing fails, attempts to sanitize the json and parse again.
 * If the sanitization fails, returns undefined
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
 */
function sanitizeJsonInput(badJsonStr: string): string {
  // Regex: match backslashes not followed by valid escape chars
  return badJsonStr.replace(/\\(?!["\\/bfnrtu])/g, "\\\\");
}

/**
 * Attempts to fix and parse loose JSON strings. Returns array of parsed objects or original string.
 * Tries to extract objects from loose JSON, similar to Python's fix_loose_json_string.
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
 * Attempts to get an object from some unknown data using a key.
 * @param data - The data object to get the object from.
 * @param key - The key to get the object from.
 * @returns The object if it exists and is an object with string keys, otherwise null.
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
