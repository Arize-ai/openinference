/**
 * Utility functions for safe and loose JSON parsing.
 */

import { ParsedInput } from "../attributes/types";

/**
 * Safely parse JSON string, returns an object or undefined on error.
 */
export function safeJsonParse(text: string): ParsedInput {
  try {
    return JSON.parse(text);
  } catch {
    try {
      return JSON.parse(sanitizeJsonInput(text));
    } catch {
      return {};
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
export function fixLooseJsonString(content: string): ParsedInput[] | [string] {
  if (!content) return [];
  const trimmed = content.trim();
  // If it's a valid JSON array or object, parse directly
  const parsed = safeJsonParse(trimmed);
  if (parsed && Array.isArray(parsed)) return parsed as ParsedInput[];
  if (parsed && typeof parsed === "object") return [parsed as ParsedInput];

  // Try to extract object-like substrings using regex
  const objStrings = trimmed.match(/\{[\s\S]*?}/g) || [];
  const fixedObjects: ParsedInput[] = [];
  for (const objStr of objStrings) {
    let objFixed = objStr.replace(/(\w+)=/g, '"$1":');
    objFixed = objFixed.replace(/:\s*([^"{},[]]+)/g, ': "$1"');
    objFixed = objFixed.replace(/'/g, '"');
    const parsedObj = safeJsonParse(objFixed);
    if (parsedObj && typeof parsedObj === "object") {
      fixedObjects.push(parsedObj);
    }
  }
  // If we found objects, return them; else, return the original string
  return fixedObjects.length > 0 ? fixedObjects : [content];
}
