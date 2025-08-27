import { isObjectWithStringKeys } from "@arizeai/openinference-core";
import { StringKeyedObject } from "../types";

/**
 * Checks if unknown data is an array of objects with string keys.
 */
export function isArrayOfObjectWithStringKeys(
  maybeArray: unknown,
): maybeArray is StringKeyedObject[] {
  return Array.isArray(maybeArray) && maybeArray.every(isObjectWithStringKeys);
}
