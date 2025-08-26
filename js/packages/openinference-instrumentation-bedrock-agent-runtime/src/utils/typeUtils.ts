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

/**
 * Type guard to check if a value is an array of StringKeyedObject.
 * @param value The value to check.
 * @returns True if the value is an array of StringKeyedObject, false otherwise.
 */
export function isStringKeyedObjectArray(
  value: unknown,
): value is StringKeyedObject[] {
  return (
    Array.isArray(value) &&
    value.every(
      (item) =>
        typeof item === "object" && item !== null && !Array.isArray(item),
    )
  );
}
