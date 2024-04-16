/**
 * Utility function that uses the type system to check if a switch statement is exhaustive.
 * If the switch statement is not exhaustive, there will be a type error caught in typescript
 *
 * See https://stackoverflow.com/questions/39419170/how-do-i-check-that-a-switch-block-is-exhaustive-in-typescript for more details.
 */
export function assertUnreachable(_: never): never {
  throw new Error("Unreachable");
}

/**
 * A type-guard function for checking if a value is an object
 */
export function isObject(
  value: unknown,
): value is Record<string | number | symbol, unknown> {
  return typeof value === "object" && value != null && !Array.isArray(value);
}

export function isString(value: unknown): value is string {
  return typeof value === "string";
}

export function isNumber(value: unknown): value is number {
  return typeof value === "number";
}

export function isNonEmptyArray(value: unknown): value is unknown[] {
  return value != null && Array.isArray(value) && value.length > 0;
}
