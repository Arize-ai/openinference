/**
 * Utility function that uses the type system to check if a switch statement is exhaustive.
 * If the switch statement is not exhaustive, there will be a type error caught in typescript
 *
 * See https://stackoverflow.com/questions/39419170/how-do-i-check-that-a-switch-block-is-exhaustive-in-typescript for more details.
 */
export function assertUnreachable(_: never): never {
  throw new Error("Unreachable");
}

export function isString(value: unknown): value is string {
  return typeof value === "string";
}

/**
 * Read a numeric property that the installed OpenAI SDK types may not declare yet
 * (for example `prompt_tokens_details.cache_write_tokens`). Returns undefined when
 * the object is missing or the property is not a number.
 */
export function getNumberProperty(obj: object | null | undefined, key: string): number | undefined {
  if (obj == null) return undefined;
  const value: unknown = Reflect.get(obj, key);
  return typeof value === "number" ? value : undefined;
}
