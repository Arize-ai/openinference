export const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((v) => typeof v === "string");

const isObjectWithStringKeys = (
  value: unknown,
): value is Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  return Object.keys(value).every((key) => typeof key === "string");
};

export const isArrayOfObjects = (
  value: unknown,
): value is Record<string, unknown>[] =>
  Array.isArray(value) && value.every(isObjectWithStringKeys);

export const assertUnreachable = (x: never): never => {
  throw new Error("Unexpected value: " + x);
};

export type Mutable<T> = { -readonly [P in keyof T]: T[P] };

export type ValueOf<T> = T[keyof T];
