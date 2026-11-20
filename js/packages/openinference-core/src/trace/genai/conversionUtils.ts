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

export type UnknownRecord = Record<string, unknown>;

const DIGITS_PATTERN = /^\d+$/;
const INTEGER_PATTERN = /^[+-]?\d+$/;

/**
 * Parses a string as JSON, returning the original value when it is not a
 * string or is not valid JSON.
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

export function parseJSONRecord(value: unknown): UnknownRecord {
  const parsedValue = parseJSONIfString(value);
  return isRecord(parsedValue) ? parsedValue : {};
}

export function parseJSONArray(value: unknown): unknown[] {
  const parsedValue = parseJSONIfString(value);
  return isArray(parsedValue) ? parsedValue : [];
}

export function stringifyJSON(value: unknown): string | undefined {
  return safelyJSONStringify(value) ?? undefined;
}

/**
 * Converts a value to a string, preserving empty strings.
 * Returns undefined for null and undefined.
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
 * Groups flattened, indexed attributes into one record per index, ordered by
 * index.
 *
 * @example
 * ```typescript
 * getIndexedBuckets({
 *   attributes: { "llm.tools.0.tool.name": "search", "llm.tools.1.tool.name": "fetch" },
 *   prefix: "llm.tools",
 * });
 * // [{ "tool.name": "search" }, { "tool.name": "fetch" }]
 * ```
 *
 * @param params.attributes - The flattened attributes to group
 * @param params.prefix - The attribute prefix that precedes the index
 */
export function getIndexedBuckets({
  attributes,
  prefix,
}: {
  attributes: UnknownRecord;
  prefix: string;
}): UnknownRecord[] {
  const keyPrefix = `${prefix}.`;
  const bucketsByIndex = new Map<number, UnknownRecord>();
  for (const [key, value] of Object.entries(attributes)) {
    if (!key.startsWith(keyPrefix)) {
      continue;
    }
    const suffix = key.slice(keyPrefix.length);
    const separatorIndex = suffix.indexOf(".");
    if (separatorIndex < 0) {
      continue;
    }
    const indexText = suffix.slice(0, separatorIndex);
    const remainder = suffix.slice(separatorIndex + 1);
    if (!DIGITS_PATTERN.test(indexText) || remainder === "") {
      continue;
    }
    const bucketIndex = Number(indexText);
    const bucket = bucketsByIndex.get(bucketIndex) ?? {};
    bucket[remainder] = value;
    bucketsByIndex.set(bucketIndex, bucket);
  }
  return [...bucketsByIndex.entries()]
    .sort(([leftIndex], [rightIndex]) => leftIndex - rightIndex)
    .map(([, bucket]) => bucket);
}

/**
 * Rebuilds a nested object from flattened attributes that share a prefix.
 *
 * @param params.attributes - The flattened attributes
 * @param params.prefix - The key prefix (including the trailing dot) to strip
 */
export function unflattenAttributes({
  attributes,
  prefix,
}: {
  attributes: UnknownRecord;
  prefix: string;
}): UnknownRecord {
  // Path segments come from attribute keys, so the containers have no
  // prototype: a `__proto__` segment is then an ordinary key instead of a way
  // to reach and write to `Object.prototype`.
  const unflattened: UnknownRecord = Object.create(null);
  for (const [key, value] of Object.entries(attributes)) {
    if (!key.startsWith(prefix)) {
      continue;
    }
    const path = key.slice(prefix.length).split(".");
    const leafKey = path[path.length - 1] ?? "";
    let current = unflattened;
    for (const segment of path.slice(0, -1)) {
      const existingValue = current[segment];
      const nextValue: UnknownRecord = isRecord(existingValue)
        ? existingValue
        : Object.create(null);
      current[segment] = nextValue;
      current = nextValue;
    }
    current[leafKey] = value;
  }
  return unflattened;
}

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

export function coerceInteger(value: unknown): number | undefined {
  if (isNumber(value)) {
    return Number.isInteger(value) ? value : undefined;
  }
  if (isString(value) && INTEGER_PATTERN.test(value.trim())) {
    return Number(value);
  }
  return undefined;
}

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

export function coerceStringArray(value: unknown): string[] | undefined {
  if (isString(value)) {
    return [value];
  }
  if (isArray(value)) {
    return value.map((item) => toOptionalString(item) ?? String(item));
  }
  return undefined;
}
