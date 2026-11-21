import type { UnknownRecord } from "../../types";
import { isRecord } from "../../utils";

/** Matches a non-negative integer index, e.g. the `0` in `llm.tools.0.tool.name`. */
const DIGITS_PATTERN = /^\d+$/;

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
