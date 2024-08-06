import { AttributeValue } from "@opentelemetry/api";

function isSparseNumericArray(
  value: unknown,
): value is (number | null | undefined)[] {
  return (
    Array.isArray(value) &&
    value.every((v) => v == null || typeof v === "number")
  );
}

function isSparseBooleanArray(
  value: unknown,
): value is (boolean | null | undefined)[] {
  return (
    Array.isArray(value) &&
    value.every((v) => v == null || typeof v === "boolean")
  );
}

function isSparseStringArray(
  value: unknown,
): value is (string | null | undefined)[] {
  return (
    Array.isArray(value) &&
    value.every((v) => v == null || typeof v === "string")
  );
}

export function isAttributeValue(value: unknown): value is AttributeValue {
  return (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean" ||
    isSparseNumericArray(value) ||
    isSparseBooleanArray(value) ||
    isSparseStringArray(value)
  );
}
