import { MimeType } from "@arizeai/openinference-semantic-conventions";
import type { Attributes, AttributeValue } from "@opentelemetry/api";

export const safelyJSONStringify = (value: unknown) => {
  try {
    return JSON.stringify(value);
  } catch {
    return null;
  }
};

export const getNumber = (value: unknown): number | undefined => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return undefined;
};

export const getString = (value: unknown): string | undefined => {
  if (typeof value === "string" && value.length > 0) return value;
  return undefined;
};

export const getStringArray = (value: unknown): string[] | undefined => {
  if (Array.isArray(value) && value.every((v) => typeof v === "string")) {
    return value as string[];
  }
  return undefined;
};

export const parseJSON = <T = unknown>(value: unknown): T | undefined => {
  const s = getString(value);
  if (!s) return undefined;
  try {
    return JSON.parse(s) as T;
  } catch {
    return undefined;
  }
};

export const getMimeType = (value: unknown): MimeType => {
  if (parseJSON(value)) return MimeType.JSON;
  return MimeType.TEXT;
};

export const set = (
  attrs: Attributes,
  key: string,
  value?: AttributeValue | null,
) => {
  if (value === undefined || value === null) return;
  attrs[key] = value;
};

export const toStringContent = (value: unknown): string => {
  if (typeof value === "string") return value;
  const json = safelyJSONStringify(value);
  if (typeof json === "string") return json;
  return String(value);
};
