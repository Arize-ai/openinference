import { describe, expect, it } from "vitest";

import {
  isArray,
  isBigInt,
  isBoolean,
  isDefined,
  isNullish,
  isNumber,
  isPresent,
  isPromise,
  isRecord,
  isString,
  isStringArray,
  isUndefined,
} from "../../src/utils/typeUtils";

describe("isPromise", () => {
  it("should correctly identify promises and non-promises", async () => {
    // Should return true for promises
    expect(isPromise(new Promise((resolve) => resolve("test")))).toBe(true);
    expect(isPromise(Promise.resolve("test"))).toBe(true);

    const rejectedPromise = Promise.reject(new Error("test"));
    rejectedPromise.catch(() => {}); // Prevent unhandled rejection
    expect(isPromise(rejectedPromise)).toBe(true);

    const asyncFn = async () => "test";
    const asyncResult = asyncFn();
    expect(isPromise(asyncResult)).toBe(true);
    await asyncResult; // Clean up

    // Should return false for non-promises
    expect(isPromise(null)).toBe(false);
    expect(isPromise(undefined)).toBe(false);
    expect(isPromise("test")).toBe(false);
    expect(isPromise(42)).toBe(false);
    expect(isPromise(true)).toBe(false);
    expect(isPromise([])).toBe(false);
    expect(isPromise({})).toBe(false);
    expect(isPromise(() => "test")).toBe(false);
  });
});

describe("primitive type guards", () => {
  it("should identify strings", () => {
    expect(isString("")).toBe(true);
    expect(isString("text")).toBe(true);
    expect(isString(1)).toBe(false);
    expect(isString(null)).toBe(false);
  });

  it("should identify numbers, including non-finite ones", () => {
    expect(isNumber(0)).toBe(true);
    expect(isNumber(Number.NaN)).toBe(true);
    expect(isNumber("1")).toBe(false);
    expect(isNumber(BigInt(1))).toBe(false);
  });

  it("should identify booleans", () => {
    expect(isBoolean(false)).toBe(true);
    expect(isBoolean(0)).toBe(false);
    expect(isBoolean("true")).toBe(false);
  });

  it("should identify bigints", () => {
    expect(isBigInt(BigInt(1))).toBe(true);
    expect(isBigInt(1)).toBe(false);
  });
});

describe("presence type guards", () => {
  it("should treat only undefined as undefined", () => {
    expect(isUndefined(undefined)).toBe(true);
    expect(isUndefined(null)).toBe(false);
    expect(isDefined(undefined)).toBe(false);
    expect(isDefined(null)).toBe(true);
    expect(isDefined(0)).toBe(true);
  });

  it("should treat null and undefined as nullish and everything else as present", () => {
    expect(isNullish(null)).toBe(true);
    expect(isNullish(undefined)).toBe(true);
    expect(isNullish(0)).toBe(false);
    expect(isPresent(null)).toBe(false);
    expect(isPresent(undefined)).toBe(false);
    expect(isPresent(0)).toBe(true);
    expect(isPresent("")).toBe(true);
    expect(isPresent(false)).toBe(true);
  });

  it("should narrow when used as a filter", () => {
    const names: string[] = ["first", undefined, "second"].filter(isDefined);
    expect(names).toEqual(["first", "second"]);
  });
});

describe("structural type guards", () => {
  it("should identify arrays", () => {
    expect(isArray([])).toBe(true);
    expect(isArray([1, "two"])).toBe(true);
    expect(isArray({ length: 0 })).toBe(false);
    expect(isArray("text")).toBe(false);
  });

  it("should identify arrays of strings", () => {
    expect(isStringArray([])).toBe(true);
    expect(isStringArray(["first", "second"])).toBe(true);
    expect(isStringArray(["first", 2])).toBe(false);
    expect(isStringArray("first")).toBe(false);
  });

  it("should identify records", () => {
    expect(isRecord({})).toBe(true);
    expect(isRecord({ key: "value" })).toBe(true);
    expect(isRecord(Object.create(null))).toBe(true);
    expect(isRecord([])).toBe(false);
    expect(isRecord(null)).toBe(false);
    expect(isRecord("text")).toBe(false);
  });
});
