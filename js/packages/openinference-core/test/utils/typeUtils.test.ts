import { describe, it, expect } from "vitest";
import { isPromise } from "../../src/utils/typeUtils";

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
