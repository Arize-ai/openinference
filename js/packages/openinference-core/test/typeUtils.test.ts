import { isAttributeValue } from "../src/typeUtils";

describe("isAttributeValue", () => {
  it("should return true for string", () => {
    expect(isAttributeValue("hello")).toBe(true);
  });
  it("should return true for number", () => {
    expect(isAttributeValue(123)).toBe(true);
  });
  it("should return true for boolean", () => {
    expect(isAttributeValue(true)).toBe(true);
  });
  it("should return true for sparse numeric array", () => {
    expect(isAttributeValue([1, null, undefined])).toBe(true);
  });
  it("should return true for sparse boolean array", () => {
    expect(isAttributeValue([true, null, undefined])).toBe(true);
  });
  it("should return true for sparse string array", () => {
    expect(isAttributeValue(["hello", null, undefined])).toBe(true);
  });
  it("should return false for an object", () => {
    expect(isAttributeValue({})).toBe(false);
  });
  it("should return false for an array of objects", () => {
    expect(isAttributeValue(["test", {}])).toBe(false);
  });
});
