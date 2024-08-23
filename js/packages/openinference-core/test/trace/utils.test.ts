import { withSafety } from "../../src";
import { diag } from "@opentelemetry/api";

describe("withSafety", () => {
  afterEach(() => {
    jest.clearAllMocks();
    jest.resetModules();
    jest.restoreAllMocks();
  });
  it("should return a function", () => {
    const safeFunction = withSafety(() => {});
    expect(typeof safeFunction).toBe("function");
  });

  it("should execute the provided function without errors", () => {
    const mockFn = jest.fn();
    const safeFunction = withSafety(mockFn);
    safeFunction();
    expect(mockFn).toHaveBeenCalled();
  });

  it("should return null", () => {
    const error = new Error("Test error");
    const mockFn = jest.fn((_a: number) => {
      throw error;
    });
    const diagMock = jest.spyOn(diag, "error");
    const safeFunction = withSafety(mockFn);
    const result = safeFunction(1);
    expect(result).toBeNull();
    expect(mockFn).toHaveBeenCalledWith(1);
    expect(diagMock).toHaveBeenCalledTimes(0);
  });

  it("should log a message and the error when one is passed in", () => {
    const error = new Error("Test error");
    const mockFn = jest.fn((_a: number) => {
      throw error;
    });
    const diagMock = jest.spyOn(diag, "error");
    const safeFunction = withSafety(mockFn, "Test message");
    const result = safeFunction(1);
    expect(result).toBeNull();
    expect(mockFn).toHaveBeenCalledWith(1);
    expect(diagMock).toHaveBeenCalledWith("Test message Error: Test error");
  });
});
