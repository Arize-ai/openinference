import { Span } from "@opentelemetry/api";

import {
  DefaultTraceConfig,
  REDACTED_VALUE,
} from "../../src/trace/trace-config/constants";
import { OISpan } from "../../src/trace/trace-config/OISpan";

import { beforeEach, describe, expect, it, type Mocked, vi } from "vitest";
describe("OISpan", () => {
  describe("OISpan", () => {
    let mockSpan: Mocked<Span>;

    beforeEach(() => {
      mockSpan = {
        setAttribute: vi.fn().mockReturnThis(),
        setAttributes: vi.fn().mockReturnThis(),
        spanContext: vi.fn(),
        addEvent: vi.fn().mockReturnThis(),
        addLink: vi.fn().mockReturnThis(),
        addLinks: vi.fn().mockReturnThis(),
        end: vi.fn().mockReturnThis(),
        isRecording: vi.fn().mockReturnThis(),
        recordException: vi.fn().mockReturnThis(),
        updateName: vi.fn().mockReturnThis(),
        setStatus: vi.fn().mockReturnThis(),
      };
    });
    it("should delegate all methods to the span", () => {
      const openInferenceSpan = new OISpan({
        span: mockSpan,
        config: DefaultTraceConfig,
      });
      openInferenceSpan.setAttribute("key", "value");
      expect(mockSpan.setAttribute).toHaveBeenCalledWith("key", "value");
      openInferenceSpan.setAttributes({ key: "value" });
      expect(mockSpan.setAttributes).toHaveBeenCalledWith({ key: "value" });
      openInferenceSpan.addEvent("name");
      expect(mockSpan.addEvent).toHaveBeenCalledWith(
        "name",
        undefined,
        undefined,
      );
      openInferenceSpan.addLink({
        context: { spanId: "spanId", traceId: "traceId", traceFlags: 1 },
      });
      expect(mockSpan.addLink).toHaveBeenCalledWith({
        context: { spanId: "spanId", traceId: "traceId", traceFlags: 1 },
      });
      openInferenceSpan.addLinks([
        { context: { spanId: "spanId", traceId: "traceId", traceFlags: 1 } },
      ]);
      expect(mockSpan.addLinks).toHaveBeenCalledWith([
        { context: { spanId: "spanId", traceId: "traceId", traceFlags: 1 } },
      ]);
      openInferenceSpan.end();
      expect(mockSpan.end).toHaveBeenCalled();
      openInferenceSpan.isRecording();
      expect(mockSpan.isRecording).toHaveBeenCalled();
      openInferenceSpan.recordException(new Error());
      expect(mockSpan.recordException).toHaveBeenCalledWith(
        new Error(),
        undefined,
      );
      openInferenceSpan.updateName("name");
      expect(mockSpan.updateName).toHaveBeenCalledWith("name");
      openInferenceSpan.setStatus({ code: 1 });
      expect(mockSpan.setStatus).toHaveBeenCalledWith({ code: 1 });
      openInferenceSpan.spanContext();
      expect(mockSpan.spanContext).toHaveBeenCalled();
    });

    describe("setAttribute", () => {
      it("should mask sensitive attributes", () => {
        const openInferenceSpan = new OISpan({
          span: mockSpan,
          config: { ...DefaultTraceConfig, hideInputs: true },
        });
        openInferenceSpan.setAttribute("input.value", "sensitiveValue");
        expect(mockSpan.setAttribute).toHaveBeenCalledWith(
          "input.value",
          REDACTED_VALUE,
        );
      });

      it("should not mask non-sensitive attributes", () => {
        const openInferenceSpan = new OISpan({
          span: mockSpan,
          config: { ...DefaultTraceConfig, hideInputs: true },
        });
        openInferenceSpan.setAttribute("normalKey", "normalValue");
        expect(mockSpan.setAttribute).toHaveBeenCalledWith(
          "normalKey",
          "normalValue",
        );
      });
    });

    describe("setAttributes", () => {
      it("should mask sensitive attributes in bulk", () => {
        const openInferenceSpan = new OISpan({
          span: mockSpan,
          config: { ...DefaultTraceConfig, hideInputs: true },
        });
        openInferenceSpan.setAttributes({
          "input.value": "sensitiveValue",
          normalKey: "normalValue",
        });
        expect(mockSpan.setAttributes).toHaveBeenCalledWith({
          "input.value": REDACTED_VALUE,
          normalKey: "normalValue",
        });
      });
    });
  });
});
