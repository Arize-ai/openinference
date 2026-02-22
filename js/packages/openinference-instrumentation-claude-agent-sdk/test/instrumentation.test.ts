import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { ClaudeAgentSDKInstrumentation, _resetPatchState } from "../src/instrumentation";

describe("ClaudeAgentSDKInstrumentation", () => {
  let instrumentation: ClaudeAgentSDKInstrumentation;
  let provider: NodeTracerProvider;
  let exporter: InMemorySpanExporter;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    instrumentation = new ClaudeAgentSDKInstrumentation({
      tracerProvider: provider,
    });
  });

  afterEach(() => {
    if (instrumentation.isEnabled()) {
      instrumentation.disable();
    }
    _resetPatchState();
    exporter.reset();
  });

  it("should be enabled by default", () => {
    expect(instrumentation.isEnabled()).toBe(true);
  });

  it("should be able to enable and disable", () => {
    expect(instrumentation.isEnabled()).toBe(true);

    instrumentation.disable();
    expect(instrumentation.isEnabled()).toBe(false);

    instrumentation.enable();
    expect(instrumentation.isEnabled()).toBe(true);
  });

  it("should initialize without errors", () => {
    expect(() => {
      new ClaudeAgentSDKInstrumentation();
    }).not.toThrow();
  });

  it("should accept custom tracer provider", () => {
    const customProvider = new NodeTracerProvider();
    const customInstrumentation = new ClaudeAgentSDKInstrumentation({
      tracerProvider: customProvider,
    });

    expect(customInstrumentation.isEnabled()).toBe(true);
    customInstrumentation.disable();
  });

  it("should accept trace config options", () => {
    const customInstrumentation = new ClaudeAgentSDKInstrumentation({
      traceConfig: {
        hideInputs: true,
        hideOutputs: true,
      },
    });

    expect(customInstrumentation.isEnabled()).toBe(true);
    customInstrumentation.disable();
  });

  it("should manually instrument a mock module", () => {
    const mockModule = {
      query: function* () {
        yield { type: "result", subtype: "success" };
      },
    };

    expect(() => {
      instrumentation.manuallyInstrument(mockModule);
    }).not.toThrow();
  });

  it("should not double-patch the module", () => {
    const callCount = { value: 0 };
    const mockModule = {
      query: function () {
        callCount.value++;
        return {
          [Symbol.asyncIterator]() {
            return {
              async next() {
                return { done: true, value: undefined };
              },
            };
          },
        };
      },
    };

    instrumentation.manuallyInstrument(mockModule);
    const firstQuery = mockModule.query;

    instrumentation.manuallyInstrument(mockModule);
    const secondQuery = mockModule.query;

    // Should not have double-wrapped
    expect(firstQuery).toBe(secondQuery);
  });
});
