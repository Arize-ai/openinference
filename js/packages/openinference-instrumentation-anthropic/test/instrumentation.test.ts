import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import { AnthropicInstrumentation } from "../src/instrumentation";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

describe("AnthropicInstrumentation", () => {
  let instrumentation: AnthropicInstrumentation;
  let provider: NodeTracerProvider;
  let exporter: InMemorySpanExporter;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(exporter));

    instrumentation = new AnthropicInstrumentation({
      tracerProvider: provider,
    });
  });

  afterEach(() => {
    if (instrumentation.isEnabled()) {
      instrumentation.disable();
    }
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
      new AnthropicInstrumentation();
    }).not.toThrow();
  });

  it("should accept custom tracer provider", () => {
    const customProvider = new NodeTracerProvider();
    const customInstrumentation = new AnthropicInstrumentation({
      tracerProvider: customProvider,
    });

    expect(customInstrumentation.isEnabled()).toBe(true);
    customInstrumentation.disable();
  });

  it("should accept trace config options", () => {
    const customInstrumentation = new AnthropicInstrumentation({
      traceConfig: {
        hideInputs: true,
        hideOutputs: true,
      },
    });

    expect(customInstrumentation.isEnabled()).toBe(true);
    customInstrumentation.disable();
  });
});
