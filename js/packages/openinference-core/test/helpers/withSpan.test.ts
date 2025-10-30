import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";

import { SpanKind, trace } from "@opentelemetry/api";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";

import { traceAgent, traceChain, traceTool, withSpan } from "../../src/helpers";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

let spanExporter: InMemorySpanExporter;
let tracerProvider: NodeTracerProvider;

describe("withSpan", () => {
  beforeEach(() => {
    // Set up in-memory span exporter and tracer provider
    spanExporter = new InMemorySpanExporter();
    tracerProvider = new NodeTracerProvider({
      resource: resourceFromAttributes({
        "service.name": "test-service",
      }),
      spanProcessors: [new SimpleSpanProcessor(spanExporter)],
    });

    tracerProvider.register();
  });

  afterEach(() => {
    // Clean up after each test
    spanExporter.reset();
    tracerProvider.shutdown();
  });

  it("should wrap synchronous functions and create spans", () => {
    const testFn = (...args: unknown[]) =>
      (args[0] as number) + (args[1] as number);

    // Use the tracer from our test provider
    const tracer = tracerProvider.getTracer("test");
    const wrappedFn = withSpan(testFn, {
      name: "add-numbers",
      tracer,
    });

    const result = wrappedFn(2, 3);

    expect(result).toBe(5);

    const spans = spanExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.name).toBe("add-numbers");
    expect(span.kind).toBe(SpanKind.INTERNAL);
    expect(span.attributes["openinference.span.kind"]).toBe(
      OpenInferenceSpanKind.CHAIN,
    );
    expect(span.status.code).toBe(1); // OK
  });

  it("should handle async functions and promises", async () => {
    const asyncFn = async (...args: unknown[]) => {
      await new Promise((resolve) => setTimeout(resolve, 10));
      return `processed: ${args[0] as string}`;
    };

    const tracer = tracerProvider.getTracer("test");
    const wrappedFn = withSpan(asyncFn, {
      name: "async-process",
      tracer,
    });
    const result = await wrappedFn("test");

    expect(result).toBe("processed: test");

    const spans = spanExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.name).toBe("async-process");
    expect(span.status.code).toBe(1); // OK
    expect(span.attributes["output.value"]).toBe("processed: test");
  });

  it("should handle promise rejections and record exceptions", async () => {
    const errorFn = async () => {
      throw new Error("Test error");
    };

    const tracer = tracerProvider.getTracer("test");
    const wrappedFn = withSpan(errorFn, {
      name: "error-function",
      tracer,
    });

    await expect(wrappedFn()).rejects.toThrow("Test error");

    const spans = spanExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.name).toBe("error-function");
    expect(span.status.code).toBe(2); // ERROR
    expect(span.status.message).toBe("Test error");
    expect(span.events).toHaveLength(1);
    expect(span.events[0].name).toBe("exception");
  });

  it("should use base attributes when provided", () => {
    const testFn = () => "result";
    const baseAttributes = {
      "service.name": "test-service",
      "service.version": "1.0.0",
    };

    const tracer = tracerProvider.getTracer("test");
    const wrappedFn = withSpan(testFn, {
      name: "test-function",
      attributes: baseAttributes,
      tracer,
    });

    wrappedFn();

    const spans = spanExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.name).toBe("test-function");
    expect(span.attributes["service.name"]).toBe("test-service");
    expect(span.attributes["service.version"]).toBe("1.0.0");
  });

  it("should use custom input and output processors", () => {
    const testFn = (query: string) => {
      return query.length;
    };

    const tracer = tracerProvider.getTracer("test");
    const wrappedFn = withSpan(testFn, {
      processInput: () => ({ "custom.input": "processed" }),
      processOutput: () => ({ "custom.output": "processed" }),
      tracer,
    });

    const result = wrappedFn("12345");

    expect(result).toEqual(5);

    const spans = spanExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.attributes["custom.input"]).toBe("processed");
    expect(span.attributes["custom.output"]).toBe("processed");
  });

  it("should use function name as span name when no name provided", () => {
    function namedFunction() {
      return "test";
    }

    const tracer = tracerProvider.getTracer("test");
    const wrappedFn = withSpan(namedFunction, { tracer });
    wrappedFn();

    const spans = spanExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.name).toBe("namedFunction");
  });

  it("should provide a meaningful function name for arrow functions", () => {
    const arrowFunc = () => "hello";
    const wrappedFn = withSpan(arrowFunc, {
      tracer: tracerProvider.getTracer("test"),
    });
    wrappedFn();
  });
  it("should set the span as active so as to allow for adding attributes to the span", () => {
    const testFn = () => {
      const span = trace.getActiveSpan();
      if (span) {
        span.setAttribute("test-attribute", "test-value");
      }
      return "result";
    };
    const tracer = tracerProvider.getTracer("test");
    const wrappedFn = withSpan(testFn, { tracer });

    wrappedFn();

    const spans = spanExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.attributes["test-attribute"]).toBe("test-value");
  });
  it.skip("should handle generator functions", async () => {
    // TODO(mikeldking): it might be the case that generators are common in genAI applications
    function* generatorFunction() {
      yield "result";
    }
    const wrappedFn = withSpan(generatorFunction);
    const result = wrappedFn();
    const value = result.next().value;
    expect(value).toBe("result");
    const spans = spanExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.name).toBe("generatorFunction");
    expect(span.attributes["output.value"]).toBe("result");
  });
});

describe("traceChain", () => {
  beforeEach(() => {
    spanExporter = new InMemorySpanExporter();
    tracerProvider = new NodeTracerProvider({
      resource: resourceFromAttributes({ "service.name": "test-service" }),
      spanProcessors: [new SimpleSpanProcessor(spanExporter)],
    });
    tracerProvider.register();
  });

  afterEach(() => {
    spanExporter.reset();
    tracerProvider.shutdown();
  });

  it("should create spans with CHAIN kind", () => {
    const testFn = () => "chain result";
    const tracer = tracerProvider.getTracer("test");
    const wrappedFn = traceChain(testFn, {
      name: "chain-operation",
      tracer,
    });

    wrappedFn();

    const spans = spanExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.name).toBe("chain-operation");
    expect(span.attributes["openinference.span.kind"]).toBe(
      OpenInferenceSpanKind.CHAIN,
    );
  });
});

describe("withAgentSpan", () => {
  beforeEach(() => {
    spanExporter = new InMemorySpanExporter();
    tracerProvider = new NodeTracerProvider({
      resource: resourceFromAttributes({ "service.name": "test-service" }),
      spanProcessors: [new SimpleSpanProcessor(spanExporter)],
    });
    tracerProvider.register();
  });

  afterEach(() => {
    spanExporter.reset();
    tracerProvider.shutdown();
  });

  it("should create spans with AGENT kind", () => {
    const testFn = () => "agent result";
    const tracer = tracerProvider.getTracer("test");
    const wrappedFn = traceAgent(testFn, {
      name: "agent-operation",
      tracer,
    });

    wrappedFn();

    const spans = spanExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.name).toBe("agent-operation");
    expect(span.attributes["openinference.span.kind"]).toBe(
      OpenInferenceSpanKind.AGENT,
    );
  });
});

describe("traceTool", () => {
  beforeEach(() => {
    spanExporter = new InMemorySpanExporter();
    tracerProvider = new NodeTracerProvider({
      resource: resourceFromAttributes({ "service.name": "test-service" }),
      spanProcessors: [new SimpleSpanProcessor(spanExporter)],
    });
    tracerProvider.register();
  });

  afterEach(() => {
    spanExporter.reset();
    tracerProvider.shutdown();
  });

  it("should create spans with TOOL kind", () => {
    const testFn = () => "tool result";
    const tracer = tracerProvider.getTracer("test");
    const wrappedFn = traceTool(testFn, {
      name: "tool-operation",
      tracer,
    });

    wrappedFn();

    const spans = spanExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.name).toBe("tool-operation");
    expect(span.attributes["openinference.span.kind"]).toBe(
      OpenInferenceSpanKind.TOOL,
    );
  });
});
