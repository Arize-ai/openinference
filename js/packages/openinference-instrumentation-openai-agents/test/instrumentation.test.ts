import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import { OpenInferenceTracingProcessor } from "../src/processor";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("OpenInferenceTracingProcessor", () => {
  let processor: OpenInferenceTracingProcessor;
  let provider: NodeTracerProvider;
  let exporter: InMemorySpanExporter;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    const tracer = provider.getTracer("test");
    processor = new OpenInferenceTracingProcessor({ tracer });
  });

  afterEach(async () => {
    await processor.shutdown();
    exporter.reset();
  });

  it("should create a span on trace start", async () => {
    const trace = {
      type: "trace" as const,
      traceId: "test-trace-id",
      name: "Test Agent Workflow",
    };

    await processor.onTraceStart(trace);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    expect(spans[0].name).toBe("Test Agent Workflow");
    expect(spans[0].attributes["openinference.span.kind"]).toBe("AGENT");
  });

  it("should create child spans for nested spans", async () => {
    const trace = {
      type: "trace" as const,
      traceId: "test-trace-id",
      name: "Test Agent Workflow",
    };

    const generationSpan = {
      type: "trace.span" as const,
      traceId: "test-trace-id",
      spanId: "generation-span-id",
      parentId: null,
      spanData: {
        type: "generation",
        name: "gpt-4o generation",
        model: "gpt-4o",
        input: [{ role: "user", content: "Hello" }],
        output: [{ role: "assistant", content: "Hi there!" }],
        usage: { input_tokens: 10, output_tokens: 5 },
      },
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      error: null,
    };

    await processor.onTraceStart(trace);
    await processor.onSpanStart(generationSpan);
    await processor.onSpanEnd(generationSpan);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    expect(spans.length).toBe(2);

    // Find the generation span
    const genSpan = spans.find((s) => s.name === "gpt-4o generation");
    expect(genSpan).toBeDefined();
    expect(genSpan?.attributes["openinference.span.kind"]).toBe("LLM");
    expect(genSpan?.attributes["llm.model_name"]).toBe("gpt-4o");
    expect(genSpan?.attributes["llm.system"]).toBe("openai");
    expect(genSpan?.attributes["llm.token_count.prompt"]).toBe(10);
    expect(genSpan?.attributes["llm.token_count.completion"]).toBe(5);
  });

  it("should handle function spans", async () => {
    const trace = {
      type: "trace" as const,
      traceId: "test-trace-id",
      name: "Test Agent Workflow",
    };

    const functionSpan = {
      type: "trace.span" as const,
      traceId: "test-trace-id",
      spanId: "function-span-id",
      parentId: null,
      spanData: {
        type: "function",
        name: "get_weather",
        input: '{"location": "San Francisco"}',
        output: '{"temperature": 72, "unit": "F"}',
      },
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      error: null,
    };

    await processor.onTraceStart(trace);
    await processor.onSpanStart(functionSpan);
    await processor.onSpanEnd(functionSpan);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    const funcSpan = spans.find((s) => s.name === "get_weather");

    expect(funcSpan).toBeDefined();
    expect(funcSpan?.attributes["openinference.span.kind"]).toBe("TOOL");
    expect(funcSpan?.attributes["tool.name"]).toBe("get_weather");
    expect(funcSpan?.attributes["input.value"]).toBe(
      '{"location": "San Francisco"}',
    );
    expect(funcSpan?.attributes["output.value"]).toBe(
      '{"temperature": 72, "unit": "F"}',
    );
  });

  it("should handle agent spans with graph tracking", async () => {
    const trace = {
      type: "trace" as const,
      traceId: "test-trace-id",
      name: "Test Agent Workflow",
    };

    const agentSpan = {
      type: "trace.span" as const,
      traceId: "test-trace-id",
      spanId: "agent-span-id",
      parentId: null,
      spanData: {
        type: "agent",
        name: "WeatherAgent",
        tools: ["get_weather"],
        handoffs: [],
      },
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      error: null,
    };

    await processor.onTraceStart(trace);
    await processor.onSpanStart(agentSpan);
    await processor.onSpanEnd(agentSpan);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    const agentOtelSpan = spans.find((s) => s.name === "WeatherAgent");

    expect(agentOtelSpan).toBeDefined();
    expect(agentOtelSpan?.attributes["openinference.span.kind"]).toBe("AGENT");
    expect(agentOtelSpan?.attributes["graph.node.id"]).toBe("WeatherAgent");
  });

  it("should handle handoff tracking for graph visualization", async () => {
    const trace = {
      type: "trace" as const,
      traceId: "test-trace-id",
      name: "Test Agent Workflow",
    };

    // First, a handoff span is created
    const handoffSpan = {
      type: "trace.span" as const,
      traceId: "test-trace-id",
      spanId: "handoff-span-id",
      parentId: null,
      spanData: {
        type: "handoff",
        from_agent: "TriageAgent",
        to_agent: "SpecialistAgent",
      },
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      error: null,
    };

    // Then the specialist agent span is created
    const specialistAgentSpan = {
      type: "trace.span" as const,
      traceId: "test-trace-id",
      spanId: "specialist-agent-span-id",
      parentId: null,
      spanData: {
        type: "agent",
        name: "SpecialistAgent",
        tools: [],
        handoffs: [],
      },
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      error: null,
    };

    await processor.onTraceStart(trace);
    await processor.onSpanStart(handoffSpan);
    await processor.onSpanEnd(handoffSpan);
    await processor.onSpanStart(specialistAgentSpan);
    await processor.onSpanEnd(specialistAgentSpan);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();

    // Check handoff span
    const handoffOtelSpan = spans.find(
      (s) => s.name === "handoff to SpecialistAgent",
    );
    expect(handoffOtelSpan).toBeDefined();
    expect(handoffOtelSpan?.attributes["openinference.span.kind"]).toBe("TOOL");

    // Check specialist agent span has parent_id set from handoff
    const specialistOtelSpan = spans.find((s) => s.name === "SpecialistAgent");
    expect(specialistOtelSpan).toBeDefined();
    expect(specialistOtelSpan?.attributes["graph.node.id"]).toBe(
      "SpecialistAgent",
    );
    expect(specialistOtelSpan?.attributes["graph.node.parent_id"]).toBe(
      "TriageAgent",
    );
  });

  it("should handle error spans", async () => {
    const trace = {
      type: "trace" as const,
      traceId: "test-trace-id",
      name: "Test Agent Workflow",
    };

    const errorSpan = {
      type: "trace.span" as const,
      traceId: "test-trace-id",
      spanId: "error-span-id",
      parentId: null,
      spanData: {
        type: "function",
        name: "failing_function",
        input: "{}",
        output: "",
      },
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      error: {
        message: "Function execution failed",
        data: { reason: "timeout" },
      },
    };

    await processor.onTraceStart(trace);
    await processor.onSpanStart(errorSpan);
    await processor.onSpanEnd(errorSpan);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    const errSpan = spans.find((s) => s.name === "failing_function");

    expect(errSpan).toBeDefined();
    expect(errSpan?.status.code).toBe(2); // SpanStatusCode.ERROR
    expect(errSpan?.status.message).toContain("Function execution failed");
  });

  it("should not create spans after shutdown", async () => {
    const trace = {
      type: "trace" as const,
      traceId: "test-trace-id",
      name: "Test Agent Workflow",
    };

    await processor.shutdown();
    await processor.onTraceStart(trace);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    expect(spans.length).toBe(0);
  });

  it("should handle message content arrays", async () => {
    const trace = {
      type: "trace" as const,
      traceId: "test-trace-id",
      name: "Test Agent Workflow",
    };

    const generationSpan = {
      type: "trace.span" as const,
      traceId: "test-trace-id",
      spanId: "generation-span-id",
      parentId: null,
      spanData: {
        type: "generation",
        name: "multimodal generation",
        model: "gpt-4o",
        input: [
          {
            role: "user",
            content: [
              { type: "text", text: "What is in this image?" },
              { type: "image_url", url: "https://example.com/image.png" },
            ],
          },
        ],
        output: [{ role: "assistant", content: "I see a cat in the image." }],
      },
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      error: null,
    };

    await processor.onTraceStart(trace);
    await processor.onSpanStart(generationSpan);
    await processor.onSpanEnd(generationSpan);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    const genSpan = spans.find((s) => s.name === "multimodal generation");

    expect(genSpan).toBeDefined();
    expect(
      genSpan?.attributes[
        "llm.input_messages.0.message.contents.0.message_content.type"
      ],
    ).toBe("text");
    expect(
      genSpan?.attributes[
        "llm.input_messages.0.message.contents.0.message_content.text"
      ],
    ).toBe("What is in this image?");
  });

  it("should handle tool calls in messages", async () => {
    const trace = {
      type: "trace" as const,
      traceId: "test-trace-id",
      name: "Test Agent Workflow",
    };

    const generationSpan = {
      type: "trace.span" as const,
      traceId: "test-trace-id",
      spanId: "generation-span-id",
      parentId: null,
      spanData: {
        type: "generation",
        name: "tool call generation",
        model: "gpt-4o",
        input: [{ role: "user", content: "What's the weather in NYC?" }],
        output: [
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_123",
                type: "function",
                function: {
                  name: "get_weather",
                  arguments: '{"location": "New York"}',
                },
              },
            ],
          },
        ],
      },
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      error: null,
    };

    await processor.onTraceStart(trace);
    await processor.onSpanStart(generationSpan);
    await processor.onSpanEnd(generationSpan);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    const genSpan = spans.find((s) => s.name === "tool call generation");

    expect(genSpan).toBeDefined();
    expect(
      genSpan?.attributes[
        "llm.output_messages.0.message.tool_calls.0.tool_call.id"
      ],
    ).toBe("call_123");
    expect(
      genSpan?.attributes[
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name"
      ],
    ).toBe("get_weather");
    expect(
      genSpan?.attributes[
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"
      ],
    ).toBe('{"location": "New York"}');
  });
});

describe("OpenAIAgentsInstrumentation", () => {
  it("should export the instrumentation class", async () => {
    const { OpenAIAgentsInstrumentation } = await import(
      "../src/instrumentation"
    );
    expect(OpenAIAgentsInstrumentation).toBeDefined();
  });

  it("should not be enabled by default", async () => {
    const { OpenAIAgentsInstrumentation } = await import(
      "../src/instrumentation"
    );
    const instrumentation = new OpenAIAgentsInstrumentation();
    expect(instrumentation.isEnabled()).toBe(false);
  });

  it("should accept custom tracer provider", async () => {
    const { OpenAIAgentsInstrumentation } = await import(
      "../src/instrumentation"
    );
    const provider = new NodeTracerProvider();
    const instrumentation = new OpenAIAgentsInstrumentation({
      tracerProvider: provider,
    });

    expect(instrumentation.tracer).toBeDefined();
  });

  it("should accept trace config options", async () => {
    const { OpenAIAgentsInstrumentation } = await import(
      "../src/instrumentation"
    );
    const instrumentation = new OpenAIAgentsInstrumentation({
      traceConfig: {
        hideInputs: true,
        hideOutputs: true,
      },
    });

    expect(instrumentation.isEnabled()).toBe(false);
  });

  it("should throw error when instrument() is called without SDK module", async () => {
    const { OpenAIAgentsInstrumentation } = await import(
      "../src/instrumentation"
    );
    const instrumentation = new OpenAIAgentsInstrumentation();

    // @ts-expect-error - testing invalid input
    expect(() => instrumentation.instrument(null)).toThrow(
      "Invalid SDK module",
    );
  });

  it("should create processor via createProcessor()", async () => {
    const { OpenAIAgentsInstrumentation } = await import(
      "../src/instrumentation"
    );
    const provider = new NodeTracerProvider();
    const instrumentation = new OpenAIAgentsInstrumentation({
      tracerProvider: provider,
    });

    const processor = instrumentation.createProcessor();
    expect(processor).toBeDefined();
    expect(instrumentation.isEnabled()).toBe(true);
    expect(instrumentation.getProcessor()).toBe(processor);
  });

  it("should instrument with mock SDK module", async () => {
    const { OpenAIAgentsInstrumentation } = await import(
      "../src/instrumentation"
    );
    const provider = new NodeTracerProvider();
    const instrumentation = new OpenAIAgentsInstrumentation({
      tracerProvider: provider,
    });

    const addedProcessors: unknown[] = [];
    const mockSdk = {
      addTraceProcessor: (processor: unknown) => {
        addedProcessors.push(processor);
      },
      startTraceExportLoop: vi.fn(),
    };

    instrumentation.instrument(mockSdk);

    expect(instrumentation.isEnabled()).toBe(true);
    expect(addedProcessors.length).toBe(1);
    expect(mockSdk.startTraceExportLoop).toHaveBeenCalled();
  });
});
