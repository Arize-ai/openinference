import type { Span as OASpan, Trace } from "@openai/agents";
import type * as AgentsModule from "@openai/agents";
import { context } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { REDACTED_VALUE, setSession } from "@arizeai/openinference-core";
import {
  GRAPH_NODE_ID,
  GRAPH_NODE_PARENT_ID,
  INPUT_VALUE,
  LLM_INPUT_MESSAGES,
  LLM_INVOCATION_PARAMETERS,
  LLM_MODEL_NAME,
  LLM_OUTPUT_MESSAGES,
  LLM_PROVIDER,
  LLM_SYSTEM,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
  LLM_TOKEN_COUNT_TOTAL,
  OUTPUT_VALUE,
  SemanticConventions,
  SESSION_ID,
  TOOL_NAME,
} from "@arizeai/openinference-semantic-conventions";

import { OpenInferenceTracingProcessor } from "../src/processor";

const OPENINFERENCE_SPAN_KIND = SemanticConventions.OPENINFERENCE_SPAN_KIND;

// ─── Helpers to create minimal fake SDK objects ──────────────────────────────

function makeTrace(overrides: Partial<Trace> = {}): Trace {
  return {
    type: "trace",
    traceId: "trace-1",
    name: "Test Workflow",
    groupId: null,
    metadata: undefined,
    start: async () => {},
    end: async () => {},
    clone: () => makeTrace(overrides),
    toJSON: () => ({}),
    ...overrides,
  } as unknown as Trace;
}

function makeSpan<T>(
  spanId: string,
  traceId: string,
  spanData: T,
  overrides: {
    parentId?: string | null;
    startedAt?: string;
    endedAt?: string;
    error?: { message: string; data?: Record<string, unknown> } | null;
  } = {},
): OASpan<T> {
  const started = overrides.startedAt ?? new Date().toISOString();
  const ended = overrides.endedAt ?? new Date().toISOString();
  return {
    type: "trace.span",
    traceId,
    spanId,
    parentId: overrides.parentId ?? null,
    spanData,
    startedAt: started,
    endedAt: ended,
    error: overrides.error ?? null,
    previousSpan: undefined,
    traceMetadata: undefined,
    tracingApiKey: undefined,
    start: () => {},
    end: () => {},
    setError: () => {},
    clone: () => makeSpan(spanId, traceId, spanData, overrides),
    toJSON: () => ({}),
  } as unknown as OASpan<T>;
}

// ─── Test setup ──────────────────────────────────────────────────────────────

describe("OpenInferenceTracingProcessor", () => {
  let exporter: InMemorySpanExporter;
  let provider: NodeTracerProvider;
  let processor: OpenInferenceTracingProcessor;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    provider.register();
    processor = new OpenInferenceTracingProcessor({ tracerProvider: provider });
  });

  afterEach(() => {
    exporter.reset();
    context.disable();
  });

  // ─── Trace lifecycle ───────────────────────────────────────────────────────

  it("creates a root span when a trace starts", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].name).toBe("Test Workflow");
    expect(spans[0].attributes[OPENINFERENCE_SPAN_KIND]).toBe("AGENT");
  });

  it("ends the root span without error when trace ends cleanly", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);
    await processor.onTraceEnd(trace);

    const [rootSpan] = exporter.getFinishedSpans();
    // SpanStatusCode.OK = 1
    expect(rootSpan.status.code).toBe(1);
  });

  it("bounds in-flight root spans by evicting the oldest root span", async () => {
    processor = new OpenInferenceTracingProcessor({
      tracerProvider: provider,
      maxRootSpansInFlight: 1,
    });

    const firstTrace = makeTrace({ traceId: "trace-1", name: "First Workflow" });
    const secondTrace = makeTrace({ traceId: "trace-2", name: "Second Workflow" });

    await processor.onTraceStart(firstTrace);
    await processor.onTraceStart(secondTrace);

    expect(exporter.getFinishedSpans().map((span) => span.name)).toEqual(["First Workflow"]);

    await processor.onTraceEnd(firstTrace);
    expect(exporter.getFinishedSpans().map((span) => span.name)).toEqual(["First Workflow"]);

    await processor.onTraceEnd(secondTrace);
    expect(exporter.getFinishedSpans().map((span) => span.name)).toEqual([
      "First Workflow",
      "Second Workflow",
    ]);
  });

  it("does not create spans when the trace starts under suppressed tracing", async () => {
    const trace = makeTrace();
    const span = makeSpan("span-suppressed", "trace-1", {
      type: "function" as const,
      name: "hidden_tool",
      input: "{}",
      output: "hidden",
    });

    await context.with(suppressTracing(context.active()), async () => {
      await processor.onTraceStart(trace);
      await processor.onSpanStart(span);
    });
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    expect(exporter.getFinishedSpans()).toHaveLength(0);
  });

  it("does not create a span when only that span starts under suppressed tracing", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const span = makeSpan("span-suppressed", "trace-1", {
      type: "function" as const,
      name: "hidden_tool",
      input: "{}",
      output: "hidden",
    });
    await context.with(suppressTracing(context.active()), async () => {
      await processor.onSpanStart(span);
    });
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].name).toBe("Test Workflow");
  });

  it("propagates OpenInference context attributes to created spans", async () => {
    const trace = makeTrace();
    const span = makeSpan("span-context", "trace-1", {
      type: "function" as const,
      name: "context_tool",
      input: "{}",
      output: "ok",
    });

    await context.with(setSession(context.active(), { sessionId: "session-123" }), async () => {
      await processor.onTraceStart(trace);
      await processor.onSpanStart(span);
    });
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    expect(spans.find((s) => s.name === "Test Workflow")!.attributes[SESSION_ID]).toBe(
      "session-123",
    );
    expect(spans.find((s) => s.name === "context_tool")!.attributes[SESSION_ID]).toBe(
      "session-123",
    );
  });

  it("applies TraceConfig masking to span attributes", async () => {
    processor = new OpenInferenceTracingProcessor({
      tracerProvider: provider,
      traceConfig: { hideInputs: true, hideOutputs: true },
    });

    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const span = makeSpan("span-masked", "trace-1", {
      type: "function" as const,
      name: "masked_tool",
      input: '{"secret":"input"}',
      output: '{"secret":"output"}',
    });
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const toolSpan = exporter.getFinishedSpans().find((s) => s.name === "masked_tool");
    expect(toolSpan!.attributes[INPUT_VALUE]).toBe(REDACTED_VALUE);
    expect(toolSpan!.attributes[OUTPUT_VALUE]).toBe(REDACTED_VALUE);
  });

  // ─── Agent span ────────────────────────────────────────────────────────────

  it("creates an AGENT span for AgentSpanData", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const span = makeSpan("span-1", "trace-1", { type: "agent" as const, name: "MyAgent" });
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    // root span + agent span
    expect(spans).toHaveLength(2);
    const agentSpan = spans.find((s) => s.name === "MyAgent");
    expect(agentSpan).toBeDefined();
    expect(agentSpan!.attributes[OPENINFERENCE_SPAN_KIND]).toBe("AGENT");
    expect(agentSpan!.attributes[GRAPH_NODE_ID]).toBe("MyAgent");
    expect(agentSpan!.attributes[LLM_SYSTEM]).toBe("openai");
  });

  // ─── Generation span ───────────────────────────────────────────────────────

  it("creates an LLM span with model and token attributes for GenerationSpanData", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const generationData = {
      type: "generation" as const,
      model: "gpt-4o",
      model_config: { temperature: 0.7 },
      input: [{ role: "user", content: "Hello" }],
      output: [{ role: "assistant", content: "Hi there!" }],
      usage: { input_tokens: 10, output_tokens: 20 },
    };
    const span = makeSpan("span-gen", "trace-1", generationData);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    // GenerationSpanData has no "name" field, so getSpanName falls back to span.spanData.type = "generation"
    const llmSpan = spans.find((s) => s.name === "generation");
    expect(llmSpan).toBeDefined();
    expect(llmSpan!.attributes[OPENINFERENCE_SPAN_KIND]).toBe("LLM");
    expect(llmSpan!.attributes[LLM_MODEL_NAME]).toBe("gpt-4o");
    expect(llmSpan!.attributes[LLM_TOKEN_COUNT_PROMPT]).toBe(10);
    expect(llmSpan!.attributes[LLM_TOKEN_COUNT_COMPLETION]).toBe(20);
    expect(llmSpan!.attributes[`${LLM_INPUT_MESSAGES}.0.message.role`]).toBe("user");
    expect(llmSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.0.message.role`]).toBe("assistant");
  });

  it("records LLM invocation parameters", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const generationData = {
      type: "generation" as const,
      model: "gpt-4o-mini",
      model_config: { temperature: 0.5, max_tokens: 512 },
    };
    const span = makeSpan("span-inv", "trace-1", generationData);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    // GenerationSpanData has no "name" field, so span name falls back to "generation"
    const llmSpan = exporter.getFinishedSpans().find((s) => s.name === "generation");
    expect(llmSpan!.attributes["llm.invocation_parameters"]).toContain("temperature");
  });

  // ─── Function / tool span ──────────────────────────────────────────────────

  it("creates a TOOL span with tool name and I/O for FunctionSpanData", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const fnData = {
      type: "function" as const,
      name: "get_weather",
      input: '{"city":"Tokyo"}',
      output: '{"temp":20,"unit":"C"}',
    };
    const span = makeSpan("span-fn", "trace-1", fnData);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const toolSpan = exporter.getFinishedSpans().find((s) => s.name === "get_weather");
    expect(toolSpan).toBeDefined();
    expect(toolSpan!.attributes[OPENINFERENCE_SPAN_KIND]).toBe("TOOL");
    expect(toolSpan!.attributes[TOOL_NAME]).toBe("get_weather");
    expect(toolSpan!.attributes[INPUT_VALUE]).toBe('{"city":"Tokyo"}');
    expect(toolSpan!.attributes[OUTPUT_VALUE]).toBe('{"temp":20,"unit":"C"}');
  });

  // ─── Handoff (multi-agent) span ────────────────────────────────────────────

  it("links agent spans via GRAPH_NODE_PARENT_ID for multi-agent handoff", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    // Start agent A
    const agentASpan = makeSpan("span-agent-a", "trace-1", {
      type: "agent" as const,
      name: "AgentA",
    });
    await processor.onSpanStart(agentASpan);
    await processor.onSpanEnd(agentASpan);

    // Handoff from A to B
    const handoffSpan = makeSpan("span-handoff", "trace-1", {
      type: "handoff" as const,
      from_agent: "AgentA",
      to_agent: "AgentB",
    });
    await processor.onSpanStart(handoffSpan);
    await processor.onSpanEnd(handoffSpan);

    // Agent B starts after the handoff
    const agentBSpan = makeSpan("span-agent-b", "trace-1", {
      type: "agent" as const,
      name: "AgentB",
    });
    await processor.onSpanStart(agentBSpan);
    await processor.onSpanEnd(agentBSpan);

    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    const handoffOtel = spans.find((s) => s.name === "handoff to AgentB");
    expect(handoffOtel).toBeDefined();
    expect(handoffOtel!.attributes[OPENINFERENCE_SPAN_KIND]).toBe("TOOL");
    expect(handoffOtel!.attributes[TOOL_NAME]).toBe("handoff_to_AgentB");
    expect(handoffOtel!.attributes[INPUT_VALUE]).toBe(JSON.stringify({ from_agent: "AgentA" }));
    expect(handoffOtel!.attributes[OUTPUT_VALUE]).toBe(JSON.stringify({ to_agent: "AgentB" }));

    const agentBOtel = spans.find((s) => s.name === "AgentB");
    expect(agentBOtel).toBeDefined();
    expect(agentBOtel!.attributes[GRAPH_NODE_PARENT_ID]).toBe("AgentA");
  });

  // ─── Guardrail span ────────────────────────────────────────────────────────

  it("creates a GUARDRAIL span for GuardrailSpanData", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const guardrailData = {
      type: "guardrail" as const,
      name: "PII check",
      triggered: false,
    };
    const span = makeSpan("span-gr", "trace-1", guardrailData);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const grSpan = exporter.getFinishedSpans().find((s) => s.name === "PII check");
    expect(grSpan).toBeDefined();
    expect(grSpan!.attributes[OPENINFERENCE_SPAN_KIND]).toBe("GUARDRAIL");
    expect(grSpan!.attributes[TOOL_NAME]).toBe("PII check");
  });

  // ─── Error handling ────────────────────────────────────────────────────────

  it("sets ERROR status when span has an error", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const span = makeSpan(
      "span-err",
      "trace-1",
      { type: "function" as const, name: "failing_tool", input: "{}", output: "" },
      { error: { message: "Tool failed", data: { code: 500 } } },
    );
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const errSpan = exporter.getFinishedSpans().find((s) => s.name === "failing_tool");
    // SpanStatusCode.ERROR = 2
    expect(errSpan!.status.code).toBe(2);
    expect(errSpan!.status.message).toContain("Tool failed");
  });

  it("ends a dangling root span when a top-level agent errors before trace end", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const span = makeSpan(
      "span-agent-error",
      "trace-1",
      {
        type: "agent" as const,
        name: "GuardrailAssistant",
      },
      { error: { message: "InputGuardrailTripwireTriggered" } },
    );

    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);

    const rootSpan = exporter.getFinishedSpans().find((s) => s.name === "Test Workflow");
    expect(rootSpan).toBeDefined();
    expect(rootSpan!.status.code).toBe(2);
    expect(rootSpan!.status.message).toContain("InputGuardrailTripwireTriggered");
  });

  // ─── Parent-child relationship ─────────────────────────────────────────────

  it("nests child spans under their parent using parentId", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const agentSpan = makeSpan("span-parent", "trace-1", {
      type: "agent" as const,
      name: "ParentAgent",
    });
    await processor.onSpanStart(agentSpan);

    const childSpan = makeSpan(
      "span-child",
      "trace-1",
      { type: "function" as const, name: "child_tool", input: "{}", output: "result" },
      { parentId: "span-parent" },
    );
    await processor.onSpanStart(childSpan);
    await processor.onSpanEnd(childSpan);
    await processor.onSpanEnd(agentSpan);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    const child = spans.find((s) => s.name === "child_tool");
    const parent = spans.find((s) => s.name === "ParentAgent");

    expect(child).toBeDefined();
    expect(parent).toBeDefined();
    // Child's parentSpanId should reference the parent
    expect(child!.parentSpanId).toBe(parent!.spanContext().spanId);
  });

  // ─── Lifecycle methods ─────────────────────────────────────────────────────

  it("shutdown and forceFlush resolve without error", async () => {
    await expect(processor.shutdown()).resolves.toBeUndefined();
    await expect(processor.forceFlush()).resolves.toBeUndefined();
  });

  it("does not create spans after the processor is disabled", async () => {
    processor.disable();

    const trace = makeTrace();
    const span = makeSpan("span-disabled", "trace-1", {
      type: "function" as const,
      name: "disabled_tool",
      input: "{}",
      output: "hidden",
    });

    await processor.onTraceStart(trace);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    expect(exporter.getFinishedSpans()).toHaveLength(0);
  });

  // ─── Chat Completions format token extraction ──────────────────────────────

  it("extracts token counts from chat completion response objects in output[]", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    // Simulate the real spanData produced when using chat_completions transport:
    // data.usage is undefined; token counts live inside output[].usage
    const generationData = {
      type: "generation" as const,
      model: "deepseek-chat",
      model_config: {},
      input: [{ role: "user", content: "hi" }],
      output: [
        {
          id: "resp-1",
          object: "chat.completion",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "Hello!" },
              finish_reason: "stop",
            },
          ],
          usage: {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            prompt_tokens_details: { cached_tokens: 3 },
          },
        },
      ],
      usage: undefined,
    };

    const span = makeSpan("span-cc", "trace-1", generationData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const llmSpan = exporter.getFinishedSpans().find((s) => s.name === "generation");
    expect(llmSpan).toBeDefined();
    expect(llmSpan!.attributes[LLM_TOKEN_COUNT_PROMPT]).toBe(10);
    expect(llmSpan!.attributes[LLM_TOKEN_COUNT_COMPLETION]).toBe(5);
    expect(llmSpan!.attributes[LLM_TOKEN_COUNT_TOTAL]).toBe(15);
    expect(llmSpan!.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]).toBe(3);
  });

  it("extracts output messages from chat completion choices", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const generationData = {
      type: "generation" as const,
      model: "gpt-4o",
      model_config: {},
      input: [{ role: "user", content: "hi" }],
      output: [
        {
          id: "resp-1",
          object: "chat.completion",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "Hello there!" },
              finish_reason: "stop",
            },
          ],
          usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
        },
      ],
      usage: undefined,
    };

    const span = makeSpan("span-msgs", "trace-1", generationData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const llmSpan = exporter.getFinishedSpans().find((s) => s.name === "generation");
    expect(llmSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.0.message.role`]).toBe("assistant");
    expect(llmSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.0.message.content`]).toBe("Hello there!");
  });

  it("indexes tool calls from each message starting at zero", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const generationData = {
      type: "generation" as const,
      model: "gpt-4o",
      model_config: {},
      input: [],
      output: [
        {
          role: "assistant",
          tool_calls: [{ id: "call_a", function: { name: "tool_a", arguments: '{"a":1}' } }],
        },
        {
          role: "assistant",
          tool_calls: [{ id: "call_b", function: { name: "tool_b", arguments: '{"b":2}' } }],
        },
      ],
      usage: undefined,
    };

    const span = makeSpan("span-msg-tool-calls", "trace-1", generationData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const llmSpan = exporter.getFinishedSpans().find((s) => s.name === "generation");
    expect(llmSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.0.message.tool_calls.0.tool_call.id`]).toBe(
      "call_a",
    );
    expect(llmSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.1.message.tool_calls.0.tool_call.id`]).toBe(
      "call_b",
    );
    expect(
      llmSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.1.message.tool_calls.1.tool_call.id`],
    ).toBeUndefined();
  });

  it("extracts tool_calls from chat completion choices", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const generationData = {
      type: "generation" as const,
      model: "gpt-4o",
      model_config: {},
      input: [{ role: "user", content: "weather in Tokyo?" }],
      output: [
        {
          id: "resp-1",
          object: "chat.completion",
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: null,
                tool_calls: [
                  {
                    id: "call_abc",
                    type: "function",
                    function: { name: "get_weather", arguments: '{"city":"Tokyo"}' },
                  },
                ],
              },
              finish_reason: "tool_calls",
            },
          ],
          usage: { prompt_tokens: 20, completion_tokens: 10, total_tokens: 30 },
        },
      ],
      usage: undefined,
    };

    const span = makeSpan("span-tc", "trace-1", generationData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const llmSpan = exporter.getFinishedSpans().find((s) => s.name === "generation");
    expect(llmSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.0.message.role`]).toBe("assistant");
    expect(llmSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.0.message.tool_calls.0.tool_call.id`]).toBe(
      "call_abc",
    );
    expect(
      llmSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.0.message.tool_calls.0.tool_call.function.name`],
    ).toBe("get_weather");
    expect(
      llmSpan!.attributes[
        `${LLM_OUTPUT_MESSAGES}.0.message.tool_calls.0.tool_call.function.arguments`
      ],
    ).toBe('{"city":"Tokyo"}');
  });

  it("accumulates token counts across multiple chat completion responses", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const generationData = {
      type: "generation" as const,
      model: "gpt-4o",
      model_config: {},
      input: [],
      output: [
        {
          id: "resp-1",
          object: "chat.completion",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "Part 1" },
              finish_reason: "length",
            },
          ],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 20 },
        },
        {
          id: "resp-2",
          object: "chat.completion",
          choices: [
            { index: 0, message: { role: "assistant", content: "Part 2" }, finish_reason: "stop" },
          ],
          usage: { prompt_tokens: 8, completion_tokens: 6, total_tokens: 30 },
        },
      ],
      usage: undefined,
    };

    const span = makeSpan("span-multi", "trace-1", generationData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const llmSpan = exporter.getFinishedSpans().find((s) => s.name === "generation");
    expect(llmSpan!.attributes[LLM_TOKEN_COUNT_PROMPT]).toBe(18); // 10 + 8
    expect(llmSpan!.attributes[LLM_TOKEN_COUNT_COMPLETION]).toBe(11); // 5 + 6
    expect(llmSpan!.attributes[LLM_TOKEN_COUNT_TOTAL]).toBe(50); // 20 + 30
  });

  it("extracts reasoning token count from completion_tokens_details", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const generationData = {
      type: "generation" as const,
      model: "o1-mini",
      model_config: {},
      input: [],
      output: [
        {
          id: "resp-1",
          object: "chat.completion",
          choices: [
            { index: 0, message: { role: "assistant", content: "42" }, finish_reason: "stop" },
          ],
          usage: {
            prompt_tokens: 5,
            completion_tokens: 50,
            total_tokens: 55,
            completion_tokens_details: { reasoning_tokens: 40 },
          },
        },
      ],
      usage: undefined,
    };

    const span = makeSpan("span-reasoning", "trace-1", generationData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const llmSpan = exporter.getFinishedSpans().find((s) => s.name === "generation");
    expect(llmSpan!.attributes[LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING]).toBe(40);
  });

  // ─── Responses API span (Responses transport) ───────────────────────────────

  it("extracts attributes from ResponseSpanData (Responses API)", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const responseData = {
      type: "response" as const,
      response_id: "resp_123",
      _input: [{ role: "user", content: "What's 2+2?" }],
      _response: {
        model: "gpt-4o",
        instructions: "You are a calculator.",
        usage: {
          input_tokens: 12,
          output_tokens: 3,
          total_tokens: 15,
          input_tokens_details: { cached_tokens: 4, cache_write_tokens: 2 },
          output_tokens_details: { reasoning_tokens: 1 },
        },
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "4" }],
          },
        ],
        tools: [
          {
            type: "function",
            name: "calculate",
            description: "Run a calculation",
            parameters: { type: "object", properties: {} },
          },
        ],
      },
    };

    const span = makeSpan("span-resp", "trace-1", responseData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const respSpan = exporter.getFinishedSpans().find((s) => s.name === "response");
    expect(respSpan).toBeDefined();
    expect(respSpan!.attributes[LLM_MODEL_NAME]).toBe("gpt-4o");
    expect(respSpan!.attributes[LLM_TOKEN_COUNT_PROMPT]).toBe(12);
    expect(respSpan!.attributes[LLM_TOKEN_COUNT_COMPLETION]).toBe(3);
    expect(respSpan!.attributes[LLM_TOKEN_COUNT_TOTAL]).toBe(15);
    expect(respSpan!.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]).toBe(4);
    expect(respSpan!.attributes["llm.token_count.prompt_details.cache_write"]).toBe(2);
    expect(respSpan!.attributes[LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING]).toBe(1);
    // System instructions become input message 0
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.0.message.role`]).toBe("system");
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.0.message.content`]).toBe(
      "You are a calculator.",
    );
    // User input follows system instructions in structured message attributes.
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.1.message.role`]).toBe("user");
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.1.message.content`]).toBe("What's 2+2?");
    // Output messages from response.output[].content
    expect(respSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.0.message.role`]).toBe("assistant");
    expect(
      respSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.0.message.contents.0.message_content.text`],
    ).toBe("4");
    // Tools schema
    expect(respSpan!.attributes["llm.tools.0.tool.json_schema"]).toContain("calculate");
  });

  it("extracts string ResponseSpanData input as a structured user message", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const responseData = {
      type: "response" as const,
      _input: "What is 42 * 17?",
      _response: {
        model: "gpt-4o",
        output: [],
      },
    };

    const span = makeSpan("span-resp-string-input", "trace-1", responseData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const respSpan = exporter.getFinishedSpans().find((s) => s.name === "response");
    expect(respSpan!.attributes[INPUT_VALUE]).toBe("What is 42 * 17?");
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.0.message.role`]).toBe("user");
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.0.message.content`]).toBe(
      "What is 42 * 17?",
    );
  });

  it("extracts Responses API input_text content parts as structured message contents", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const responseData = {
      type: "response" as const,
      _input: [
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "Summarize this trace." }],
        },
      ],
      _response: {
        model: "gpt-4o",
        output: [],
      },
    };

    const span = makeSpan("span-resp-input-text", "trace-1", responseData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const respSpan = exporter.getFinishedSpans().find((s) => s.name === "response");
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.0.message.role`]).toBe("user");
    expect(
      respSpan!.attributes[`${LLM_INPUT_MESSAGES}.0.message.contents.0.message_content.type`],
    ).toBe("text");
    expect(
      respSpan!.attributes[`${LLM_INPUT_MESSAGES}.0.message.contents.0.message_content.text`],
    ).toBe("Summarize this trace.");
  });

  it("extracts ResponseSpanData function call history as structured input messages", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const responseData = {
      type: "response" as const,
      _input: [
        { type: "message", role: "user", content: "What is the weather in Tokyo?" },
        {
          type: "function_call",
          callId: "call_weather",
          name: "get_weather",
          arguments: '{"city":"Tokyo"}',
        },
        {
          type: "function_call_result",
          callId: "call_weather",
          output: { type: "text", text: '{"temperature":22}' },
        },
      ],
      _response: {
        instructions: "Use tools when needed.",
        output: [],
      },
    };

    const span = makeSpan("span-resp-history", "trace-1", responseData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const respSpan = exporter.getFinishedSpans().find((s) => s.name === "response");
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.0.message.role`]).toBe("system");
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.1.message.role`]).toBe("user");
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.1.message.content`]).toBe(
      "What is the weather in Tokyo?",
    );
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.2.message.role`]).toBe("assistant");
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.2.message.tool_calls.0.tool_call.id`]).toBe(
      "call_weather",
    );
    expect(
      respSpan!.attributes[`${LLM_INPUT_MESSAGES}.2.message.tool_calls.0.tool_call.function.name`],
    ).toBe("get_weather");
    expect(
      respSpan!.attributes[
        `${LLM_INPUT_MESSAGES}.2.message.tool_calls.0.tool_call.function.arguments`
      ],
    ).toBe('{"city":"Tokyo"}');
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.3.message.role`]).toBe("tool");
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.3.message.tool_call_id`]).toBe(
      "call_weather",
    );
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.3.message.content`]).toBe(
      '{"temperature":22}',
    );
  });

  it("extracts function_call from Responses API output", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const responseData = {
      type: "response" as const,
      _response: {
        output: [
          {
            type: "function_call",
            call_id: "call_xyz",
            name: "lookup",
            arguments: '{"q":"hi"}',
          },
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "done" }],
          },
        ],
      },
    };

    const span = makeSpan("span-fc", "trace-1", responseData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const respSpan = exporter.getFinishedSpans().find((s) => s.name === "response");
    expect(respSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.0.message.role`]).toBe("assistant");
    expect(respSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.0.message.tool_calls.0.tool_call.id`]).toBe(
      "call_xyz",
    );
    expect(
      respSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.0.message.tool_calls.0.tool_call.function.name`],
    ).toBe("lookup");
    expect(
      respSpan!.attributes[
        `${LLM_OUTPUT_MESSAGES}.0.message.tool_calls.0.tool_call.function.arguments`
      ],
    ).toBe('{"q":"hi"}');
    expect(respSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.1.message.role`]).toBe("assistant");
    expect(
      respSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.1.message.contents.0.message_content.text`],
    ).toBe("done");
  });

  // ─── Custom span ────────────────────────────────────────────────────────────

  it("creates a CHAIN span for CustomSpanData with serialised data", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const span = makeSpan("span-custom", "trace-1", {
      type: "custom" as const,
      name: "my_custom_step",
      data: { stage: "validate", count: 3 },
    });
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const customSpan = exporter.getFinishedSpans().find((s) => s.name === "my_custom_step");
    expect(customSpan).toBeDefined();
    expect(customSpan!.attributes[OPENINFERENCE_SPAN_KIND]).toBe("CHAIN");
    expect(customSpan!.attributes[OUTPUT_VALUE]).toContain("validate");
  });

  // ─── MCP list-tools span ────────────────────────────────────────────────────

  it("serialises MCPListToolsSpanData result as JSON output", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const span = makeSpan("span-mcp", "trace-1", {
      type: "mcp_tools" as const,
      server: "my-mcp-server",
      result: ["tool_a", "tool_b"],
    });
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const mcpSpan = exporter.getFinishedSpans().find((s) => s.name === "mcp_tools");
    expect(mcpSpan).toBeDefined();
    expect(mcpSpan!.attributes[OPENINFERENCE_SPAN_KIND]).toBe("TOOL");
    expect(mcpSpan!.attributes[OUTPUT_VALUE]).toBe('["tool_a","tool_b"]');
  });

  // ─── Time precision ────────────────────────────────────────────────────────

  it("respects sub-millisecond ISO 8601 timestamp precision", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const span = makeSpan(
      "span-time",
      "trace-1",
      { type: "agent" as const, name: "TimeAgent" },
      {
        startedAt: "2024-01-01T00:00:00.123456Z",
        endedAt: "2024-01-01T00:00:00.234567Z",
      },
    );
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const timeSpan = exporter.getFinishedSpans().find((s) => s.name === "TimeAgent");
    // OTel HrTime is [seconds, nanoseconds]; check that the nanosecond
    // component carries sub-millisecond precision from the ISO string.
    const [, startNs] = timeSpan!.startTime;
    expect(startNs).toBe(123_456_000);
  });

  // ─── Handoff queue capacity ─────────────────────────────────────────────────

  it("does not retain unbounded handoff entries (LRU eviction)", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    // Emit a single matching handoff+agent pair to confirm LRU eviction does
    // not break the happy path. The internal cap (1000) is enforced silently.
    const handoffSpan = makeSpan("span-handoff", "trace-1", {
      type: "handoff" as const,
      from_agent: "A",
      to_agent: "B",
    });
    await processor.onSpanStart(handoffSpan);
    await processor.onSpanEnd(handoffSpan);

    const agentBSpan = makeSpan("span-b", "trace-1", { type: "agent" as const, name: "B" });
    await processor.onSpanStart(agentBSpan);
    await processor.onSpanEnd(agentBSpan);
    await processor.onTraceEnd(trace);

    const agentB = exporter.getFinishedSpans().find((s) => s.name === "B");
    expect(agentB!.attributes[GRAPH_NODE_PARENT_ID]).toBe("A");
  });

  // ─── Responses API invocation parameters & provider ─────────────────────────

  it("extracts invocation parameters and provider from ResponseSpanData", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const responseData = {
      type: "response" as const,
      _input: "hi",
      _response: {
        model: "gpt-4o",
        instructions: "Do not leak this prompt.",
        output_text: "Do not leak this output.",
        temperature: 0.5,
        top_p: 0.9,
        max_output_tokens: null,
        usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
        output: [],
        tools: [],
        status: "completed",
        error: null,
        object: "response",
      },
    };

    const span = makeSpan("span-resp-params", "trace-1", responseData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const respSpan = exporter.getFinishedSpans().find((s) => s.name === "response");
    expect(respSpan!.attributes[LLM_PROVIDER]).toBe("openai");
    const params = JSON.parse(respSpan!.attributes[LLM_INVOCATION_PARAMETERS] as string);
    expect(params.temperature).toBe(0.5);
    expect(params.top_p).toBe(0.9);
    // Bulky/duplicated and null-valued fields are excluded.
    expect(params.output).toBeUndefined();
    expect(params.usage).toBeUndefined();
    expect(params.tools).toBeUndefined();
    expect(params.instructions).toBeUndefined();
    expect(params.output_text).toBeUndefined();
    expect(params.status).toBeUndefined();
    expect(params.max_output_tokens).toBeUndefined();
  });

  it("does not leak response prompts or generated text through invocation parameters", async () => {
    processor = new OpenInferenceTracingProcessor({
      tracerProvider: provider,
      traceConfig: { hideInputs: true, hideOutputs: true },
    });
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const responseData = {
      type: "response" as const,
      _input: "secret user input",
      _response: {
        model: "gpt-4o",
        instructions: "secret system prompt",
        output_text: "secret generated text",
        temperature: 0.25,
        output: [],
      },
    };

    const span = makeSpan("span-resp-redacted-params", "trace-1", responseData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const respSpan = exporter.getFinishedSpans().find((s) => s.name === "response");
    expect(respSpan!.attributes[INPUT_VALUE]).toBe(REDACTED_VALUE);
    expect(respSpan!.attributes[OUTPUT_VALUE]).toBe(REDACTED_VALUE);

    const params = JSON.parse(respSpan!.attributes[LLM_INVOCATION_PARAMETERS] as string);
    expect(params.temperature).toBe(0.25);
    expect(params.instructions).toBeUndefined();
    expect(params.output_text).toBeUndefined();
    expect(JSON.stringify(params)).not.toContain("secret");
  });

  // ─── Lifted input/output on AGENT spans ──────────────────────────────────────

  it("lifts LLM input/output onto the enclosing agent span and the root span", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const agentSpan = makeSpan("span-agent", "trace-1", {
      type: "agent" as const,
      name: "WeatherAgent",
    });
    await processor.onSpanStart(agentSpan);

    const responseData = {
      type: "response" as const,
      _input: "What is the weather in Tokyo?",
      _response: {
        model: "gpt-4o",
        output_text: "Tokyo: 22°C, sunny.",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "Tokyo: 22°C, sunny." }],
          },
        ],
      },
    };
    const responseSpan = makeSpan("span-resp-io", "trace-1", responseData as never, {
      parentId: "span-agent",
    });
    await processor.onSpanStart(responseSpan);
    await processor.onSpanEnd(responseSpan);
    await processor.onSpanEnd(agentSpan);
    await processor.onTraceEnd(trace);

    const spans = exporter.getFinishedSpans();
    const agent = spans.find((s) => s.name === "WeatherAgent");
    expect(agent!.attributes[INPUT_VALUE]).toBe("What is the weather in Tokyo?");
    expect(agent!.attributes[OUTPUT_VALUE]).toBe("Tokyo: 22°C, sunny.");
    const root = spans.find((s) => s.name === "Test Workflow");
    expect(root!.attributes[INPUT_VALUE]).toBe("What is the weather in Tokyo?");
    expect(root!.attributes[OUTPUT_VALUE]).toBe("Tokyo: 22°C, sunny.");
  });

  it("keeps the first input and last output across multiple LLM spans", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const agentSpan = makeSpan("span-agent-multi", "trace-1", {
      type: "agent" as const,
      name: "MultiTurnAgent",
    });
    await processor.onSpanStart(agentSpan);

    const firstTurn = makeSpan(
      "span-turn-1",
      "trace-1",
      {
        type: "response" as const,
        _input: "What is the weather in Tokyo?",
        _response: { model: "gpt-4o", output: [] },
      } as never,
      { parentId: "span-agent-multi" },
    );
    await processor.onSpanStart(firstTurn);
    await processor.onSpanEnd(firstTurn);

    const secondTurn = makeSpan(
      "span-turn-2",
      "trace-1",
      {
        type: "response" as const,
        _input: [{ role: "user", content: "tool result history" }],
        _response: { model: "gpt-4o", output_text: "Tokyo: 22°C, sunny.", output: [] },
      } as never,
      { parentId: "span-agent-multi" },
    );
    await processor.onSpanStart(secondTurn);
    await processor.onSpanEnd(secondTurn);
    await processor.onSpanEnd(agentSpan);
    await processor.onTraceEnd(trace);

    const agent = exporter.getFinishedSpans().find((s) => s.name === "MultiTurnAgent");
    expect(agent!.attributes[INPUT_VALUE]).toBe("What is the weather in Tokyo?");
    expect(agent!.attributes[OUTPUT_VALUE]).toBe("Tokyo: 22°C, sunny.");
  });

  // ─── Exception events ────────────────────────────────────────────────────────

  it("records an exception event when a span has an error", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const span = makeSpan(
      "span-error-event",
      "trace-1",
      { type: "agent" as const, name: "FailingAgent" },
      { error: { message: "Guardrail tripped", data: { reason: "sensitive" } } },
    );
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const failing = exporter.getFinishedSpans().find((s) => s.name === "FailingAgent");
    expect(failing!.events).toHaveLength(1);
    expect(failing!.events[0].name).toBe("exception");
    expect(failing!.events[0].attributes!["exception.message"]).toContain("Guardrail tripped");
    expect(failing!.events[0].attributes!["exception.message"]).toContain("sensitive");
  });

  // ─── Timestamp robustness ────────────────────────────────────────────────────

  it("falls back to the SDK clock for unparseable timestamps", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const before = Date.now();
    const span = makeSpan(
      "span-bad-time",
      "trace-1",
      { type: "agent" as const, name: "BadClockAgent" },
      { startedAt: "not-a-timestamp", endedAt: "also-not-a-timestamp" },
    );
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const badSpan = exporter.getFinishedSpans().find((s) => s.name === "BadClockAgent");
    const [startSeconds] = badSpan!.startTime;
    // The span is timestamped "now", not at epoch 0.
    expect(startSeconds).toBeGreaterThanOrEqual(Math.floor(before / 1000));
  });

  // ─── Zero token counts ───────────────────────────────────────────────────────

  it("records zero token counts observed in chat completion usage", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const generationData = {
      type: "generation" as const,
      model: "gpt-4o",
      model_config: {},
      input: [],
      output: [
        {
          id: "resp-1",
          object: "chat.completion",
          choices: [
            { index: 0, message: { role: "assistant", content: "" }, finish_reason: "stop" },
          ],
          usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        },
      ],
      usage: undefined,
    };

    const span = makeSpan("span-zero-usage", "trace-1", generationData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const llmSpan = exporter.getFinishedSpans().find((s) => s.name === "generation");
    expect(llmSpan!.attributes[LLM_TOKEN_COUNT_PROMPT]).toBe(0);
    expect(llmSpan!.attributes[LLM_TOKEN_COUNT_COMPLETION]).toBe(0);
    expect(llmSpan!.attributes[LLM_TOKEN_COUNT_TOTAL]).toBe(0);
  });

  // ─── Empty tool call arguments ───────────────────────────────────────────────

  it("records empty-object tool call arguments", async () => {
    const trace = makeTrace();
    await processor.onTraceStart(trace);

    const generationData = {
      type: "generation" as const,
      model: "gpt-4o",
      model_config: {},
      input: [],
      output: [
        {
          id: "resp-1",
          object: "chat.completion",
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: null,
                tool_calls: [
                  {
                    id: "call_noargs",
                    type: "function",
                    function: { name: "list_files", arguments: "{}" },
                  },
                ],
              },
              finish_reason: "tool_calls",
            },
          ],
        },
      ],
      usage: undefined,
    };

    const span = makeSpan("span-empty-args", "trace-1", generationData as never);
    await processor.onSpanStart(span);
    await processor.onSpanEnd(span);
    await processor.onTraceEnd(trace);

    const llmSpan = exporter.getFinishedSpans().find((s) => s.name === "generation");
    expect(
      llmSpan!.attributes[
        `${LLM_OUTPUT_MESSAGES}.0.message.tool_calls.0.tool_call.function.arguments`
      ],
    ).toBe("{}");
  });
});

// ─── OpenAIAgentsInstrumentation wrapper ─────────────────────────────────────

describe("OpenAIAgentsInstrumentation", () => {
  function makeAgentsModule() {
    const setTraceProcessors = vi.fn();
    const addTraceProcessor = vi.fn();
    const module = {
      setTraceProcessors,
      addTraceProcessor,
    } as unknown as typeof AgentsModule & { openInferencePatched?: boolean };
    return { addTraceProcessor, module, setTraceProcessors };
  }

  function setModuleExports(
    instrumentation: unknown,
    module: typeof AgentsModule & { openInferencePatched?: boolean },
  ) {
    const state = instrumentation as { _modules: Array<{ moduleExports?: typeof AgentsModule }> };
    state._modules[0].moduleExports = module;
  }

  it("constructs without arguments", async () => {
    const { OpenAIAgentsInstrumentation } = await import("../src/instrumentation");
    expect(() => new OpenAIAgentsInstrumentation()).not.toThrow();
  });

  it("accepts tracerProvider and traceConfig in the constructor", async () => {
    const { OpenAIAgentsInstrumentation } = await import("../src/instrumentation");
    const provider = new NodeTracerProvider();
    expect(
      () =>
        new OpenAIAgentsInstrumentation({
          tracerProvider: provider,
          traceConfig: { hideInputs: true, hideOutputs: true },
        }),
    ).not.toThrow();
  });

  it("is enabled by default (inherits from InstrumentationBase)", async () => {
    const { OpenAIAgentsInstrumentation } = await import("../src/instrumentation");
    const instrumentation = new OpenAIAgentsInstrumentation();
    expect(instrumentation.isEnabled()).toBe(true);
  });

  it("can be disabled and re-enabled", async () => {
    const { OpenAIAgentsInstrumentation } = await import("../src/instrumentation");
    const instrumentation = new OpenAIAgentsInstrumentation();
    instrumentation.disable();
    expect(instrumentation.isEnabled()).toBe(false);
    instrumentation.enable();
    expect(instrumentation.isEnabled()).toBe(true);
  });

  it("manuallyInstrument does not throw", async () => {
    const { OpenAIAgentsInstrumentation } = await import("../src/instrumentation");
    const agents = await import("@openai/agents");
    const provider = new NodeTracerProvider();
    const instrumentation = new OpenAIAgentsInstrumentation({ tracerProvider: provider });
    expect(() => instrumentation.manuallyInstrument(agents)).not.toThrow();
    instrumentation.uninstrument();
  });

  it("instrument registers the processor exclusively by default", async () => {
    const { OpenAIAgentsInstrumentation } = await import("../src/instrumentation");
    const provider = new NodeTracerProvider();
    const instrumentation = new OpenAIAgentsInstrumentation({ tracerProvider: provider });
    const agents = makeAgentsModule();
    setModuleExports(instrumentation, agents.module);

    instrumentation.instrument();

    expect(agents.setTraceProcessors).toHaveBeenCalledWith([
      expect.any(OpenInferenceTracingProcessor),
    ]);
    expect(agents.addTraceProcessor).not.toHaveBeenCalled();
    instrumentation.uninstrument();
  });

  it("passes maxRootSpansInFlight to the registered processor", async () => {
    const { OpenAIAgentsInstrumentation } = await import("../src/instrumentation");
    const exporter = new InMemorySpanExporter();
    const provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    const instrumentation = new OpenAIAgentsInstrumentation({
      tracerProvider: provider,
      maxRootSpansInFlight: 1,
    });
    const agents = makeAgentsModule();
    setModuleExports(instrumentation, agents.module);

    instrumentation.instrument();

    const registeredProcessors = agents.setTraceProcessors.mock.calls[0][0];
    const registeredProcessor = registeredProcessors[0] as OpenInferenceTracingProcessor;
    await registeredProcessor.onTraceStart(
      makeTrace({ traceId: "trace-1", name: "First Workflow" }),
    );
    await registeredProcessor.onTraceStart(
      makeTrace({ traceId: "trace-2", name: "Second Workflow" }),
    );

    expect(exporter.getFinishedSpans().map((span) => span.name)).toEqual(["First Workflow"]);
    instrumentation.uninstrument();
  });

  it("instrument supports additive processor registration", async () => {
    const { OpenAIAgentsInstrumentation } = await import("../src/instrumentation");
    const provider = new NodeTracerProvider();
    const instrumentation = new OpenAIAgentsInstrumentation({ tracerProvider: provider });
    const agents = makeAgentsModule();
    setModuleExports(instrumentation, agents.module);

    instrumentation.instrument({ exclusiveProcessor: false });

    expect(agents.addTraceProcessor).toHaveBeenCalledWith(
      expect.any(OpenInferenceTracingProcessor),
    );
    expect(agents.setTraceProcessors).not.toHaveBeenCalled();
    agents.setTraceProcessors.mockClear();

    instrumentation.uninstrument();

    expect(agents.setTraceProcessors).not.toHaveBeenCalled();
    expect(agents.module.openInferencePatched).toBe(false);
  });

  it("manuallyInstrument supports additive processor registration", async () => {
    const { OpenAIAgentsInstrumentation } = await import("../src/instrumentation");
    const provider = new NodeTracerProvider();
    const instrumentation = new OpenAIAgentsInstrumentation({ tracerProvider: provider });
    const agents = makeAgentsModule();

    instrumentation.manuallyInstrument(agents.module, { exclusiveProcessor: false });

    expect(agents.addTraceProcessor).toHaveBeenCalledWith(
      expect.any(OpenInferenceTracingProcessor),
    );
    expect(agents.setTraceProcessors).not.toHaveBeenCalled();
    instrumentation.uninstrument();
  });

  it("constructor exclusiveProcessor option controls the registration mode", async () => {
    const { OpenAIAgentsInstrumentation } = await import("../src/instrumentation");
    const provider = new NodeTracerProvider();
    const instrumentation = new OpenAIAgentsInstrumentation({
      tracerProvider: provider,
      exclusiveProcessor: false,
    });
    const agents = makeAgentsModule();
    setModuleExports(instrumentation, agents.module);

    // No per-call option: the constructor value applies, as it would when the
    // module is patched automatically via registerInstrumentations/NodeSDK.
    instrumentation.instrument();

    expect(agents.addTraceProcessor).toHaveBeenCalledWith(
      expect.any(OpenInferenceTracingProcessor),
    );
    expect(agents.setTraceProcessors).not.toHaveBeenCalled();
    instrumentation.uninstrument();
  });

  it("re-registers the processor with the SDK when the tracer provider changes", async () => {
    const { OpenAIAgentsInstrumentation } = await import("../src/instrumentation");
    const instrumentation = new OpenAIAgentsInstrumentation();
    const agents = makeAgentsModule();
    setModuleExports(instrumentation, agents.module);

    instrumentation.instrument();
    expect(agents.setTraceProcessors).toHaveBeenCalledTimes(1);

    const exporter = new InMemorySpanExporter();
    const provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    instrumentation.setTracerProvider(provider);

    // A replacement processor bound to the new provider is registered.
    expect(agents.setTraceProcessors).toHaveBeenCalledTimes(2);
    const [replacement] = agents.setTraceProcessors.mock.calls[1][0] as [
      OpenInferenceTracingProcessor,
    ];
    expect(replacement).not.toBe(agents.setTraceProcessors.mock.calls[0][0][0]);

    await replacement.onTraceStart(makeTrace({ traceId: "trace-rt", name: "Rewired Workflow" }));
    await replacement.onTraceEnd(makeTrace({ traceId: "trace-rt", name: "Rewired Workflow" }));
    expect(exporter.getFinishedSpans().map((s) => s.name)).toEqual(["Rewired Workflow"]);
    instrumentation.uninstrument();
  });

  it("uninstrument is a no-op on an instance that did not patch", async () => {
    const { OpenAIAgentsInstrumentation } = await import("../src/instrumentation");
    const provider = new NodeTracerProvider();
    const patcher = new OpenAIAgentsInstrumentation({ tracerProvider: provider });
    const bystander = new OpenAIAgentsInstrumentation({ tracerProvider: provider });
    const agents = makeAgentsModule();
    setModuleExports(patcher, agents.module);

    patcher.instrument();
    agents.setTraceProcessors.mockClear();

    bystander.uninstrument();

    expect(agents.setTraceProcessors).not.toHaveBeenCalled();
    expect(agents.module.openInferencePatched).toBe(true);

    patcher.uninstrument();
    expect(agents.module.openInferencePatched).toBe(false);
  });
});
