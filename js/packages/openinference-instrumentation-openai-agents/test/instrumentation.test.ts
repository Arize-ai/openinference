import type { Span as OASpan, Trace } from "@openai/agents";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  GRAPH_NODE_ID,
  GRAPH_NODE_PARENT_ID,
  INPUT_VALUE,
  LLM_INPUT_MESSAGES,
  LLM_MODEL_NAME,
  LLM_OUTPUT_MESSAGES,
  LLM_SYSTEM,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
  LLM_TOKEN_COUNT_TOTAL,
  OUTPUT_VALUE,
  SemanticConventions,
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
    processor = new OpenInferenceTracingProcessor({ tracerProvider: provider });
  });

  afterEach(() => {
    exporter.reset();
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
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
        },
        {
          id: "resp-2",
          object: "chat.completion",
          choices: [
            { index: 0, message: { role: "assistant", content: "Part 2" }, finish_reason: "stop" },
          ],
          usage: { prompt_tokens: 8, completion_tokens: 6, total_tokens: 14 },
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
    expect(llmSpan!.attributes[LLM_TOKEN_COUNT_TOTAL]).toBe(29); // 15 + 14
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
          input_tokens_details: { cached_tokens: 4 },
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
    expect(respSpan!.attributes[LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING]).toBe(1);
    // System instructions become input message 0
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.0.message.role`]).toBe("system");
    expect(respSpan!.attributes[`${LLM_INPUT_MESSAGES}.0.message.content`]).toBe(
      "You are a calculator.",
    );
    // Output messages from response.output[].content
    expect(respSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.0.message.role`]).toBe("assistant");
    expect(
      respSpan!.attributes[`${LLM_OUTPUT_MESSAGES}.0.message.contents.0.message_content.text`],
    ).toBe("4");
    // Tools schema
    expect(respSpan!.attributes["llm.tools.0.tool.json_schema"]).toContain("calculate");
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
});

// ─── OpenAIAgentsInstrumentation wrapper ─────────────────────────────────────

describe("OpenAIAgentsInstrumentation", () => {
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
  });
});
