import { context, SpanStatusCode } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import {
  chat,
  type ChatMiddlewareConfig,
  type ChatMiddlewareContext,
  type DefaultMessageMetadataByModality,
  type StreamChunk,
  type TextAdapter,
  type ToolCall,
} from "@tanstack/ai";
import { describe, expect, it } from "vitest";

import { setMetadata, setSession, setTags, setUser } from "@arizeai/openinference-core";
import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { openInferenceMiddleware } from "../src";

function createTracer() {
  const exporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  provider.register();

  return {
    exporter,
    tracer: provider.getTracer("test-tracer"),
  };
}

function createContext(overrides: Partial<ChatMiddlewareContext> = {}): ChatMiddlewareContext {
  return {
    requestId: "req-1",
    streamId: "stream-1",
    phase: "init",
    iteration: 0,
    chunkIndex: 0,
    abort: () => {},
    context: undefined,
    defer: () => {},
    provider: "openai",
    model: "gpt-4o-mini",
    source: "server",
    streaming: true,
    systemPrompts: [],
    toolNames: ["get_weather"],
    options: { temperature: 0.2 },
    modelOptions: {},
    messageCount: 1,
    hasTools: true,
    currentMessageId: null,
    accumulatedContent: "",
    messages: [{ role: "user", content: "What is the weather?" }],
    createId: (prefix: string) => `${prefix}-1`,
    ...overrides,
  };
}

function createConfig(messages: ChatMiddlewareConfig["messages"]): ChatMiddlewareConfig {
  return {
    messages,
    systemPrompts: [],
    tools: [
      {
        name: "get_weather",
        description: "Get the weather for a city",
        inputSchema: {
          type: "object",
          properties: {
            city: { type: "string" },
          },
          required: ["city"],
        },
      },
    ],
    temperature: 0.2,
    topP: undefined,
    maxTokens: undefined,
    metadata: undefined,
    modelOptions: {},
  };
}

function createFakeTextAdapter(
  chunks: StreamChunk[],
): TextAdapter<string, Record<string, never>, readonly ["text"], DefaultMessageMetadataByModality> {
  return {
    kind: "text",
    name: "fake",
    model: "fake-model",
    "~types": undefined as never,
    async *chatStream() {
      for (const chunk of chunks) {
        yield chunk;
      }
    },
    async structuredOutput() {
      return {
        data: { ok: true },
        rawText: '{"ok":true}',
      };
    },
  };
}

describe("openInferenceMiddleware", () => {
  it("emits agent, llm, and tool spans in logical order", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });
    const toolCall: ToolCall = {
      id: "tool-1",
      type: "function",
      function: {
        name: "get_weather",
        arguments: JSON.stringify({ city: "Boston" }),
      },
    };

    const firstCtx = createContext({ phase: "beforeModel", iteration: 0 });
    await middleware.onStart?.(createContext());
    await middleware.onConfig?.(
      firstCtx,
      createConfig(firstCtx.messages as ChatMiddlewareConfig["messages"]),
    );
    await middleware.onChunk?.(firstCtx, {
      type: "TOOL_CALL_START",
      timestamp: Date.now(),
      toolCallId: "tool-1",
      toolName: "get_weather",
    });
    await middleware.onChunk?.(firstCtx, {
      type: "TOOL_CALL_ARGS",
      timestamp: Date.now(),
      toolCallId: "tool-1",
      delta: JSON.stringify({ city: "Boston" }),
      args: JSON.stringify({ city: "Boston" }),
    });
    await middleware.onChunk?.(firstCtx, {
      type: "TOOL_CALL_END",
      timestamp: Date.now(),
      toolCallId: "tool-1",
      toolName: "get_weather",
      input: { city: "Boston" },
    });
    await middleware.onChunk?.(firstCtx, {
      type: "RUN_FINISHED",
      timestamp: Date.now(),
      runId: "run-1",
      finishReason: "tool_calls",
      usage: {
        promptTokens: 10,
        completionTokens: 4,
        totalTokens: 14,
      },
    });
    await middleware.onBeforeToolCall?.(createContext({ phase: "beforeTools", iteration: 0 }), {
      toolCall,
      tool: undefined,
      args: { city: "Boston" },
      toolName: "get_weather",
      toolCallId: "tool-1",
    });
    await middleware.onAfterToolCall?.(createContext({ phase: "afterTools", iteration: 0 }), {
      toolCall,
      tool: undefined,
      toolName: "get_weather",
      toolCallId: "tool-1",
      ok: true,
      duration: 12,
      result: { forecast: "sunny", temperatureF: 70 },
    });

    const secondMessages: ChatMiddlewareConfig["messages"] = [
      { role: "user", content: "What is the weather?" },
      { role: "assistant", content: null, toolCalls: [toolCall] },
      {
        role: "tool",
        content: JSON.stringify({ forecast: "sunny", temperatureF: 70 }),
        toolCallId: "tool-1",
      },
    ];
    const secondCtx = createContext({
      phase: "beforeModel",
      iteration: 1,
      messages: secondMessages,
    });
    await middleware.onConfig?.(secondCtx, createConfig(secondMessages));
    await middleware.onChunk?.(secondCtx, {
      type: "TEXT_MESSAGE_CONTENT",
      timestamp: Date.now(),
      messageId: "msg-2",
      delta: "It is sunny in Boston.",
      content: "It is sunny in Boston.",
    });
    await middleware.onChunk?.(secondCtx, {
      type: "RUN_FINISHED",
      timestamp: Date.now(),
      runId: "run-2",
      finishReason: "stop",
      usage: {
        promptTokens: 12,
        completionTokens: 5,
        totalTokens: 17,
      },
    });
    await middleware.onFinish?.(createContext({ iteration: 1, messages: secondMessages }), {
      finishReason: "stop",
      duration: 25,
      content: "It is sunny in Boston.",
      usage: {
        promptTokens: 22,
        completionTokens: 9,
        totalTokens: 31,
      },
    });

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(4);

    const agentSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.AGENT,
    );
    const llmSpans = spans.filter(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.LLM,
    );
    const toolSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.TOOL,
    );

    expect(agentSpan).toBeDefined();
    expect(llmSpans).toHaveLength(2);
    expect(toolSpan).toBeDefined();

    expect(agentSpan?.name).toBe("ai.chat");
    expect(llmSpans[0]?.name).toBe("ai.llm 1");
    expect(toolSpan?.name).toBe("ai.tool get_weather");
    expect(agentSpan?.attributes["output.value"]).toBe("It is sunny in Boston.");
    expect(agentSpan?.attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL]).toBeUndefined();
    expect(
      llmSpans[0]?.attributes[
        `${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`
      ],
    ).toBe("user");
    expect(llmSpans[0]?.attributes[SemanticConventions.LLM_SYSTEM]).toBe("openai");
    expect(llmSpans[0]?.attributes[SemanticConventions.LLM_MODEL_NAME]).toBe("gpt-4o-mini");
    expect(
      llmSpans[0]?.attributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.0.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
      ],
    ).toBe("get_weather");
    expect(llmSpans[0]?.attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL]).toBe(14);
    expect(
      llmSpans[1]?.attributes[
        `${SemanticConventions.LLM_INPUT_MESSAGES}.2.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`
      ],
    ).toBe("tool-1");
    expect(
      llmSpans[1]?.attributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`
      ],
    ).toBe("It is sunny in Boston.");
    expect(toolSpan?.attributes[SemanticConventions.TOOL_NAME]).toBe("get_weather");
    expect(toolSpan?.attributes["output.value"]).toBe(
      JSON.stringify({ forecast: "sunny", temperatureF: 70 }),
    );
    expect(toolSpan?.parentSpanId).toBe(agentSpan?.spanContext().spanId);
  });

  it("captures spec-shaped llm inputs including system prompts, invocation parameters, and tool schema", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });
    const ctx = createContext({
      phase: "beforeModel",
      systemPrompts: ["You are a weather assistant."],
    });

    await middleware.onStart?.(ctx);
    await middleware.onConfig?.(ctx, {
      ...createConfig(ctx.messages as ChatMiddlewareConfig["messages"]),
      systemPrompts: ["You are a weather assistant."],
      metadata: { requestType: "forecast" },
      topP: 0.9,
      maxTokens: 128,
    });
    await middleware.onChunk?.(ctx, {
      type: "TEXT_MESSAGE_CONTENT",
      timestamp: Date.now(),
      messageId: "msg-1",
      delta: "The weather is sunny.",
      content: "The weather is sunny.",
    });
    await middleware.onChunk?.(ctx, {
      type: "RUN_FINISHED",
      timestamp: Date.now(),
      runId: "run-1",
      finishReason: "stop",
      usage: {
        promptTokens: 8,
        completionTokens: 5,
        totalTokens: 13,
      },
    });
    await middleware.onFinish?.(ctx, {
      finishReason: "stop",
      duration: 10,
      content: "The weather is sunny.",
      usage: {
        promptTokens: 8,
        completionTokens: 5,
        totalTokens: 13,
      },
    });

    const llmSpan = exporter
      .getFinishedSpans()
      .find(
        (span) =>
          span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
          OpenInferenceSpanKind.LLM,
      );

    expect(llmSpan).toBeDefined();
    expect(
      llmSpan?.attributes[
        `${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`
      ],
    ).toBe("system");
    expect(
      llmSpan?.attributes[
        `${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`
      ],
    ).toBe("You are a weather assistant.");
    expect(
      llmSpan?.attributes[
        `${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_ROLE}`
      ],
    ).toBe("user");
    expect(
      llmSpan?.attributes[
        `${SemanticConventions.LLM_TOOLS}.0.${SemanticConventions.TOOL_JSON_SCHEMA}`
      ],
    ).toContain('"name":"get_weather"');
    expect(llmSpan?.attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS]).toBe(
      JSON.stringify({
        model: "gpt-4o-mini",
        temperature: 0.2,
        topP: 0.9,
        maxTokens: 128,
        metadata: { requestType: "forecast" },
        modelOptions: {},
      }),
    );
    expect(llmSpan?.attributes["input.mime_type"]).toBe("application/json");
    expect(llmSpan?.attributes["output.mime_type"]).toBe("text/plain");
  });

  it("preserves multiple system prompts in order on llm input messages", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });
    const systemPrompts = [
      "You are a careful research assistant.",
      "Always cite uncertainty when the answer is approximate.",
    ];
    const ctx = createContext({
      phase: "beforeModel",
      systemPrompts,
    });

    await middleware.onStart?.(ctx);
    await middleware.onConfig?.(ctx, {
      ...createConfig(ctx.messages as ChatMiddlewareConfig["messages"]),
      systemPrompts,
    });
    await middleware.onChunk?.(ctx, {
      type: "TEXT_MESSAGE_CONTENT",
      timestamp: Date.now(),
      messageId: "msg-1",
      delta: "The weather is probably sunny.",
      content: "The weather is probably sunny.",
    });
    await middleware.onChunk?.(ctx, {
      type: "RUN_FINISHED",
      timestamp: Date.now(),
      runId: "run-1",
      finishReason: "stop",
      usage: {
        promptTokens: 9,
        completionTokens: 5,
        totalTokens: 14,
      },
    });
    await middleware.onFinish?.(ctx, {
      finishReason: "stop",
      duration: 10,
      content: "The weather is probably sunny.",
      usage: {
        promptTokens: 9,
        completionTokens: 5,
        totalTokens: 14,
      },
    });

    const llmSpan = exporter
      .getFinishedSpans()
      .find(
        (span) =>
          span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
          OpenInferenceSpanKind.LLM,
      );

    expect(llmSpan).toBeDefined();
    expect(
      llmSpan?.attributes[
        `${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`
      ],
    ).toBe("system");
    expect(
      llmSpan?.attributes[
        `${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`
      ],
    ).toBe(systemPrompts[0]);
    expect(
      llmSpan?.attributes[
        `${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_ROLE}`
      ],
    ).toBe("system");
    expect(
      llmSpan?.attributes[
        `${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENT}`
      ],
    ).toBe(systemPrompts[1]);
    expect(
      llmSpan?.attributes[
        `${SemanticConventions.LLM_INPUT_MESSAGES}.2.${SemanticConventions.MESSAGE_ROLE}`
      ],
    ).toBe("user");
  });

  it("keeps token counts on llm spans only", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });
    const ctx = createContext({ phase: "beforeModel" });

    await middleware.onStart?.(createContext());
    await middleware.onConfig?.(
      ctx,
      createConfig(ctx.messages as ChatMiddlewareConfig["messages"]),
    );
    await middleware.onChunk?.(ctx, {
      type: "TEXT_MESSAGE_CONTENT",
      timestamp: Date.now(),
      messageId: "msg-1",
      delta: "OpenInference standardizes AI tracing.",
      content: "OpenInference standardizes AI tracing.",
    });
    await middleware.onChunk?.(ctx, {
      type: "RUN_FINISHED",
      timestamp: Date.now(),
      runId: "run-1",
      finishReason: "stop",
      usage: {
        promptTokens: 11,
        completionTokens: 6,
        totalTokens: 17,
      },
    });
    await middleware.onFinish?.(ctx, {
      finishReason: "stop",
      duration: 10,
      content: "OpenInference standardizes AI tracing.",
      usage: {
        promptTokens: 11,
        completionTokens: 6,
        totalTokens: 17,
      },
    });

    const spans = exporter.getFinishedSpans();
    const agentSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.AGENT,
    );
    const llmSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.LLM,
    );

    expect(agentSpan?.attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT]).toBeUndefined();
    expect(agentSpan?.attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]).toBeUndefined();
    expect(agentSpan?.attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL]).toBeUndefined();
    expect(llmSpan?.attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT]).toBe(11);
    expect(llmSpan?.attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]).toBe(6);
    expect(llmSpan?.attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL]).toBe(17);
  });

  it("marks llm and agent spans as errors when the model stream fails", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });
    const ctx = createContext({ phase: "beforeModel" });
    const error = new Error("model stream failed");

    await middleware.onStart?.(createContext());
    await middleware.onConfig?.(
      ctx,
      createConfig(ctx.messages as ChatMiddlewareConfig["messages"]),
    );
    await middleware.onChunk?.(ctx, {
      type: "RUN_ERROR",
      timestamp: Date.now(),
      error: {
        message: error.message,
        code: "MODEL_ERROR",
      },
      model: "gpt-4o-mini",
    });
    await middleware.onError?.(ctx, {
      error,
      duration: 15,
    });

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(2);

    const agentSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.AGENT,
    );
    const llmSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.LLM,
    );

    expect(agentSpan?.status.code).toBe(SpanStatusCode.ERROR);
    expect(llmSpan?.status.code).toBe(SpanStatusCode.ERROR);
    expect(agentSpan?.status.message).toBe("model stream failed");
    expect(llmSpan?.status.message).toBe("model stream failed");
    expect(agentSpan?.events.some((event) => event.name === "exception")).toBe(true);
    expect(llmSpan?.events.some((event) => event.name === "exception")).toBe(true);
  });

  it("instruments a real streaming chat() run with a local fake adapter", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });
    const adapter = createFakeTextAdapter([
      {
        type: "TEXT_MESSAGE_CONTENT",
        timestamp: Date.now(),
        messageId: "msg-1",
        delta: "OpenInference traces AI applications.",
        content: "OpenInference traces AI applications.",
      },
      {
        type: "RUN_FINISHED",
        timestamp: Date.now(),
        runId: "run-1",
        model: "fake-model",
        finishReason: "stop",
        usage: {
          promptTokens: 4,
          completionTokens: 5,
          totalTokens: 9,
        },
      },
    ]);

    const chunks = [] as StreamChunk[];
    for await (const chunk of chat({
      adapter,
      messages: [{ role: "user", content: "What is OpenInference?" }],
      middleware: [middleware],
    })) {
      chunks.push(chunk);
    }

    expect(chunks).toHaveLength(2);

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(2);

    const agentSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.AGENT,
    );
    const llmSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.LLM,
    );

    expect(agentSpan?.attributes["output.value"]).toBe("OpenInference traces AI applications.");
    expect(llmSpan?.attributes[SemanticConventions.LLM_MODEL_NAME]).toBe("fake-model");
    expect(llmSpan?.attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL]).toBe(9);
  });

  it("instruments a real non-streaming chat() run with a local fake adapter", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });
    const adapter = createFakeTextAdapter([
      {
        type: "TEXT_MESSAGE_CONTENT",
        timestamp: Date.now(),
        messageId: "msg-1",
        delta: "A non-streaming reply.",
        content: "A non-streaming reply.",
      },
      {
        type: "RUN_FINISHED",
        timestamp: Date.now(),
        runId: "run-1",
        model: "fake-model",
        finishReason: "stop",
        usage: {
          promptTokens: 3,
          completionTokens: 4,
          totalTokens: 7,
        },
      },
    ]);

    const text = await chat({
      adapter,
      stream: false,
      messages: [{ role: "user", content: "Reply briefly." }],
      middleware: [middleware],
    });

    expect(text).toBe("A non-streaming reply.");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(2);
    expect(
      spans.some(
        (span) =>
          span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
          OpenInferenceSpanKind.AGENT,
      ),
    ).toBe(true);
    expect(
      spans.some(
        (span) =>
          span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
          OpenInferenceSpanKind.LLM,
      ),
    ).toBe(true);
  });

  it("captures multiple tool calls in a single llm span", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });
    const ctx = createContext({ phase: "beforeModel" });

    await middleware.onStart?.(createContext());
    await middleware.onConfig?.(
      ctx,
      createConfig(ctx.messages as ChatMiddlewareConfig["messages"]),
    );
    await middleware.onChunk?.(ctx, {
      type: "TOOL_CALL_START",
      timestamp: Date.now(),
      toolCallId: "tool-1",
      toolName: "get_weather",
    });
    await middleware.onChunk?.(ctx, {
      type: "TOOL_CALL_ARGS",
      timestamp: Date.now(),
      toolCallId: "tool-1",
      delta: '{"city":"Boston"}',
      args: '{"city":"Boston"}',
    });
    await middleware.onChunk?.(ctx, {
      type: "TOOL_CALL_END",
      timestamp: Date.now(),
      toolCallId: "tool-1",
      toolName: "get_weather",
      input: { city: "Boston" },
    });
    await middleware.onChunk?.(ctx, {
      type: "TOOL_CALL_START",
      timestamp: Date.now(),
      toolCallId: "tool-2",
      toolName: "get_weather",
    });
    await middleware.onChunk?.(ctx, {
      type: "TOOL_CALL_ARGS",
      timestamp: Date.now(),
      toolCallId: "tool-2",
      delta: '{"city":"Seattle"}',
      args: '{"city":"Seattle"}',
    });
    await middleware.onChunk?.(ctx, {
      type: "TOOL_CALL_END",
      timestamp: Date.now(),
      toolCallId: "tool-2",
      toolName: "get_weather",
      input: { city: "Seattle" },
    });
    await middleware.onChunk?.(ctx, {
      type: "RUN_FINISHED",
      timestamp: Date.now(),
      runId: "run-1",
      finishReason: "tool_calls",
      usage: {
        promptTokens: 10,
        completionTokens: 8,
        totalTokens: 18,
      },
    });
    await middleware.onFinish?.(ctx, {
      finishReason: "tool_calls",
      duration: 15,
      content: "",
      usage: {
        promptTokens: 10,
        completionTokens: 8,
        totalTokens: 18,
      },
    });

    const llmSpan = exporter
      .getFinishedSpans()
      .find(
        (span) =>
          span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
          OpenInferenceSpanKind.LLM,
      );

    expect(
      llmSpan?.attributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.0.${SemanticConventions.TOOL_CALL_ID}`
      ],
    ).toBe("tool-1");
    expect(
      llmSpan?.attributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.1.${SemanticConventions.TOOL_CALL_ID}`
      ],
    ).toBe("tool-2");
  });

  it("marks tool spans as errors when tool execution fails", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });
    const toolCall: ToolCall = {
      id: "tool-1",
      type: "function",
      function: {
        name: "get_weather",
        arguments: JSON.stringify({ city: "Boston" }),
      },
    };

    await middleware.onStart?.(createContext());
    await middleware.onBeforeToolCall?.(createContext({ phase: "beforeTools" }), {
      toolCall,
      tool: undefined,
      args: { city: "Boston" },
      toolName: "get_weather",
      toolCallId: "tool-1",
    });
    await middleware.onAfterToolCall?.(createContext({ phase: "afterTools" }), {
      toolCall,
      tool: undefined,
      toolName: "get_weather",
      toolCallId: "tool-1",
      ok: false,
      duration: 3,
      error: new Error("tool boom"),
    });
    await middleware.onFinish?.(createContext(), {
      finishReason: "stop",
      duration: 5,
      content: "",
    });

    const toolSpan = exporter
      .getFinishedSpans()
      .find(
        (span) =>
          span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
          OpenInferenceSpanKind.TOOL,
      );

    expect(toolSpan?.status.code).toBe(SpanStatusCode.ERROR);
    expect(toolSpan?.status.message).toBe("tool boom");
    expect(toolSpan?.events.some((event) => event.name === "exception")).toBe(true);
  });

  it("propagates OpenInference context attributes onto emitted spans", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });
    const ctx = createContext({ phase: "beforeModel" });

    const activeContext = setTags(
      setUser(
        setMetadata(setSession(context.active(), { sessionId: "session-1" }), {
          feature: "tanstack-ai",
          numeric: 7,
        }),
        { userId: "user-1" },
      ),
      ["benchmark", "tanstack"],
    );

    context.with(activeContext, () => {
      middleware.onStart?.(createContext());
      middleware.onConfig?.(ctx, createConfig(ctx.messages as ChatMiddlewareConfig["messages"]));
      middleware.onChunk?.(ctx, {
        type: "TEXT_MESSAGE_CONTENT",
        timestamp: Date.now(),
        messageId: "msg-1",
        delta: "A traced response.",
        content: "A traced response.",
      });
      middleware.onChunk?.(ctx, {
        type: "RUN_FINISHED",
        timestamp: Date.now(),
        runId: "run-1",
        finishReason: "stop",
        usage: {
          promptTokens: 3,
          completionTokens: 3,
          totalTokens: 6,
        },
      });
      middleware.onFinish?.(ctx, {
        finishReason: "stop",
        duration: 10,
        content: "A traced response.",
      });
    });

    const spans = exporter.getFinishedSpans();
    const agentSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.AGENT,
    );
    const llmSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.LLM,
    );

    expect(agentSpan?.name).toBe("ai.chat");
    expect(llmSpan?.name).toBe("ai.llm 1");
    expect(agentSpan?.attributes[SemanticConventions.SESSION_ID]).toBe("session-1");
    expect(agentSpan?.attributes[SemanticConventions.USER_ID]).toBe("user-1");
    expect(llmSpan?.attributes[SemanticConventions.METADATA]).toBe(
      JSON.stringify({ feature: "tanstack-ai", numeric: 7 }),
    );
    expect(agentSpan?.attributes[SemanticConventions.TAG_TAGS]).toBe(
      JSON.stringify(["benchmark", "tanstack"]),
    );
    expect(llmSpan?.attributes[SemanticConventions.SESSION_ID]).toBe("session-1");
  });

  it("respects traceConfig masking on emitted spans", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({
      tracer,
      traceConfig: { hideInputs: true, hideOutputs: true },
    });
    const ctx = createContext({ phase: "beforeModel" });

    await middleware.onStart?.(createContext());
    await middleware.onConfig?.(
      ctx,
      createConfig(ctx.messages as ChatMiddlewareConfig["messages"]),
    );
    await middleware.onChunk?.(ctx, {
      type: "TEXT_MESSAGE_CONTENT",
      timestamp: Date.now(),
      messageId: "msg-1",
      delta: "A masked response.",
      content: "A masked response.",
    });
    await middleware.onChunk?.(ctx, {
      type: "RUN_FINISHED",
      timestamp: Date.now(),
      runId: "run-1",
      finishReason: "stop",
      usage: {
        promptTokens: 4,
        completionTokens: 4,
        totalTokens: 8,
      },
    });
    await middleware.onFinish?.(ctx, {
      finishReason: "stop",
      duration: 10,
      content: "A masked response.",
    });

    const spans = exporter.getFinishedSpans();
    const agentSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.AGENT,
    );
    const llmSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.LLM,
    );

    expect(agentSpan?.attributes["input.value"]).toBe("__REDACTED__");
    expect(agentSpan?.attributes["output.value"]).toBe("__REDACTED__");
    expect(llmSpan?.attributes["input.value"]).toBe("__REDACTED__");
    expect(llmSpan?.attributes["output.value"]).toBe("__REDACTED__");
  });

  it("marks unfinished tool spans as errors when the request aborts", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });
    const toolCall: ToolCall = {
      id: "tool-1",
      type: "function",
      function: {
        name: "get_weather",
        arguments: JSON.stringify({ city: "Boston" }),
      },
    };

    await middleware.onStart?.(createContext());
    await middleware.onBeforeToolCall?.(createContext({ phase: "beforeTools" }), {
      toolCall,
      tool: undefined,
      args: { city: "Boston" },
      toolName: "get_weather",
      toolCallId: "tool-1",
    });
    await middleware.onAbort?.(createContext(), {
      reason: "user cancelled",
      duration: 5,
    });

    const toolSpan = exporter
      .getFinishedSpans()
      .find(
        (span) =>
          span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
          OpenInferenceSpanKind.TOOL,
      );

    expect(toolSpan?.status.code).toBe(SpanStatusCode.ERROR);
    expect(toolSpan?.status.message).toBe("user cancelled");
  });

  it("respects suppressed tracing", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });

    await context.with(suppressTracing(context.active()), async () => {
      await middleware.onStart?.(createContext({ requestId: "req-suppressed" }));
    });

    expect(exporter.getFinishedSpans()).toHaveLength(0);
  });
});
