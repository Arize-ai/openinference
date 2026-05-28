import { context } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { EventType, type ChatMiddlewareConfig, type ChatMiddlewareContext } from "@tanstack/ai";
import { describe, expect, it } from "vitest";

import { setSession } from "@arizeai/openinference-core";
import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { convertTanStackAISpanToOpenInference, openInferenceMiddleware } from "../src";

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
    phase: "beforeModel",
    iteration: 0,
    chunkIndex: 0,
    abort: () => {},
    context: undefined,
    defer: () => {},
    provider: "openai",
    model: "gpt-4o-mini",
    source: "server",
    streaming: true,
    systemPrompts: ["Be concise"],
    toolNames: [],
    options: { temperature: 0.2 },
    modelOptions: {},
    messageCount: 1,
    hasTools: false,
    currentMessageId: null,
    accumulatedContent: "",
    messages: [{ role: "user", content: "What is OpenInference?" }],
    createId: (prefix: string) => `${prefix}-1`,
    ...overrides,
  };
}

function createConfig(overrides: Partial<ChatMiddlewareConfig> = {}): ChatMiddlewareConfig {
  return {
    messages: [{ role: "user", content: "What is OpenInference?" }],
    systemPrompts: ["Be concise"],
    tools: [],
    temperature: 0.2,
    modelOptions: {},
    ...overrides,
  };
}

describe("openInferenceMiddleware", () => {
  it("wraps TanStack OTEL spans with OpenInference attributes", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });
    const ctx = createContext();

    await middleware.onStart?.(ctx);
    await middleware.onConfig?.(ctx, createConfig());
    await middleware.onChunk?.(ctx, {
      type: EventType.TEXT_MESSAGE_CONTENT,
      timestamp: Date.now(),
      messageId: "msg-1",
      delta: "OpenInference traces AI applications.",
      content: "OpenInference traces AI applications.",
    });
    await middleware.onChunk?.(ctx, {
      type: EventType.RUN_FINISHED,
      timestamp: Date.now(),
      threadId: "thread-1",
      runId: "run-1",
      finishReason: "stop",
      usage: { promptTokens: 3, completionTokens: 4, totalTokens: 7 },
    });
    await middleware.onFinish?.(ctx, {
      finishReason: "stop",
      duration: 10,
      content: "OpenInference traces AI applications.",
      usage: { promptTokens: 3, completionTokens: 4, totalTokens: 7 },
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

    expect(agentSpan).toBeDefined();
    expect(llmSpan).toBeDefined();
    expect(agentSpan?.attributes[SemanticConventions.AGENT_NAME]).toBe("chat gpt-4o-mini");
    expect(llmSpan?.attributes[SemanticConventions.LLM_SYSTEM]).toBe("openai");
    expect(llmSpan?.attributes[SemanticConventions.LLM_MODEL_NAME]).toBe("gpt-4o-mini");
    expect(llmSpan?.attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL]).toBe(7);
    expect(llmSpan?.attributes["llm.finish_reason"]).toBe("stop");
    expect(llmSpan?.attributes["llm.input_messages.0.message.role"]).toBe("system");
    expect(llmSpan?.attributes["llm.input_messages.1.message.role"]).toBe("user");
    expect(llmSpan?.attributes["llm.output_messages.0.message.content"]).toBe(
      "OpenInference traces AI applications.",
    );
  });

  it("propagates OpenInference context attributes", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });
    const ctx = createContext();

    await context.with(setSession(context.active(), { sessionId: "session-1" }), async () => {
      await middleware.onStart?.(ctx);
      await middleware.onFinish?.(ctx, { finishReason: "stop", duration: 1, content: "ok" });
    });

    expect(exporter.getFinishedSpans()[0]?.attributes[SemanticConventions.SESSION_ID]).toBe(
      "session-1",
    );
  });

  it("respects traceConfig redaction", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({
      tracer,
      traceConfig: { hideInputText: true, hideOutputText: true },
    });
    const ctx = createContext();

    await middleware.onStart?.(ctx);
    await middleware.onConfig?.(ctx, createConfig());
    await middleware.onChunk?.(ctx, {
      type: EventType.TEXT_MESSAGE_CONTENT,
      timestamp: Date.now(),
      messageId: "msg-1",
      delta: "secret",
      content: "secret",
    });
    await middleware.onChunk?.(ctx, {
      type: EventType.RUN_FINISHED,
      timestamp: Date.now(),
      threadId: "thread-1",
      runId: "run-1",
      finishReason: "stop",
    });
    await middleware.onFinish?.(ctx, { finishReason: "stop", duration: 1, content: "secret" });

    const llmSpan = exporter
      .getFinishedSpans()
      .find(
        (span) =>
          span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
          OpenInferenceSpanKind.LLM,
      );

    expect(llmSpan?.attributes["llm.input_messages.0.message.content"]).toBe("__REDACTED__");
    expect(llmSpan?.attributes["llm.output_messages.0.message.content"]).toBe("__REDACTED__");
  });

  it("respects suppressed tracing", async () => {
    const { exporter, tracer } = createTracer();
    const middleware = openInferenceMiddleware({ tracer });

    await context.with(suppressTracing(context.active()), async () => {
      await middleware.onStart?.(createContext());
    });

    expect(exporter.getFinishedSpans()).toHaveLength(0);
  });
});

describe("convertTanStackAISpanToOpenInference", () => {
  it("runs custom span kind resolvers after the TanStack root-span default is computed", () => {
    const attributes = convertTanStackAISpanToOpenInference(
      {
        name: "chat gpt-4o",
        attributes: {
          "gen_ai.request.model": "gpt-4o",
          "tanstack.ai.iterations": 2,
        },
      },
      {
        spanKindResolver: ({ defaultKind }) => defaultKind,
      },
    );

    expect(attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
      OpenInferenceSpanKind.AGENT,
    );
  });
});
