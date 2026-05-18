import { context } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import {
  chat,
  EventType,
  toolDefinition,
  type DefaultMessageMetadataByModality,
  type StreamChunk,
  type TextAdapter,
} from "@tanstack/ai";
import { describe, expect, it } from "vitest";
import { z } from "zod";

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
    tracer: provider.getTracer("middleware-e2e-test"),
  };
}

function createFakeTextAdapter(
  chunks: Array<Record<string, unknown>>,
): TextAdapter<string, Record<string, never>, readonly ["text"], DefaultMessageMetadataByModality> {
  return {
    kind: "text",
    name: "fake",
    model: "fake-model",
    "~types": undefined as never,
    async *chatStream() {
      for (const chunk of chunks) {
        yield chunk as StreamChunk;
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

function createFinishedTextAdapter(content = "OpenInference makes LLM traces easier to inspect.") {
  return createFakeTextAdapter([
    {
      type: EventType.TEXT_MESSAGE_CONTENT,
      timestamp: Date.now(),
      messageId: "msg-1",
      delta: content,
      content,
    },
    {
      type: EventType.RUN_FINISHED,
      timestamp: Date.now(),
      runId: "run-1",
      model: "fake-model",
      finishReason: "stop",
      usage: {
        promptTokens: 6,
        completionTokens: 8,
        totalTokens: 14,
      },
    },
  ]);
}

describe("openInferenceMiddleware e2e", () => {
  it("emits agent and llm spans for a real non-streaming chat() run", async () => {
    const { exporter, tracer } = createTracer();
    const adapter = createFinishedTextAdapter();

    const text = await chat({
      adapter,
      stream: false,
      systemPrompts: ["You are a concise technical explainer."],
      messages: [{ role: "user", content: "Explain OpenInference in one sentence." }],
      middleware: [openInferenceMiddleware({ tracer })],
    });

    expect(text).toBe("OpenInference makes LLM traces easier to inspect.");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(2);

    const agentSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.AGENT,
    );
    const llmSpans = spans.filter(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.LLM,
    );
    const llmSpan = llmSpans.find((span) => span.name === "chat fake-model #0");

    expect(agentSpan?.name).toBe("chat fake-model");
    expect(llmSpan?.name).toBe("chat fake-model #0");
    expect(llmSpan?.attributes[SemanticConventions.LLM_MODEL_NAME]).toBe("fake-model");
    expect(
      llmSpan?.attributes[
        `${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`
      ],
    ).toBe("system");
    expect(
      llmSpan?.attributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`
      ],
    ).toBe("OpenInference makes LLM traces easier to inspect.");
  });

  it("emits agent, llm, and tool spans for a real streaming tool loop", async () => {
    const { exporter, tracer } = createTracer();
    let iteration = 0;
    const adapter: TextAdapter<
      string,
      Record<string, never>,
      readonly ["text"],
      DefaultMessageMetadataByModality
    > = {
      kind: "text",
      name: "fake",
      model: "fake-model",
      "~types": undefined as never,
      async *chatStream() {
        iteration += 1;
        if (iteration === 1) {
          yield {
            type: EventType.TOOL_CALL_START,
            timestamp: Date.now(),
            toolCallId: "tool-1",
            toolCallName: "get_weather",
            toolName: "get_weather",
          };
          yield {
            type: EventType.TOOL_CALL_ARGS,
            timestamp: Date.now(),
            toolCallId: "tool-1",
            delta: '{"city":"Boston"}',
            args: '{"city":"Boston"}',
          };
          yield {
            type: EventType.TOOL_CALL_END,
            timestamp: Date.now(),
            toolCallId: "tool-1",
            toolName: "get_weather",
            input: { city: "Boston" },
          };
          yield {
            type: EventType.RUN_FINISHED,
            timestamp: Date.now(),
            threadId: "thread-1",
            runId: "run-1",
            model: "fake-model",
            finishReason: "tool_calls",
            usage: {
              promptTokens: 7,
              completionTokens: 5,
              totalTokens: 12,
            },
          };
          return;
        }

        yield {
          type: EventType.TEXT_MESSAGE_CONTENT,
          timestamp: Date.now(),
          messageId: "msg-2",
          delta: "It is sunny in Boston.",
          content: "It is sunny in Boston.",
        };
        yield {
          type: EventType.RUN_FINISHED,
          timestamp: Date.now(),
          threadId: "thread-1",
          runId: "run-2",
          model: "fake-model",
          finishReason: "stop",
          usage: {
            promptTokens: 9,
            completionTokens: 6,
            totalTokens: 15,
          },
        };
      },
      async structuredOutput() {
        return {
          data: { ok: true },
          rawText: '{"ok":true}',
        };
      },
    };
    const weatherTool = toolDefinition({
      name: "get_weather",
      description: "Get the weather for a city",
      inputSchema: z.object({ city: z.string() }),
      outputSchema: z.object({ forecast: z.string(), temperatureF: z.number() }),
    }).server(async ({ city }) => ({
      forecast: city === "Boston" ? "sunny" : "unknown",
      temperatureF: city === "Boston" ? 70 : 65,
    }));

    const chunks = [] as StreamChunk[];
    for await (const chunk of chat({
      adapter,
      messages: [{ role: "user", content: "What is the weather in Boston?" }],
      tools: [weatherTool],
      middleware: [openInferenceMiddleware({ tracer })],
    })) {
      chunks.push(chunk);
    }

    const chunkTypes = chunks.map((chunk) => chunk.type);
    expect(chunkTypes).toContain("TOOL_CALL_START");
    expect(chunkTypes).toContain("TOOL_CALL_ARGS");
    expect(chunkTypes).toContain("TEXT_MESSAGE_CONTENT");
    expect(chunkTypes.filter((type) => type === "RUN_FINISHED")).toHaveLength(2);

    const spans = exporter.getFinishedSpans();
    const agentSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.AGENT,
    );
    const llmSpans = spans.filter(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.LLM,
    );
    const llmSpan = llmSpans.find((span) => span.name === "chat fake-model #0");

    expect(agentSpan).toBeDefined();
    expect(llmSpan).toBeDefined();
    const toolSpans = spans.filter(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.TOOL,
    );
    expect(toolSpans).toHaveLength(1);
    expect(agentSpan?.name).toBe("chat fake-model");
    expect(toolSpans[0]?.name).toBe("execute_tool get_weather");
    expect(toolSpans[0]?.attributes[SemanticConventions.TOOL_NAME]).toBe("get_weather");
    expect(llmSpan?.attributes["llm.finish_reason"]).toBe("tool_calls");
    expect(llmSpans).toHaveLength(2);
  });

  it("propagates OpenInference context attributes and custom enrichment through a real chat() run", async () => {
    const { exporter, tracer } = createTracer();
    const adapter = createFinishedTextAdapter("Context attributes should land on spans.");
    const activeContext = setTags(
      setMetadata(
        setUser(setSession(context.active(), { sessionId: "session-1" }), { userId: "user-1" }),
        { release: "test" },
      ),
      ["e2e", "tanstack"],
    );

    await context.with(activeContext, async () => {
      await chat({
        adapter,
        stream: false,
        messages: [{ role: "user", content: "Check context attributes." }],
        middleware: [
          openInferenceMiddleware({
            tracer,
            attributeEnricher: () => ({ "test.enriched": "yes" }),
          }),
        ],
      });
    });

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(2);
    for (const span of spans) {
      expect(span.attributes[SemanticConventions.SESSION_ID]).toBe("session-1");
      expect(span.attributes[SemanticConventions.USER_ID]).toBe("user-1");
      expect(span.attributes[SemanticConventions.METADATA]).toBe('{"release":"test"}');
      expect(span.attributes[SemanticConventions.TAG_TAGS]).toBe('["e2e","tanstack"]');
      expect(span.attributes["test.enriched"]).toBe("yes");
    }
  });

  it("applies traceConfig redaction during a real chat() run", async () => {
    const { exporter, tracer } = createTracer();
    const adapter = createFinishedTextAdapter("secret output");

    await chat({
      adapter,
      stream: false,
      messages: [{ role: "user", content: "secret input" }],
      middleware: [
        openInferenceMiddleware({
          tracer,
          traceConfig: { hideInputText: true, hideOutputText: true },
        }),
      ],
    });

    const llmSpan = exporter
      .getFinishedSpans()
      .find(
        (span) =>
          span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
          OpenInferenceSpanKind.LLM,
      );

    expect(llmSpan?.attributes["llm.input_messages.0.message.content"]).toBe("__REDACTED__");
    expect(
      llmSpan?.attributes["llm.input_messages.0.message.contents.0.message_content.text"],
    ).toBe("__REDACTED__");
    expect(llmSpan?.attributes["llm.output_messages.0.message.content"]).toBe("__REDACTED__");
    expect(
      llmSpan?.attributes["llm.output_messages.0.message.contents.0.message_content.text"],
    ).toBe("__REDACTED__");
  });

  it("does not over-redact output text when only input text is hidden", async () => {
    const { exporter, tracer } = createTracer();
    const adapter = createFinishedTextAdapter("visible output");

    await chat({
      adapter,
      stream: false,
      messages: [{ role: "user", content: "secret input" }],
      middleware: [
        openInferenceMiddleware({
          tracer,
          traceConfig: { hideInputText: true },
        }),
      ],
    });

    const llmSpan = exporter
      .getFinishedSpans()
      .find(
        (span) =>
          span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
          OpenInferenceSpanKind.LLM,
      );

    expect(llmSpan?.attributes["llm.input_messages.0.message.content"]).toBe("__REDACTED__");
    expect(llmSpan?.attributes["llm.output_messages.0.message.content"]).toBe("visible output");
  });

  it("suppresses spans for a real chat() run when tracing is suppressed", async () => {
    const { exporter, tracer } = createTracer();
    const adapter = createFinishedTextAdapter("This should not be traced.");

    await context.with(suppressTracing(context.active()), async () => {
      await chat({
        adapter,
        stream: false,
        messages: [{ role: "user", content: "Do not trace this." }],
        middleware: [openInferenceMiddleware({ tracer })],
      });
    });

    expect(exporter.getFinishedSpans()).toHaveLength(0);
  });
});
