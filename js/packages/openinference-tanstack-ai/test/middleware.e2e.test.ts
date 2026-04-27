import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import {
  chat,
  toolDefinition,
  type DefaultMessageMetadataByModality,
  type StreamChunk,
  type TextAdapter,
} from "@tanstack/ai";
import { describe, expect, it } from "vitest";
import { z } from "zod";

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

describe("openInferenceMiddleware e2e", () => {
  it("emits agent and llm spans for a real non-streaming chat() run", async () => {
    const { exporter, tracer } = createTracer();
    const adapter = createFakeTextAdapter([
      {
        type: "TEXT_MESSAGE_CONTENT",
        timestamp: Date.now(),
        messageId: "msg-1",
        delta: "OpenInference makes LLM traces easier to inspect.",
        content: "OpenInference makes LLM traces easier to inspect.",
      },
      {
        type: "RUN_FINISHED",
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
    const llmSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.LLM,
    );

    expect(agentSpan?.name).toBe("ai.chat");
    expect(llmSpan?.name).toBe("ai.llm 1");
    expect(agentSpan?.attributes["output.value"]).toBe(
      "OpenInference makes LLM traces easier to inspect.",
    );
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
            type: "TOOL_CALL_START",
            timestamp: Date.now(),
            toolCallId: "tool-1",
            toolName: "get_weather",
          };
          yield {
            type: "TOOL_CALL_ARGS",
            timestamp: Date.now(),
            toolCallId: "tool-1",
            delta: '{"city":"Boston"}',
            args: '{"city":"Boston"}',
          };
          yield {
            type: "TOOL_CALL_END",
            timestamp: Date.now(),
            toolCallId: "tool-1",
            toolName: "get_weather",
            input: { city: "Boston" },
          };
          yield {
            type: "RUN_FINISHED",
            timestamp: Date.now(),
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
          type: "TEXT_MESSAGE_CONTENT",
          timestamp: Date.now(),
          messageId: "msg-2",
          delta: "It is sunny in Boston.",
          content: "It is sunny in Boston.",
        };
        yield {
          type: "RUN_FINISHED",
          timestamp: Date.now(),
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
    const llmSpan = spans.find(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.LLM,
    );

    expect(agentSpan).toBeDefined();
    expect(llmSpan).toBeDefined();
    const toolSpans = spans.filter(
      (span) =>
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.TOOL,
    );
    expect(toolSpans).toHaveLength(1);
    expect(agentSpan?.name).toBe("ai.chat");
    expect(toolSpans[0]?.name).toBe("ai.tool get_weather");
    expect(toolSpans[0]?.attributes[SemanticConventions.TOOL_NAME]).toBe("get_weather");
    expect(
      llmSpan?.attributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.0.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
      ],
    ).toBe("get_weather");
    expect(
      spans.filter(
        (span) =>
          span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
          OpenInferenceSpanKind.LLM,
      ),
    ).toHaveLength(2);
  });
});
