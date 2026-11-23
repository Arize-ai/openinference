import {
  InMemorySpanExporter,
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { TraceConfigOptions } from "../../src";
import { OITracer, OPENINFERENCE_ENABLE_GENAI_SEMCONV, REDACTED_VALUE } from "../../src";

describe("GenAI semantic conventions export", () => {
  const memoryExporter = new InMemorySpanExporter();
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });

  const createTracer = (traceConfig?: TraceConfigOptions) =>
    new OITracer({ tracer: tracerProvider.getTracer("test"), traceConfig });

  const getGenAIAttributeKeys = () =>
    Object.keys(memoryExporter.getFinishedSpans()[0]?.attributes ?? {}).filter((key) =>
      key.startsWith("gen_ai."),
    );

  const llmAttributes = {
    "openinference.span.kind": "LLM",
    "llm.model_name": "gpt-4o",
    "llm.provider": "openai",
    "llm.input_messages.0.message.role": "user",
    "llm.input_messages.0.message.content": "secret question",
  };

  beforeEach(() => {
    memoryExporter.reset();
  });
  afterEach(() => {
    delete process.env[OPENINFERENCE_ENABLE_GENAI_SEMCONV];
  });

  it("should not emit gen_ai attributes by default", () => {
    const span = createTracer().startSpan("llm", { attributes: llmAttributes });
    span.end();
    expect(getGenAIAttributeKeys()).toEqual([]);
  });

  it("should emit gen_ai attributes at span end when enabled via trace config", () => {
    const span = createTracer({ enableGenAISemconv: true }).startSpan("llm", {
      attributes: llmAttributes,
    });
    span.setAttribute("llm.token_count.prompt", 3);
    span.setAttributes({ "llm.token_count.completion": 4 });
    span.end();

    const [finishedSpan] = memoryExporter.getFinishedSpans();
    expect(finishedSpan?.attributes).toMatchObject({
      ...llmAttributes,
      "gen_ai.operation.name": "chat",
      "gen_ai.provider.name": "openai",
      "gen_ai.request.model": "gpt-4o",
      "gen_ai.usage.input_tokens": 3,
      "gen_ai.usage.output_tokens": 4,
      "gen_ai.input.messages": JSON.stringify([
        { role: "user", parts: [{ type: "text", content: "secret question" }] },
      ]),
    });
  });

  it("should emit gen_ai attributes for active spans", () => {
    createTracer({ enableGenAISemconv: true }).startActiveSpan(
      "tool",
      { attributes: { "openinference.span.kind": "TOOL", "tool.name": "search" } },
      (span) => span.end(),
    );
    expect(memoryExporter.getFinishedSpans()[0]?.attributes).toMatchObject({
      "gen_ai.operation.name": "execute_tool",
      "gen_ai.tool.name": "search",
    });
  });

  it("should be enabled by the OPENINFERENCE_ENABLE_GENAI_SEMCONV environment variable", () => {
    process.env[OPENINFERENCE_ENABLE_GENAI_SEMCONV] = "true";
    const span = createTracer().startSpan("llm", { attributes: llmAttributes });
    span.end();
    expect(memoryExporter.getFinishedSpans()[0]?.attributes).toMatchObject({
      "gen_ai.operation.name": "chat",
    });
  });

  it("should let the trace config override the environment variable", () => {
    process.env[OPENINFERENCE_ENABLE_GENAI_SEMCONV] = "true";
    const span = createTracer({ enableGenAISemconv: false }).startSpan("llm", {
      attributes: llmAttributes,
    });
    span.end();
    expect(getGenAIAttributeKeys()).toEqual([]);
  });

  it("should not overwrite gen_ai attributes that were set explicitly", () => {
    const span = createTracer({ enableGenAISemconv: true }).startSpan("llm", {
      attributes: llmAttributes,
    });
    span.setAttribute("gen_ai.request.model", "explicit-model");
    span.setAttribute("gen_ai.operation.name", "plan");
    span.end();
    expect(memoryExporter.getFinishedSpans()[0]?.attributes).toMatchObject({
      "gen_ai.request.model": "explicit-model",
      "gen_ai.operation.name": "plan",
      "gen_ai.provider.name": "openai",
    });
  });

  it("should derive gen_ai attributes from masked values", () => {
    const span = createTracer({ enableGenAISemconv: true, hideInputText: true }).startSpan("llm", {
      attributes: llmAttributes,
    });
    span.end();
    const attributes = memoryExporter.getFinishedSpans()[0]?.attributes ?? {};
    expect(attributes["gen_ai.input.messages"]).toBe(
      JSON.stringify([{ role: "user", parts: [{ type: "text", content: REDACTED_VALUE }] }]),
    );
    expect(JSON.stringify(attributes)).not.toContain("secret question");
  });

  it("should not derive messages that masking removed", () => {
    const span = createTracer({ enableGenAISemconv: true, hideInputs: true }).startSpan("llm", {
      attributes: llmAttributes,
    });
    span.end();
    const attributes = memoryExporter.getFinishedSpans()[0]?.attributes ?? {};
    expect(attributes).not.toHaveProperty("gen_ai.input.messages");
    expect(JSON.stringify(attributes)).not.toContain("secret question");
  });

  it("should tolerate end being called more than once", () => {
    const span = createTracer({ enableGenAISemconv: true }).startSpan("llm", {
      attributes: llmAttributes,
    });
    span.end();
    expect(() => span.end()).not.toThrow();
    expect(memoryExporter.getFinishedSpans()).toHaveLength(1);
  });
});
