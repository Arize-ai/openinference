import type { ConverseResponse, InvokeModelResponse } from "@aws-sdk/client-bedrock-runtime";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import { LLM_FINISH_REASON, LLMSystem } from "@arizeai/openinference-semantic-conventions";

import { extractConverseResponseAttributes } from "../src/attributes/converse-response-attributes";
import { extractInvokeModelResponseAttributes } from "../src/attributes/invoke-model-response-attributes";
import { consumeBedrockStreamChunks } from "../src/attributes/invoke-model-streaming-response-attributes";

describe("finish reason", () => {
  let exporter: InMemorySpanExporter;
  let provider: NodeTracerProvider;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
  });

  afterEach(async () => {
    await provider.shutdown();
  });

  it.each([
    {
      providerName: "Nova",
      modelType: LLMSystem.AMAZON,
      body: { stopReason: "end_turn" },
      expected: "end_turn",
    },
    {
      providerName: "Anthropic",
      modelType: LLMSystem.ANTHROPIC,
      body: { stop_reason: "tool_use" },
      expected: "tool_use",
    },
    {
      providerName: "Meta",
      modelType: LLMSystem.META,
      body: { stop_reason: "length" },
      expected: "length",
    },
    {
      providerName: "Cohere Command R",
      modelType: LLMSystem.COHERE,
      body: { finish_reason: "MAX_TOKENS" },
      expected: "MAX_TOKENS",
    },
    {
      providerName: "Cohere Command",
      modelType: LLMSystem.COHERE,
      body: { generations: [{ finish_reason: "COMPLETE" }] },
      expected: "COMPLETE",
    },
    {
      providerName: "Mistral",
      modelType: LLMSystem.MISTRALAI,
      body: { outputs: [{ stop_reason: "stop" }] },
      expected: "stop",
    },
    {
      providerName: "Titan",
      modelType: LLMSystem.AMAZON,
      body: { results: [{ completionReason: "FINISH" }] },
      expected: "FINISH",
    },
    {
      providerName: "AI21 Jurassic",
      modelType: LLMSystem.AI21,
      body: { completions: [{ finishReason: { reason: "length" } }] },
      expected: "length",
    },
    {
      providerName: "AI21 string reason",
      modelType: LLMSystem.AI21,
      body: { completions: [{ finishReason: "stop" }] },
      expected: "stop",
    },
    {
      providerName: "AI21 Jamba",
      modelType: LLMSystem.AI21,
      body: { choices: [{ finish_reason: "stop" }] },
      expected: "stop",
    },
    {
      providerName: "OpenAI-compatible",
      modelType: LLMSystem.OPENAI,
      body: { choices: [{ finish_reason: "tool_calls" }, { finish_reason: "length" }] },
      expected: "tool_calls",
    },
    {
      providerName: "top-level camelCase",
      modelType: LLMSystem.AI21,
      body: { finishReason: "stop" },
      expected: "stop",
    },
    {
      providerName: "top-level Titan",
      modelType: LLMSystem.AMAZON,
      body: { completionReason: "CONTENT_FILTERED" },
      expected: "CONTENT_FILTERED",
    },
  ])("records the native InvokeModel reason for $providerName", ({ modelType, body, expected }) => {
    const span = provider.getTracer("test").startSpan("invoke_model");
    const response: InvokeModelResponse = {
      body: Buffer.from(JSON.stringify(body)),
      contentType: "application/json",
    };
    extractInvokeModelResponseAttributes({ span, response, modelType });
    span.end();
    expect(exporter.getFinishedSpans()[0].attributes[LLM_FINISH_REASON]).toBe(expected);
  });

  it.each([
    {},
    { stop_reason: null },
    { stopReason: "" },
    { finish_reason: 42 },
    { finishReason: {} },
    { generations: [] },
    { outputs: [null] },
    { results: "invalid" },
    { choices: [{ finish_reason: null }, { finish_reason: "stop" }] },
    { completions: [{ finishReason: { reason: 42 } }] },
  ])("omits unavailable or malformed InvokeModel reasons: %j", (body) => {
    const span = provider.getTracer("test").startSpan("invoke_model");
    extractInvokeModelResponseAttributes({
      span,
      response: { body: Buffer.from(JSON.stringify(body)), contentType: "application/json" },
      modelType: LLMSystem.AMAZON,
    });
    span.end();
    expect(exporter.getFinishedSpans()[0].attributes).not.toHaveProperty(LLM_FINISH_REASON);
  });

  it.each(["end_turn", "max_tokens", "tool_use", "content_filtered", undefined] as const)(
    "records the Converse reason %s and retains the legacy attribute",
    (stopReason) => {
      const span = provider.getTracer("test").startSpan("converse");
      // Exercise an omitted reason even though the SDK declares it required.
      const response = {
        output: { message: { role: "assistant", content: [{ text: "Hello" }] } },
        ...(stopReason !== undefined && { stopReason }),
      } as ConverseResponse;
      extractConverseResponseAttributes({ span, response });
      span.end();
      const attributes = exporter.getFinishedSpans()[0].attributes;
      expect(attributes[LLM_FINISH_REASON]).toBe(stopReason);
      expect(attributes["llm.stop_reason"]).toBe(stopReason);
      if (stopReason === undefined) expect(attributes).not.toHaveProperty(LLM_FINISH_REASON);
    },
  );

  it.each([
    {
      name: "Nova messageStop",
      modelType: LLMSystem.AMAZON,
      event: { messageStop: { stopReason: "max_tokens" } },
      expected: "max_tokens",
    },
    {
      name: "Anthropic message_delta without usage",
      modelType: LLMSystem.ANTHROPIC,
      event: { type: "message_delta", delta: { stop_reason: "tool_use" } },
      expected: "tool_use",
    },
    {
      name: "Anthropic message_start",
      modelType: LLMSystem.ANTHROPIC,
      event: { type: "message_start", message: { stop_reason: "end_turn" } },
      expected: "end_turn",
    },
    {
      name: "Meta",
      modelType: LLMSystem.META,
      event: { stop_reason: "length" },
      expected: "length",
    },
    {
      name: "Titan",
      modelType: LLMSystem.AMAZON,
      event: { completionReason: "FINISH" },
      expected: "FINISH",
    },
    {
      name: "missing reason",
      modelType: LLMSystem.ANTHROPIC,
      event: { type: "message_delta", delta: { stop_reason: null } },
      expected: undefined,
    },
  ])(
    "retains the streamed reason for $name through later chunks",
    async ({ modelType, event, expected }) => {
      const span = provider.getTracer("test").startSpan("invoke_model_stream");
      async function* stream() {
        for (const data of [
          event,
          { type: "message_delta", delta: { stop_reason: null }, usage: { output_tokens: 3 } },
          { metadata: { usage: { inputTokens: 2, outputTokens: 3 } } },
        ]) {
          yield { chunk: { bytes: Buffer.from(JSON.stringify(data)) } };
        }
      }
      await consumeBedrockStreamChunks({ span, stream: stream(), modelType });
      span.end();
      const attributes = exporter.getFinishedSpans()[0].attributes;
      expect(attributes[LLM_FINISH_REASON]).toBe(expected);
      if (expected === undefined) expect(attributes).not.toHaveProperty(LLM_FINISH_REASON);
    },
  );
});
