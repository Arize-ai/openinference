import Anthropic from "@anthropic-ai/sdk";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { AnthropicInstrumentation } from "../src/instrumentation";
import { vcrFetch } from "./helpers/vcr";

const memoryExporter = new InMemorySpanExporter();

describe("AnthropicInstrumentation - multiple tool_result blocks", () => {
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  tracerProvider.register();
  const instrumentation = new AnthropicInstrumentation({ tracerProvider });
  instrumentation.disable();
  instrumentation._modules[0].moduleExports = Anthropic;

  beforeAll(() => {
    instrumentation.enable();
  });

  beforeEach(() => {
    memoryExporter.reset();
  });

  afterEach(() => {
    instrumentation.disable();
    instrumentation.enable();
  });

  it("records each tool_result block of a message as its own tool message", async () => {
    const client = new Anthropic({
      apiKey: "fake-api-key",
      fetch: vcrFetch("multi-tool-result-non-streaming"),
    });

    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      messages: [
        { role: "user", content: "Add 2+2 and get the weather in Paris." },
        {
          role: "assistant",
          content: [
            {
              type: "tool_use",
              id: "toolu_01AAA",
              name: "calculator",
              input: { expression: "2+2" },
            },
            {
              type: "tool_use",
              id: "toolu_02BBB",
              name: "get_weather",
              input: { city: "Paris" },
            },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "toolu_01AAA", content: "4" },
            {
              type: "tool_result",
              tool_use_id: "toolu_02BBB",
              content: "sunny, 22C",
            },
          ],
        },
      ],
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const attributes = { ...spans[0].attributes };

    const prefix2 = `${SemanticConventions.LLM_INPUT_MESSAGES}.2.`;
    expect(attributes[`${prefix2}${SemanticConventions.MESSAGE_ROLE}`]).toBe("tool");
    expect(attributes[`${prefix2}${SemanticConventions.MESSAGE_TOOL_CALL_ID}`]).toBe("toolu_01AAA");
    expect(attributes[`${prefix2}${SemanticConventions.MESSAGE_CONTENT}`]).toBe("4");

    const prefix3 = `${SemanticConventions.LLM_INPUT_MESSAGES}.3.`;
    expect(attributes[`${prefix3}${SemanticConventions.MESSAGE_ROLE}`]).toBe("tool");
    expect(attributes[`${prefix3}${SemanticConventions.MESSAGE_TOOL_CALL_ID}`]).toBe("toolu_02BBB");
    expect(attributes[`${prefix3}${SemanticConventions.MESSAGE_CONTENT}`]).toBe("sunny, 22C");
  });

  const assistantToolUseTurn: Anthropic.Messages.MessageParam = {
    role: "assistant",
    content: [
      { type: "tool_use", id: "toolu_01AAA", name: "calculator", input: { expression: "2+2" } },
      { type: "tool_use", id: "toolu_02BBB", name: "get_weather", input: { city: "Paris" } },
    ],
  };

  it("records tool messages before the rest of a mixed message, with contiguous content indices", async () => {
    const client = new Anthropic({
      apiKey: "fake-api-key",
      fetch: vcrFetch("multi-tool-result-non-streaming"),
    });

    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      messages: [
        { role: "user", content: "Add 2+2 and get the weather in Paris." },
        assistantToolUseTurn,
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "toolu_01AAA", content: "4" },
            { type: "tool_result", tool_use_id: "toolu_02BBB", content: "sunny, 22C" },
            { type: "text", text: "Now answer in one sentence." },
          ],
        },
      ],
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const attributes = { ...spans[0].attributes };

    const prefix2 = `${SemanticConventions.LLM_INPUT_MESSAGES}.2.`;
    expect(attributes[`${prefix2}${SemanticConventions.MESSAGE_ROLE}`]).toBe("tool");
    expect(attributes[`${prefix2}${SemanticConventions.MESSAGE_TOOL_CALL_ID}`]).toBe("toolu_01AAA");
    expect(attributes[`${prefix2}${SemanticConventions.MESSAGE_CONTENT}`]).toBe("4");

    const prefix3 = `${SemanticConventions.LLM_INPUT_MESSAGES}.3.`;
    expect(attributes[`${prefix3}${SemanticConventions.MESSAGE_ROLE}`]).toBe("tool");
    expect(attributes[`${prefix3}${SemanticConventions.MESSAGE_TOOL_CALL_ID}`]).toBe("toolu_02BBB");
    expect(attributes[`${prefix3}${SemanticConventions.MESSAGE_CONTENT}`]).toBe("sunny, 22C");

    const prefix4 = `${SemanticConventions.LLM_INPUT_MESSAGES}.4.`;
    expect(attributes[`${prefix4}${SemanticConventions.MESSAGE_ROLE}`]).toBe("user");
    expect(attributes[`${prefix4}${SemanticConventions.MESSAGE_TOOL_CALL_ID}`]).toBeUndefined();
    expect(
      attributes[
        `${prefix4}${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TYPE}`
      ],
    ).toBe("text");
    expect(
      attributes[
        `${prefix4}${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TEXT}`
      ],
    ).toBe("Now answer in one sentence.");
    expect(
      Object.keys(attributes).filter((key) =>
        key.startsWith(`${prefix4}${SemanticConventions.MESSAGE_CONTENTS}.`),
      ),
    ).toHaveLength(2);
    expect(
      attributes[`${SemanticConventions.LLM_INPUT_MESSAGES}.5.${SemanticConventions.MESSAGE_ROLE}`],
    ).toBeUndefined();
  });

  it("records a single tool_result message as one tool message", async () => {
    const client = new Anthropic({
      apiKey: "fake-api-key",
      fetch: vcrFetch("multi-tool-result-non-streaming"),
    });

    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      messages: [
        { role: "user", content: "Add 2+2." },
        {
          role: "assistant",
          content: [
            {
              type: "tool_use",
              id: "toolu_01AAA",
              name: "calculator",
              input: { expression: "2+2" },
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "toolu_01AAA",
              content: [{ type: "text", text: "4" }],
            },
          ],
        },
      ],
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const attributes = { ...spans[0].attributes };

    const prefix2 = `${SemanticConventions.LLM_INPUT_MESSAGES}.2.`;
    expect(attributes[`${prefix2}${SemanticConventions.MESSAGE_ROLE}`]).toBe("tool");
    expect(attributes[`${prefix2}${SemanticConventions.MESSAGE_TOOL_CALL_ID}`]).toBe("toolu_01AAA");
    expect(attributes[`${prefix2}${SemanticConventions.MESSAGE_CONTENT}`]).toBe(
      JSON.stringify([{ type: "text", text: "4" }]),
    );
    expect(
      attributes[`${SemanticConventions.LLM_INPUT_MESSAGES}.3.${SemanticConventions.MESSAGE_ROLE}`],
    ).toBeUndefined();
  });
});
