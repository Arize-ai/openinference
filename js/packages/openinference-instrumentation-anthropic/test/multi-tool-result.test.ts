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
});
