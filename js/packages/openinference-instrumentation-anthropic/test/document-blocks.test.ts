import Anthropic from "@anthropic-ai/sdk";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { AnthropicInstrumentation } from "../src/instrumentation";

const {
  LLM_INPUT_MESSAGES,
  MESSAGE_ROLE,
  MESSAGE_CONTENTS,
  MESSAGE_CONTENT_TYPE,
  MESSAGE_CONTENT_TEXT,
} = SemanticConventions;

const memoryExporter = new InMemorySpanExporter();

const responseBody = {
  id: "msg_doc123",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-6",
  content: [{ type: "text", text: "This is a PDF about onboarding." }],
  stop_reason: "end_turn",
  stop_sequence: null,
  usage: { input_tokens: 25, output_tokens: 7 },
};

const stubFetch = (async () =>
  new Response(JSON.stringify(responseBody), {
    status: 200,
    headers: { "content-type": "application/json" },
  })) as typeof fetch;

describe("AnthropicInstrumentation - document content blocks", () => {
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

  it("records a base64 document block with its media type, without the bytes", async () => {
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: stubFetch });
    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 64,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "document",
              source: {
                type: "base64",
                media_type: "application/pdf",
                data: "JVBERi0xLjQKJdPr6eEKMSAwIG9iago=",
              },
            },
            { type: "text", text: "Summarize this PDF" },
          ],
        },
      ],
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const attributes = { ...spans[0].attributes };
    expect(attributes[`${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`]).toBe("user");
    expect(
      attributes[`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`],
    ).toBe("document");
    expect(
      attributes[
        `${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.message_content.document.media_type`
      ],
    ).toBe("application/pdf");
    expect(
      attributes[`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TYPE}`],
    ).toBe("text");
    expect(
      attributes[`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TEXT}`],
    ).toBe("Summarize this PDF");
  });

  it("records a plain-text document block as text content", async () => {
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: stubFetch });
    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 64,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "document",
              source: {
                type: "text",
                media_type: "text/plain",
                data: "Q3 revenue grew 12% year over year.",
              },
              title: "Q3 report",
            },
          ],
        },
      ],
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const attributes = { ...spans[0].attributes };
    expect(
      attributes[`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`],
    ).toBe("text");
    expect(
      attributes[`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TEXT}`],
    ).toBe("Q3 revenue grew 12% year over year.");
  });
});
