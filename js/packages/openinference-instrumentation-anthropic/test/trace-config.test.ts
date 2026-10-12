import Anthropic from "@anthropic-ai/sdk";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { TraceConfigOptions } from "@arizeai/openinference-core";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { AnthropicInstrumentation } from "../src/instrumentation";
import { jsonFetch } from "./helpers/sse";
import { waitForSpans as waitForSpansOn } from "./helpers/waitForSpans";

const {
  LLM_INPUT_MESSAGES,
  MESSAGE_CONTENT_TEXT,
  MESSAGE_CONTENT_TYPE,
  MESSAGE_CONTENTS,
  METADATA,
} = SemanticConventions;

const memoryExporter = new InMemorySpanExporter();
const waitForSpans = (count: number) => waitForSpansOn(memoryExporter, count);
const tracerProvider = new NodeTracerProvider({
  spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
});
tracerProvider.register();

const citation = {
  type: "char_location",
  cited_text: "SECRET CITED TEXT",
  document_index: 0,
  document_title: null,
  start_char_index: 0,
  end_char_index: 5,
};

const responseMessage = {
  id: "msg_config",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-6",
  content: [{ type: "text", text: "Answer", citations: [citation] }],
  stop_reason: "refusal",
  stop_sequence: null,
  stop_details: { type: "refusal", category: "cyber", explanation: "SECRET EXPLANATION" },
  usage: { input_tokens: 1, output_tokens: 1 },
};

const request = {
  model: "claude-sonnet-4-6",
  max_tokens: 100,
  messages: [
    {
      role: "user" as const,
      content: [
        {
          type: "document" as const,
          source: {
            type: "base64" as const,
            media_type: "application/pdf" as const,
            data: "JVBERi0",
          },
        },
        {
          type: "tool_result" as const,
          tool_use_id: "toolu_1",
          content: "ok",
        },
      ],
    },
    {
      role: "assistant" as const,
      content: [
        {
          type: "web_fetch_tool_result",
          tool_use_id: "srvtoolu_1",
          content: {
            type: "web_fetch_result",
            url: "https://example.com/cat",
            content: {
              type: "document",
              source: { type: "base64", media_type: "application/pdf", data: "JVBERi0" },
            },
            nested: [
              { type: "image", source: { type: "url", url: "https://example.com/cat.png" } },
            ],
          },
        } as unknown as Anthropic.Beta.BetaWebFetchToolResultBlockParam,
      ],
    },
  ],
};

const inputText = (messageIndex: number, index: number) =>
  `${LLM_INPUT_MESSAGES}.${messageIndex}.${MESSAGE_CONTENTS}.${index}.${MESSAGE_CONTENT_TEXT}`;

describe("AnthropicInstrumentation - trace config", () => {
  let instrumentation: AnthropicInstrumentation | undefined;

  function instrument(traceConfig: TraceConfigOptions) {
    instrumentation = new AnthropicInstrumentation({ tracerProvider, traceConfig });
    instrumentation.disable();
    instrumentation._modules[0].moduleExports = Anthropic;
    instrumentation.enable();
  }

  beforeEach(() => {
    memoryExporter.reset();
  });

  afterEach(() => {
    instrumentation?.disable();
  });

  async function traceRequest() {
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: jsonFetch(responseMessage) });
    await client.beta.messages.create(request);
    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    return span.attributes;
  }

  it("keeps cited text and the refusal explanation when nothing is hidden", async () => {
    instrument({});
    const metadata = JSON.parse((await traceRequest())[METADATA] as string);
    expect(metadata.citations["0"][0].cited_text).toBe("SECRET CITED TEXT");
    expect(metadata.stop_details.explanation).toBe("SECRET EXPLANATION");
  });

  it.each([{ hideOutputs: true }, { hideOutputText: true }])(
    "drops cited text and the refusal explanation with %o",
    async (traceConfig) => {
      instrument(traceConfig);
      const attributes = await traceRequest();
      expect(attributes[METADATA]).not.toContain("SECRET");
      const metadata = JSON.parse(attributes[METADATA] as string);
      expect(metadata.citations["0"][0]).toEqual({ ...citation, cited_text: undefined });
      expect(metadata.stop_details).toEqual({ type: "refusal", category: "cyber" });
    },
  );

  it("drops cited text with hideInputs, as it quotes the prompt", async () => {
    instrument({ hideInputs: true });
    const metadata = JSON.parse((await traceRequest())[METADATA] as string);
    expect(metadata.citations["0"][0].cited_text).toBeUndefined();
    expect(metadata.stop_details.explanation).toBe("SECRET EXPLANATION");
  });

  it("never encodes base64 data in a block recorded as JSON", async () => {
    instrument({});
    const attributes = await traceRequest();
    expect(
      attributes[`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`],
    ).toBe("document");
    const documentJSON = JSON.parse(attributes[inputText(0, 0)] as string);
    expect(documentJSON.source).toEqual({
      type: "base64",
      media_type: "application/pdf",
      data: "__REDACTED__",
    });
    const resultJSON = attributes[inputText(1, 0)] as string;
    expect(resultJSON).not.toContain("JVBERi0");
    // Without hideInputImages, a nested image is kept.
    expect(resultJSON).toContain("https://example.com/cat.png");
  });

  it("hides nested images of a block recorded as JSON with hideInputImages", async () => {
    instrument({ hideInputImages: true });
    const attributes = await traceRequest();
    const resultJSON = JSON.parse(attributes[inputText(1, 0)] as string);
    expect(resultJSON.content.nested).toEqual([{ type: "image", source: "__REDACTED__" }]);
  });
});
