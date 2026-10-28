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
  LLM_INVOCATION_PARAMETERS,
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

  // Citations name and quote the sources, so they are left out when the input
  // or output messages are hidden, and otherwise only their cited text is when
  // the input or output text is. The refusal explanation is model output.
  it.each<[TraceConfigOptions, boolean, boolean, boolean]>([
    [{}, false, false, false],
    [{ hideInputs: true }, true, true, false],
    [{ hideInputMessages: true }, true, true, false],
    [{ hideInputText: true }, false, true, false],
    [{ hideOutputs: true }, true, true, true],
    [{ hideOutputMessages: true }, true, true, true],
    [{ hideOutputText: true }, false, true, true],
  ])(
    "with %o, hides citations: %s, cited text: %s, the refusal explanation: %s",
    async (traceConfig, hidesCitations, hidesCitedText, hidesExplanation) => {
      instrument(traceConfig);
      const metadata = JSON.parse((await traceRequest())[METADATA] as string);
      if (hidesCitations) {
        expect(metadata.citations).toBeUndefined();
      } else if (hidesCitedText) {
        expect(metadata.citations["0"][0]).toEqual({ ...citation, cited_text: undefined });
      } else {
        expect(metadata.citations["0"][0]).toEqual(citation);
      }
      expect(metadata.stop_details).toEqual(
        hidesExplanation
          ? { type: "refusal", category: "cyber" }
          : { type: "refusal", category: "cyber", explanation: "SECRET EXPLANATION" },
      );
    },
  );

  it("keeps the system prompt out of the invocation parameters, so hideInputs hides it", async () => {
    instrument({ hideInputs: true });
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: jsonFetch(responseMessage) });
    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      system: "SECRET SYSTEM PROMPT",
      messages: [{ role: "user", content: "Hi" }],
    });
    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    expect(JSON.parse(span.attributes[LLM_INVOCATION_PARAMETERS] as string)).toEqual({
      model: "claude-sonnet-4-6",
      max_tokens: 100,
    });
    expect(JSON.stringify(span.attributes)).not.toContain("SECRET SYSTEM PROMPT");
  });

  it("never encodes base64 data in a block recorded as JSON", async () => {
    instrument({});
    const attributes = await traceRequest();
    // The tool_result is recorded as its own tool message ahead of the rest of
    // the first message, so the document is in message 1 and the web fetch
    // result in message 2.
    expect(
      attributes[`${LLM_INPUT_MESSAGES}.1.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`],
    ).toBe("document");
    const documentJSON = JSON.parse(attributes[inputText(1, 0)] as string);
    expect(documentJSON.source).toEqual({
      type: "base64",
      media_type: "application/pdf",
      data: "__REDACTED__",
    });
    const resultJSON = attributes[inputText(2, 0)] as string;
    expect(resultJSON).not.toContain("JVBERi0");
    // Without hideInputImages, a nested image is kept.
    expect(resultJSON).toContain("https://example.com/cat.png");
  });

  it("hides nested images of a block recorded as JSON with hideInputImages", async () => {
    instrument({ hideInputImages: true });
    const attributes = await traceRequest();
    const resultJSON = JSON.parse(attributes[inputText(2, 0)] as string);
    expect(resultJSON.content.nested).toEqual([{ type: "image", source: "__REDACTED__" }]);
  });
});
