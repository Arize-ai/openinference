import Anthropic from "@anthropic-ai/sdk";
import { context } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

import { setMetadata } from "@arizeai/openinference-core";
import { MimeType, SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { AnthropicInstrumentation } from "../src/instrumentation";
import type { StreamEvent } from "./helpers/sse";
import { jsonFetch, messageDelta, messageStart, sseFetch } from "./helpers/sse";
import { waitForSpans as waitForSpansOn } from "./helpers/waitForSpans";

const {
  IMAGE_URL,
  LLM_INPUT_MESSAGES,
  LLM_OUTPUT_MESSAGES,
  LLM_TOOLS,
  MESSAGE_CONTENT,
  MESSAGE_CONTENT_ID,
  MESSAGE_CONTENT_IMAGE,
  MESSAGE_CONTENT_SIGNATURE,
  MESSAGE_CONTENT_TEXT,
  MESSAGE_CONTENT_TYPE,
  MESSAGE_CONTENTS,
  MESSAGE_ROLE,
  MESSAGE_TOOL_CALLS,
  METADATA,
  OUTPUT_MIME_TYPE,
  OUTPUT_VALUE,
  TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
  TOOL_CALL_FUNCTION_NAME,
  TOOL_CALL_ID,
  TOOL_DESCRIPTION,
  TOOL_JSON_SCHEMA,
  TOOL_NAME,
} = SemanticConventions;

const memoryExporter = new InMemorySpanExporter();
const waitForSpans = (count: number) => waitForSpansOn(memoryExporter, count);

const output = (postfix: string) => `${LLM_OUTPUT_MESSAGES}.0.${postfix}`;
const outputContent = (index: number, postfix: string) =>
  output(`${MESSAGE_CONTENTS}.${index}.${postfix}`);
const outputToolCall = (index: number, postfix: string) =>
  output(`${MESSAGE_TOOL_CALLS}.${index}.${postfix}`);
const input = (messageIndex: number, postfix: string) =>
  `${LLM_INPUT_MESSAGES}.${messageIndex}.${postfix}`;
const inputContent = (messageIndex: number, index: number, postfix: string) =>
  input(messageIndex, `${MESSAGE_CONTENTS}.${index}.${postfix}`);

const citation = {
  type: "web_search_result_location",
  cited_text: "Paris is the capital of France.",
  url: "https://example.com/paris",
  title: "Paris",
  encrypted_index: "abc",
};

const webSearchResult = {
  type: "web_search_tool_result",
  tool_use_id: "srvtoolu_1",
  content: [
    {
      type: "web_search_result",
      url: "https://example.com/paris",
      title: "Paris",
      encrypted_content: "xyz",
      page_age: null,
    },
  ],
};

const mcpToolResult = {
  type: "mcp_tool_result",
  tool_use_id: "mcptoolu_1",
  is_error: false,
  content: [{ type: "text", text: "3 open issues" }],
};

/** A response with every kind of block, in an order that interleaves them. */
const contentBlocks = [
  { type: "compaction", content: "Summary of earlier turns", encrypted_content: null },
  {
    type: "server_tool_use",
    id: "srvtoolu_1",
    name: "web_search",
    input: { query: "capital of France" },
  },
  webSearchResult,
  { type: "text", text: "Paris is the capital.", citations: [citation] },
  { type: "tool_use", id: "toolu_1", name: "get_weather", input: { city: "Paris" } },
  {
    type: "mcp_tool_use",
    id: "mcptoolu_1",
    name: "list_issues",
    server_name: "github",
    input: { repo: "openinference" },
  },
  mcpToolResult,
  { type: "container_upload", file_id: "file_1" },
  { type: "some_future_block", value: 1 },
];

const responseMessage = {
  id: "msg_blocks",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-6",
  content: contentBlocks,
  stop_reason: "tool_use",
  stop_sequence: null,
  stop_details: null,
  container: null,
  context_management: null,
  usage: { input_tokens: 10, output_tokens: 20 },
};

function contentBlockEvents(): StreamEvent[] {
  return [
    messageStart(),
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "compaction", content: null, encrypted_content: null },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "compaction_delta", content: "Summary of ", encrypted_content: null },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "compaction_delta", content: "earlier turns", encrypted_content: null },
    },
    { type: "content_block_stop", index: 0 },
    {
      type: "content_block_start",
      index: 1,
      content_block: { type: "server_tool_use", id: "srvtoolu_1", name: "web_search", input: {} },
    },
    {
      type: "content_block_delta",
      index: 1,
      delta: { type: "input_json_delta", partial_json: '{"query": "capital ' },
    },
    {
      type: "content_block_delta",
      index: 1,
      delta: { type: "input_json_delta", partial_json: 'of France"}' },
    },
    { type: "content_block_stop", index: 1 },
    { type: "content_block_start", index: 2, content_block: webSearchResult },
    { type: "content_block_stop", index: 2 },
    {
      type: "content_block_start",
      index: 3,
      content_block: { type: "text", text: "", citations: null },
    },
    { type: "content_block_delta", index: 3, delta: { type: "citations_delta", citation } },
    {
      type: "content_block_delta",
      index: 3,
      delta: { type: "text_delta", text: "Paris is the capital." },
    },
    { type: "content_block_stop", index: 3 },
    {
      type: "content_block_start",
      index: 4,
      content_block: { type: "tool_use", id: "toolu_1", name: "get_weather", input: {} },
    },
    {
      type: "content_block_delta",
      index: 4,
      delta: { type: "input_json_delta", partial_json: '{"city": "Paris"}' },
    },
    { type: "content_block_stop", index: 4 },
    {
      type: "content_block_start",
      index: 5,
      content_block: {
        type: "mcp_tool_use",
        id: "mcptoolu_1",
        name: "list_issues",
        server_name: "github",
        input: { repo: "openinference" },
      },
    },
    { type: "content_block_stop", index: 5 },
    { type: "content_block_start", index: 6, content_block: mcpToolResult },
    { type: "content_block_stop", index: 6 },
    {
      type: "content_block_start",
      index: 7,
      content_block: { type: "container_upload", file_id: "file_1" },
    },
    { type: "content_block_stop", index: 7 },
    {
      type: "content_block_start",
      index: 8,
      content_block: { type: "some_future_block", value: 1 },
    },
    { type: "content_block_stop", index: 8 },
    messageDelta({ stopReason: "tool_use", usage: { output_tokens: 20 } }),
    { type: "message_stop" },
  ];
}

const request = {
  model: "claude-sonnet-4-6",
  max_tokens: 100,
  messages: [{ role: "user" as const, content: "What is the capital of France?" }],
};

/**
 * The output attributes a response with {@link contentBlocks} must have,
 * whether it was streamed or not.
 */
function expectContentBlockAttributes(attributes: Record<string, unknown>) {
  // compaction: the summary is recorded as text
  expect(attributes[outputContent(0, MESSAGE_CONTENT_TYPE)]).toBe("text");
  expect(attributes[outputContent(0, MESSAGE_CONTENT_TEXT)]).toBe("Summary of earlier turns");

  // server, client and MCP tool calls are all tool calls, in content order
  expect(attributes[outputContent(1, MESSAGE_CONTENT_TYPE)]).toBe("tool_use");
  expect(attributes[outputContent(1, TOOL_CALL_FUNCTION_NAME)]).toBe("web_search");
  expect(attributes[outputToolCall(0, TOOL_CALL_ID)]).toBe("srvtoolu_1");
  expect(attributes[outputToolCall(0, TOOL_CALL_FUNCTION_NAME)]).toBe("web_search");
  expect(attributes[outputToolCall(0, TOOL_CALL_FUNCTION_ARGUMENTS_JSON)]).toBe(
    JSON.stringify({ query: "capital of France" }),
  );
  expect(attributes[outputToolCall(1, TOOL_CALL_ID)]).toBe("toolu_1");
  expect(attributes[outputToolCall(1, TOOL_CALL_FUNCTION_ARGUMENTS_JSON)]).toBe(
    JSON.stringify({ city: "Paris" }),
  );
  expect(attributes[outputToolCall(2, TOOL_CALL_ID)]).toBe("mcptoolu_1");
  expect(attributes[outputToolCall(2, TOOL_CALL_FUNCTION_NAME)]).toBe("list_issues");
  expect(attributes[outputToolCall(2, TOOL_CALL_FUNCTION_ARGUMENTS_JSON)]).toBe(
    JSON.stringify({ repo: "openinference" }),
  );
  expect(attributes[outputToolCall(3, TOOL_CALL_ID)]).toBeUndefined();

  // results and other blocks keep their index, as their type and JSON
  expect(attributes[outputContent(2, MESSAGE_CONTENT_TYPE)]).toBe("web_search_tool_result");
  expect(JSON.parse(attributes[outputContent(2, MESSAGE_CONTENT_TEXT)] as string)).toEqual(
    webSearchResult,
  );
  expect(attributes[outputContent(3, MESSAGE_CONTENT_TEXT)]).toBe("Paris is the capital.");
  expect(attributes[outputContent(6, MESSAGE_CONTENT_TYPE)]).toBe("mcp_tool_result");
  expect(attributes[outputContent(7, MESSAGE_CONTENT_TYPE)]).toBe("container_upload");
  expect(attributes[outputContent(8, MESSAGE_CONTENT_TYPE)]).toBe("some_future_block");
  for (let index = 0; index < contentBlocks.length; index++) {
    expect(attributes[outputContent(index, MESSAGE_CONTENT_TYPE)], `index ${index}`).toEqual(
      expect.any(String),
    );
  }

  // citations and MCP servers have no convention of their own
  const metadata = JSON.parse(attributes[METADATA] as string);
  expect(metadata.citations).toEqual({ "3": [citation] });
  expect(metadata.mcp_server_names).toEqual({ mcptoolu_1: "github" });

  // the output is the whole message as JSON
  expect(attributes[OUTPUT_MIME_TYPE]).toBe(MimeType.JSON);
  const outputMessage = JSON.parse(attributes[OUTPUT_VALUE] as string);
  expect(outputMessage.id).toBe("msg_blocks");
  expect(outputMessage.stop_reason).toBe("tool_use");
  expect(outputMessage.content.length).toBe(contentBlocks.length);
}

describe("AnthropicInstrumentation - content blocks", () => {
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

  it("records every output content block of a non-streaming response", async () => {
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: jsonFetch(responseMessage) });
    await client.beta.messages.create(request);

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    expectContentBlockAttributes(span.attributes);
  });

  it("records every output content block of a streaming response", async () => {
    const client = new Anthropic({
      apiKey: "fake-api-key",
      fetch: sseFetch(contentBlockEvents().map(withMessageId)),
    });
    const stream = await client.beta.messages.create({ ...request, stream: true });
    for await (const _event of stream) {
      // drain
    }

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    expectContentBlockAttributes(span.attributes);
  });

  it("does not modify the events the caller receives", async () => {
    const client = new Anthropic({
      apiKey: "fake-api-key",
      fetch: sseFetch(contentBlockEvents()),
    });
    const stream = await client.beta.messages.create({ ...request, stream: true });
    const starts = [];
    for await (const event of stream) {
      if (event.type === "content_block_start") {
        starts.push(event.content_block);
      }
    }
    expect(starts[0]).toEqual({ type: "compaction", content: null, encrypted_content: null });
    expect(starts[1]).toEqual({
      type: "server_tool_use",
      id: "srvtoolu_1",
      name: "web_search",
      input: {},
    });
  });

  it("merges response metadata with the metadata set on the context, which wins", async () => {
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: jsonFetch(responseMessage) });
    await context.with(
      setMetadata(context.active(), { tenant: "acme", citations: "caller value" }),
      () => client.beta.messages.create(request),
    );

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    const metadata = JSON.parse(span.attributes[METADATA] as string);
    expect(metadata.tenant).toBe("acme");
    expect(metadata.citations).toBe("caller value");
    expect(metadata.mcp_server_names).toEqual({ mcptoolu_1: "github" });
  });

  it("records the system prompt and every input content block", async () => {
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: jsonFetch(responseMessage) });
    const inlineTool = {
      name: "lookup",
      description: "Look a word up",
      input_schema: { type: "object" as const, properties: {} },
    };
    await client.beta.messages.create({
      ...request,
      system: [
        { type: "text", text: "You are terse." },
        { type: "text", text: "Answer in French." },
      ],
      tools: [
        {
          name: "get_weather",
          description: "Get the weather",
          input_schema: { type: "object", properties: {} },
        },
        { type: "web_search_20250305", name: "web_search" },
      ],
      messages: [
        { role: "system", content: "Mid-conversation instruction." },
        {
          role: "user",
          content: [
            { type: "image", source: { type: "url", url: "https://example.com/cat.png" } },
            { type: "image", source: { type: "file", file_id: "file_image" } },
            { type: "image", source: { type: "base64", media_type: "image/png", data: "iVBOR" } },
          ],
        },
        {
          role: "assistant",
          content: [
            { type: "server_tool_use", id: "srvtoolu_1", name: "web_search", input: { q: "x" } },
            { ...webSearchResult, content: [] } as Anthropic.Beta.BetaWebSearchToolResultBlockParam,
            { type: "compaction", content: "Earlier summary", signature: "sig" },
          ],
        },
        {
          role: "system",
          content: [
            { type: "tool_addition", tool: { type: "tool_definition", definition: inlineTool } },
            { type: "tool_removal", tool: { type: "tool_reference", name: "get_weather" } },
          ],
        },
      ],
    });

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    const attributes = span.attributes;

    // The top-level system prompt is message 0, ahead of the conversation.
    expect(attributes[input(0, MESSAGE_ROLE)]).toBe("system");
    expect(attributes[inputContent(0, 0, MESSAGE_CONTENT_TEXT)]).toBe("You are terse.");
    expect(attributes[inputContent(0, 1, MESSAGE_CONTENT_TEXT)]).toBe("Answer in French.");
    expect(attributes[input(1, MESSAGE_ROLE)]).toBe("system");
    expect(attributes[input(1, MESSAGE_CONTENT)]).toBe("Mid-conversation instruction.");
    expect(attributes[input(2, MESSAGE_ROLE)]).toBe("user");

    // images: a URL, a Files API file, and base64 data as a data URI
    const image = (index: number, postfix: string) =>
      attributes[inputContent(2, index, `${MESSAGE_CONTENT_IMAGE}.${postfix}`)];
    expect(attributes[inputContent(2, 0, MESSAGE_CONTENT_TYPE)]).toBe("image");
    expect(image(0, IMAGE_URL)).toBe("https://example.com/cat.png");
    expect(attributes[inputContent(2, 1, MESSAGE_CONTENT_ID)]).toBe("file_image");
    expect(image(1, IMAGE_URL)).toBeUndefined();
    expect(image(2, IMAGE_URL)).toBe("data:image/png;base64,iVBOR");
    expect(image(2, "media_type")).toBe("image/png");

    // server tool calls are tool calls; their results and compactions keep their index
    expect(attributes[input(3, `${MESSAGE_TOOL_CALLS}.0.${TOOL_CALL_FUNCTION_NAME}`)]).toBe(
      "web_search",
    );
    expect(attributes[inputContent(3, 0, MESSAGE_CONTENT_TYPE)]).toBe("tool_use");
    expect(attributes[inputContent(3, 1, MESSAGE_CONTENT_TYPE)]).toBe("web_search_tool_result");
    expect(attributes[inputContent(3, 2, MESSAGE_CONTENT_TYPE)]).toBe("text");
    expect(attributes[inputContent(3, 2, MESSAGE_CONTENT_TEXT)]).toBe("Earlier summary");
    expect(attributes[inputContent(3, 2, MESSAGE_CONTENT_SIGNATURE)]).toBe("sig");

    // tool changes are recorded in place, and an inline tool is one of llm.tools
    expect(attributes[input(4, MESSAGE_ROLE)]).toBe("system");
    expect(attributes[inputContent(4, 0, MESSAGE_CONTENT_TYPE)]).toBe("tool_addition");
    expect(attributes[inputContent(4, 1, MESSAGE_CONTENT_TYPE)]).toBe("tool_removal");
    expect(attributes[`${LLM_TOOLS}.0.${TOOL_NAME}`]).toBe("get_weather");
    expect(attributes[`${LLM_TOOLS}.0.${TOOL_DESCRIPTION}`]).toBe("Get the weather");
    expect(attributes[`${LLM_TOOLS}.1.${TOOL_NAME}`]).toBe("web_search");
    expect(attributes[`${LLM_TOOLS}.1.${TOOL_DESCRIPTION}`]).toBeUndefined();
    expect(attributes[`${LLM_TOOLS}.2.${TOOL_NAME}`]).toBe("lookup");
    expect(attributes[`${LLM_TOOLS}.2.${TOOL_DESCRIPTION}`]).toBe("Look a word up");
    expect(attributes[`${LLM_TOOLS}.2.${TOOL_JSON_SCHEMA}`]).toBe(JSON.stringify(inlineTool));
  });

  it("redacts base64 data in the array content of a tool result", async () => {
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: jsonFetch(responseMessage) });
    await client.messages.create({
      ...request,
      messages: [
        { role: "user", content: "Snapshot camera A" },
        {
          role: "assistant",
          content: [{ type: "tool_use", id: "toolu_a", name: "snapshot", input: { cam: "A" } }],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "toolu_a",
              content: [
                { type: "text", text: "Camera A" },
                {
                  type: "image",
                  source: { type: "base64", media_type: "image/png", data: "iVBOR" },
                },
              ],
            },
          ],
        },
      ],
    });

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    const attributes = span.attributes;
    expect(attributes[input(2, "message.tool_call_id")]).toBe("toolu_a");
    expect(JSON.parse(attributes[input(2, MESSAGE_CONTENT)] as string)).toEqual([
      { type: "text", text: "Camera A" },
      {
        type: "image",
        source: { type: "base64", media_type: "image/png", data: "__REDACTED__" },
      },
    ]);
  });

  it("records a string system prompt as the first input message", async () => {
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: jsonFetch(responseMessage) });
    await client.messages.create({ ...request, system: "You are terse." });

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    expect(span.attributes[input(0, MESSAGE_ROLE)]).toBe("system");
    expect(span.attributes[input(0, MESSAGE_CONTENT)]).toBe("You are terse.");
    expect(span.attributes[input(1, MESSAGE_ROLE)]).toBe("user");
    expect(span.attributes[input(1, MESSAGE_CONTENT)]).toBe("What is the capital of France?");
  });

  it.each([{ system: "" }, { system: [] }])(
    "adds no system message for an empty system prompt ($system)",
    async ({ system }) => {
      const client = new Anthropic({ apiKey: "fake-api-key", fetch: jsonFetch(responseMessage) });
      await client.messages.create({ ...request, system });

      await waitForSpans(1);
      const [span] = memoryExporter.getFinishedSpans();
      expect(span.attributes[input(0, MESSAGE_ROLE)]).toBe("user");
      expect(span.attributes[input(0, MESSAGE_CONTENT)]).toBe("What is the capital of France?");
      expect(span.attributes[input(1, MESSAGE_ROLE)]).toBeUndefined();
    },
  );

  it("keeps the other request attributes when a message or block is malformed", async () => {
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: jsonFetch(responseMessage) });
    const malformed = [
      { role: "user", content: null },
      {
        role: "user",
        content: [
          { type: "tool_addition" },
          { type: "image" },
          { type: "text", text: "still here" },
        ],
      },
      { role: "user", content: "last" },
    ] as unknown as Anthropic.Beta.BetaMessageParam[];
    await client.beta.messages.create({ ...request, system: "Be brief.", messages: malformed });

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    const attributes = span.attributes;
    expect(attributes[SemanticConventions.LLM_MODEL_NAME]).toBe("claude-sonnet-4-6");
    expect(attributes[SemanticConventions.INPUT_VALUE]).toEqual(expect.any(String));
    expect(attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS]).toBe(
      JSON.stringify({ model: "claude-sonnet-4-6", max_tokens: 100 }),
    );
    expect(attributes[input(0, MESSAGE_CONTENT)]).toBe("Be brief.");
    expect(attributes[input(1, MESSAGE_ROLE)]).toBe("user");
    // Malformed blocks keep their index as their type and JSON.
    expect(attributes[inputContent(2, 0, MESSAGE_CONTENT_TYPE)]).toBe("tool_addition");
    expect(attributes[inputContent(2, 1, MESSAGE_CONTENT_TYPE)]).toBe("image");
    expect(attributes[inputContent(2, 1, MESSAGE_CONTENT_TEXT)]).toBe(
      JSON.stringify({ type: "image" }),
    );
    expect(attributes[inputContent(2, 2, MESSAGE_CONTENT_TEXT)]).toBe("still here");
    expect(attributes[input(3, MESSAGE_CONTENT)]).toBe("last");
  });
});

/** Gives the streamed message the id of the non-streaming one. */
function withMessageId(event: StreamEvent): StreamEvent {
  if (event.type !== "message_start") {
    return event;
  }
  return messageStart({ id: "msg_blocks" });
}
