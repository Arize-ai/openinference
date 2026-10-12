import Anthropic from "@anthropic-ai/sdk";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

import { LLMProvider, SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { AnthropicInstrumentation } from "../src/instrumentation";
import { jsonFetch, messageDelta, messageStart, sseFetch } from "./helpers/sse";
import { waitForSpans as waitForSpansOn } from "./helpers/waitForSpans";

const {
  LLM_PROVIDER,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_TOTAL,
  METADATA,
} = SemanticConventions;

const memoryExporter = new InMemorySpanExporter();
const waitForSpans = (count: number) => waitForSpansOn(memoryExporter, count);

const request = {
  model: "claude-sonnet-4-6",
  max_tokens: 100,
  messages: [{ role: "user" as const, content: "Hello" }],
};

const stopDetails = { type: "refusal", category: "cyber", explanation: null };
const container = { id: "container_1", expires_at: "2026-10-12T00:00:00Z", skills: null };
const contextManagement = {
  applied_edits: [
    { type: "clear_tool_uses_20250919", cleared_tool_uses: 2, cleared_input_tokens: 50 },
  ],
};
const inputTransformations = [{ type: "thinking_dropped", message_index: 1, content_index: 0 }];
const iterations = [
  { type: "compaction", input_tokens: 1000, output_tokens: 200 },
  { type: "message", input_tokens: 10, output_tokens: 20 },
];
const serverToolUse = { web_search_requests: 1, web_fetch_requests: 0 };

const usage = {
  input_tokens: 10,
  output_tokens: 20,
  cache_creation_input_tokens: 0,
  cache_read_input_tokens: 0,
  output_tokens_details: { thinking_tokens: 12 },
  iterations,
  server_tool_use: serverToolUse,
};

const responseMessage = {
  id: "msg_details",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-6",
  content: [{ type: "text", text: "Hi", citations: null }],
  stop_reason: "refusal",
  stop_sequence: null,
  stop_details: stopDetails,
  container,
  context_management: contextManagement,
  diagnostics: { cache_miss_reason: { type: "prompt_changed" } },
  input_transformations: inputTransformations,
  usage,
};

function expectResponseDetails(attributes: Record<string, unknown>) {
  // Compaction tokens are not part of the top-level usage, and stay out.
  expect(attributes[LLM_TOKEN_COUNT_PROMPT]).toBe(10);
  expect(attributes[LLM_TOKEN_COUNT_COMPLETION]).toBe(20);
  expect(attributes[LLM_TOKEN_COUNT_TOTAL]).toBe(30);
  expect(attributes[LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING]).toBe(12);

  const metadata = JSON.parse(attributes[METADATA] as string);
  expect(metadata).toEqual({
    stop_details: stopDetails,
    cache_miss_reason: { type: "prompt_changed" },
    container_id: "container_1",
    context_management_applied_edits: contextManagement.applied_edits,
    input_transformations: inputTransformations,
    usage_iterations: iterations,
    server_tool_use: serverToolUse,
  });
}

describe("AnthropicInstrumentation - response details", () => {
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

  it("records usage details and response metadata of a non-streaming response", async () => {
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: jsonFetch(responseMessage) });
    await client.beta.messages.create(request);

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    expectResponseDetails(span.attributes);
  });

  it("records usage details and response metadata of a streaming response", async () => {
    const client = new Anthropic({
      apiKey: "fake-api-key",
      fetch: sseFetch([
        messageStart({
          diagnostics: { cache_miss_reason: { type: "prompt_changed" } },
          input_transformations: inputTransformations,
          usage: { input_tokens: 10, output_tokens: 1 },
        }),
        {
          type: "content_block_start",
          index: 0,
          content_block: { type: "text", text: "", citations: null },
        },
        { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hi" } },
        { type: "content_block_stop", index: 0 },
        {
          type: "message_delta",
          delta: {
            stop_reason: "refusal",
            stop_sequence: null,
            stop_details: stopDetails,
            container,
          },
          context_management: contextManagement,
          usage: {
            output_tokens: 20,
            output_tokens_details: { thinking_tokens: 12 },
            iterations,
            server_tool_use: serverToolUse,
          },
        },
        { type: "message_stop" },
      ]),
    });
    const stream = await client.beta.messages.create({ ...request, stream: true });
    for await (const _event of stream) {
      // drain
    }

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    expectResponseDetails(span.attributes);
    const output = JSON.parse(span.attributes[SemanticConventions.OUTPUT_VALUE] as string);
    expect(output.stop_details).toEqual(stopDetails);
    expect(output.container).toEqual(container);
    expect(output.context_management).toEqual(contextManagement);
    expect(output.input_transformations).toEqual(inputTransformations);
    expect(output.usage.output_tokens).toBe(20);
    expect(output.usage.input_tokens).toBe(10);
  });

  it("leaves zero server tool counts out of the metadata", async () => {
    const client = new Anthropic({
      apiKey: "fake-api-key",
      fetch: jsonFetch({
        ...responseMessage,
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          server_tool_use: { web_search_requests: 0, web_fetch_requests: 0 },
        },
      }),
    });
    await client.messages.create(request);

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    const metadata = JSON.parse(String(span.attributes[METADATA] ?? "{}"));
    expect(metadata.server_tool_use).toBeUndefined();
  });

  it("records no context management edits when none were applied", async () => {
    const client = new Anthropic({
      apiKey: "fake-api-key",
      fetch: jsonFetch({
        ...responseMessage,
        context_management: { applied_edits: [] },
      }),
    });
    await client.beta.messages.create(request);

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    const metadata = JSON.parse(span.attributes[METADATA] as string);
    expect(metadata).not.toHaveProperty("context_management_applied_edits");
    expect(metadata.container_id).toBe("container_1");
  });

  it("leaves token counts out when the usage has none", async () => {
    const client = new Anthropic({
      apiKey: "fake-api-key",
      fetch: sseFetch([
        messageStart({ usage: { input_tokens: null, output_tokens: null } }),
        messageDelta({ usage: { input_tokens: null, output_tokens: null } }),
        { type: "message_stop" },
      ]),
    });
    const stream = await client.messages.create({ ...request, stream: true });
    for await (const _event of stream) {
      // drain
    }

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    expect(span.attributes[LLM_TOKEN_COUNT_PROMPT]).toBeUndefined();
    expect(span.attributes[LLM_TOKEN_COUNT_COMPLETION]).toBeUndefined();
    expect(span.attributes[LLM_TOKEN_COUNT_TOTAL]).toBeUndefined();
    expect(span.attributes[METADATA]).toBeUndefined();
  });

  describe("llm.provider", () => {
    class AnthropicBedrock extends Anthropic {}
    class AnthropicVertex extends Anthropic {}
    class AnthropicFoundry extends Anthropic {}
    // A bundler may rename classes; the client fields still identify the cloud.
    class RenamedBedrock extends Anthropic {
      awsRegion = "us-east-1";
    }
    class RenamedVertex extends Anthropic {
      region = "us-east5";
      projectId = "my-project";
    }

    it.each([
      { name: "Anthropic", Client: Anthropic, provider: LLMProvider.ANTHROPIC },
      { name: "AnthropicBedrock", Client: AnthropicBedrock, provider: LLMProvider.AWS },
      { name: "AnthropicVertex", Client: AnthropicVertex, provider: LLMProvider.GOOGLE },
      { name: "AnthropicFoundry", Client: AnthropicFoundry, provider: LLMProvider.AZURE },
      { name: "a renamed Bedrock client", Client: RenamedBedrock, provider: LLMProvider.AWS },
      { name: "a renamed Vertex client", Client: RenamedVertex, provider: LLMProvider.GOOGLE },
    ])("is $provider for $name", async ({ Client, provider }) => {
      const client = new Client({ apiKey: "fake-api-key", fetch: jsonFetch(responseMessage) });
      await client.messages.create(request);
      await client.beta.messages.create(request);

      await waitForSpans(2);
      const spans = memoryExporter.getFinishedSpans();
      expect(spans).toHaveLength(2);
      for (const span of spans) {
        expect(span.attributes[LLM_PROVIDER]).toBe(provider);
      }
    });
  });
});
