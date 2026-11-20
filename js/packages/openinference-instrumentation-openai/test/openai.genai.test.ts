import type { Attributes } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import OpenAI, { AzureOpenAI } from "openai";
import { vi } from "vitest";

import type { TraceConfigOptions } from "@arizeai/openinference-core";

import { OpenAIInstrumentation } from "../src";
import { getAPIKey, resetRecordings, startRecording } from "./helpers/recordings";
import { mockAPIPromise } from "./mockAPIPromise";

const GEN_AI_PREFIX = "gen_ai.";

/** `gen_ai.*` attributes whose values are JSON encoded on the span. */
const JSON_ENCODED_ATTRIBUTES = [
  "gen_ai.input.messages",
  "gen_ai.output.messages",
  "gen_ai.tool.definitions",
];

/**
 * Picks the `gen_ai.*` attributes of a span, decoding the JSON encoded ones so
 * a snapshot shows their structure.
 * @param attributes - The attributes of an exported span
 * @returns The `gen_ai.*` attributes, keyed by attribute name
 */
function getGenAIAttributes(attributes: Attributes): Record<string, unknown> {
  const genAIAttributes: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(attributes)) {
    if (!key.startsWith(GEN_AI_PREFIX)) {
      continue;
    }
    const isJSONEncoded = JSON_ENCODED_ATTRIBUTES.includes(key) && typeof value === "string";
    genAIAttributes[key] = isJSONEncoded ? JSON.parse(value) : value;
  }
  return genAIAttributes;
}

const memoryExporter = new InMemorySpanExporter();
const tracerProvider = new NodeTracerProvider({
  spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
});
tracerProvider.register();

/** The function tool the chat completion tests offer the model. */
const weatherTool = {
  type: "function" as const,
  function: {
    name: "getWeather",
    description: "Get the weather for a location",
    parameters: {
      type: "object",
      properties: { location: { type: "string" } },
      required: ["location"],
    },
  },
};

/** A chat completion request the model answers with a call to `getWeather`. */
const chatRequest = {
  model: "gpt-4o",
  temperature: 0.2,
  max_completion_tokens: 256,
  messages: [
    { role: "system" as const, content: "You are a weather assistant." },
    { role: "user" as const, content: "What is the weather in Boston?" },
  ],
  tools: [weatherTool],
};

/**
 * A system prompt long enough to be eligible for prompt caching, which OpenAI
 * applies to prompts of 1024 tokens or more.
 */
const cacheableInstructions = Array.from(
  { length: 150 },
  (_, index) =>
    `Guideline ${index + 1}: when asked about the weather in region ${index + 1}, answer in one short sentence.`,
).join("\n");

/**
 * Covers the dependent side of the GenAI semantic convention export: the
 * conversion lives in `@arizeai/openinference-core`, and these tests check
 * that it reaches the spans this instrumentation creates. The requests replay
 * the network recordings in `test/recordings`; see `helpers/recordings.ts` for
 * how to capture them again.
 */
describe("OpenAIInstrumentation with enableGenAISemconv", () => {
  let instrumentation: OpenAIInstrumentation | undefined;
  let openai: OpenAI;

  /**
   * Instruments the OpenAI SDK with the given trace config.
   * @param traceConfig - The trace config under test
   */
  function instrument(traceConfig?: TraceConfigOptions) {
    instrumentation = new OpenAIInstrumentation({ tracerProvider, traceConfig });
    instrumentation.disable();
    // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
    instrumentation._modules[0].moduleExports = OpenAI;
    instrumentation.enable();
    // Retries are off so a request without a recording fails at once.
    openai = new OpenAI({ apiKey: getAPIKey("OPENAI_API_KEY"), maxRetries: 0 });
  }

  /**
   * Reads the `gen_ai.*` attributes of the exported spans.
   * @param expectedSpanCount - The number of spans the test should have exported
   * @returns The `gen_ai.*` attributes of each span, in export order
   */
  function getExportedGenAIAttributesBySpan(expectedSpanCount: number): Record<string, unknown>[] {
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(expectedSpanCount);
    return spans.map((span) => getGenAIAttributes(span.attributes));
  }

  /**
   * Reads the `gen_ai.*` attributes of the only exported span.
   * @returns The `gen_ai.*` attributes of the span
   */
  function getExportedGenAIAttributes(): Record<string, unknown> {
    return getExportedGenAIAttributesBySpan(1)[0];
  }

  beforeEach(() => {
    memoryExporter.reset();
  });
  afterEach(() => {
    instrumentation?.disable();
    resetRecordings();
    vi.clearAllMocks();
  });

  it("should not emit gen_ai attributes by default", async () => {
    instrument();
    const recording = await startRecording("genai-chat-completion-tool-calls");

    await openai.chat.completions.create(chatRequest);
    recording.stop();

    expect(getExportedGenAIAttributes()).toEqual({});
  });

  it("should emit gen_ai attributes for a chat completion with tool calls", async () => {
    instrument({ enableGenAISemconv: true });
    const recording = await startRecording("genai-chat-completion-tool-calls");

    await openai.chat.completions.create(chatRequest);
    recording.stop();

    expect(getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.input.messages": [
          {
            "parts": [
              {
                "content": "You are a weather assistant.",
                "type": "text",
              },
            ],
            "role": "system",
          },
          {
            "parts": [
              {
                "content": "What is the weather in Boston?",
                "type": "text",
              },
            ],
            "role": "user",
          },
        ],
        "gen_ai.operation.name": "chat",
        "gen_ai.output.messages": [
          {
            "finish_reason": "tool_call",
            "parts": [
              {
                "arguments": {
                  "location": "Boston",
                },
                "id": "call_Wri3i5R1q4TQnj5bMfHiGAo6",
                "name": "getWeather",
                "type": "tool_call",
              },
            ],
            "role": "assistant",
          },
        ],
        "gen_ai.provider.name": "openai",
        "gen_ai.request.max_tokens": 256,
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.request.temperature": 0.2,
        "gen_ai.response.finish_reasons": [
          "tool_call",
        ],
        "gen_ai.response.id": "chatcmpl-EmQ2huFReYnIm1oKEo5PZArQTxPaw",
        "gen_ai.response.model": "gpt-4o-2024-08-06",
        "gen_ai.tool.definitions": [
          {
            "description": "Get the weather for a location",
            "name": "getWeather",
            "parameters": {
              "properties": {
                "location": {
                  "type": "string",
                },
              },
              "required": [
                "location",
              ],
              "type": "object",
            },
            "type": "function",
          },
        ],
        "gen_ai.usage.cache_read.input_tokens": 0,
        "gen_ai.usage.input_tokens": 57,
        "gen_ai.usage.output_tokens": 14,
      }
    `);
  });

  it("should emit gen_ai attributes for a streamed chat completion", async () => {
    instrument({ enableGenAISemconv: true });
    const recording = await startRecording("genai-chat-completion-stream");

    const stream = await openai.chat.completions.create({
      messages: [{ role: "user", content: "Say this is a test" }],
      model: "gpt-4o",
      stream: true,
      stream_options: { include_usage: true },
    });
    for await (const _chunk of stream) {
      // Drain the stream so the span ends.
    }
    recording.stop();

    expect(getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.input.messages": [
          {
            "parts": [
              {
                "content": "Say this is a test",
                "type": "text",
              },
            ],
            "role": "user",
          },
        ],
        "gen_ai.operation.name": "chat",
        "gen_ai.output.messages": [
          {
            "finish_reason": "stop",
            "parts": [
              {
                "content": "This is a test.",
                "type": "text",
              },
            ],
            "role": "assistant",
          },
        ],
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.request.stream": true,
        "gen_ai.response.finish_reasons": [
          "stop",
        ],
        "gen_ai.response.model": "gpt-4o-2024-08-06",
        "gen_ai.usage.cache_read.input_tokens": 0,
        "gen_ai.usage.input_tokens": 12,
        "gen_ai.usage.output_tokens": 5,
      }
    `);
  });

  it("should emit gen_ai attributes for embeddings", async () => {
    instrument({ enableGenAISemconv: true });
    const recording = await startRecording("genai-embeddings");

    await openai.embeddings.create({
      input: "A happy moment",
      model: "text-embedding-3-small",
      // Keeps the recorded vector short.
      dimensions: 3,
    });
    recording.stop();

    expect(getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.embeddings.dimension.count": 3,
        "gen_ai.operation.name": "embeddings",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "text-embedding-3-small",
        "gen_ai.usage.input_tokens": 3,
      }
    `);
  });

  it("should emit gen_ai attributes for a multi-turn chat with a tool result", async () => {
    instrument({ enableGenAISemconv: true });
    const recording = await startRecording("genai-chat-completion-tool-result");

    await openai.chat.completions.create({
      ...chatRequest,
      messages: [
        ...chatRequest.messages,
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_5ERYvu4iTGSvDlcDQjDP3g3J",
              type: "function",
              function: { name: "getWeather", arguments: '{"location":"Boston"}' },
            },
          ],
        },
        {
          role: "tool",
          tool_call_id: "call_5ERYvu4iTGSvDlcDQjDP3g3J",
          content: '{"temperature":52,"precipitation":"rainy"}',
        },
      ],
    });
    recording.stop();

    const genAIAttributes = getExportedGenAIAttributes();
    expect(genAIAttributes["gen_ai.input.messages"]).toMatchInlineSnapshot(`
      [
        {
          "parts": [
            {
              "content": "You are a weather assistant.",
              "type": "text",
            },
          ],
          "role": "system",
        },
        {
          "parts": [
            {
              "content": "What is the weather in Boston?",
              "type": "text",
            },
          ],
          "role": "user",
        },
        {
          "parts": [
            {
              "arguments": {
                "location": "Boston",
              },
              "id": "call_5ERYvu4iTGSvDlcDQjDP3g3J",
              "name": "getWeather",
              "type": "tool_call",
            },
          ],
          "role": "assistant",
        },
        {
          "parts": [
            {
              "id": "call_5ERYvu4iTGSvDlcDQjDP3g3J",
              "response": {
                "precipitation": "rainy",
                "temperature": 52,
              },
              "type": "tool_call_response",
            },
          ],
          "role": "tool",
        },
      ]
    `);
    expect(genAIAttributes["gen_ai.output.messages"]).toMatchInlineSnapshot(`
      [
        {
          "finish_reason": "stop",
          "parts": [
            {
              "content": "The weather in Boston is currently 52°F and rainy.",
              "type": "text",
            },
          ],
          "role": "assistant",
        },
      ]
    `);
  });

  it("should emit cache read tokens for a cached prompt", async () => {
    instrument({ enableGenAISemconv: true });
    const recording = await startRecording("genai-chat-completion-cached-prompt");
    const cachedPromptRequest = {
      model: "gpt-4o",
      max_completion_tokens: 32,
      messages: [
        { role: "system" as const, content: cacheableInstructions },
        { role: "user" as const, content: "What is the weather in region 7?" },
      ],
    };

    // The first request writes the prompt cache and the second reads it.
    await openai.chat.completions.create(cachedPromptRequest);
    await openai.chat.completions.create(cachedPromptRequest);
    recording.stop();

    const [, cachedSpanAttributes] = getExportedGenAIAttributesBySpan(2);
    expect({
      "gen_ai.usage.cache_read.input_tokens":
        cachedSpanAttributes["gen_ai.usage.cache_read.input_tokens"],
      "gen_ai.usage.input_tokens": cachedSpanAttributes["gen_ai.usage.input_tokens"],
      "gen_ai.usage.output_tokens": cachedSpanAttributes["gen_ai.usage.output_tokens"],
    }).toMatchInlineSnapshot(`
      {
        "gen_ai.usage.cache_read.input_tokens": 2944,
        "gen_ai.usage.input_tokens": 3170,
        "gen_ai.usage.output_tokens": 4,
      }
    `);
  });

  it("should emit gen_ai attributes for the Responses API", async () => {
    instrument({ enableGenAISemconv: true });
    const recording = await startRecording("genai-responses-tool-call");

    await openai.responses.create({
      model: "gpt-4.1",
      instructions: "You are a weather assistant.",
      input: "What is the weather in Boston?",
      temperature: 0.2,
      max_output_tokens: 256,
      tools: [
        {
          type: "function",
          name: "get_weather",
          description: "Get the weather for a location",
          parameters: {
            type: "object",
            properties: { location: { type: "string" } },
            required: ["location"],
            additionalProperties: false,
          },
          strict: true,
        },
      ],
    });
    recording.stop();

    expect(getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.input.messages": [
          {
            "parts": [
              {
                "content": "You are a weather assistant.",
                "type": "text",
              },
            ],
            "role": "system",
          },
          {
            "parts": [
              {
                "content": "What is the weather in Boston?",
                "type": "text",
              },
            ],
            "role": "user",
          },
        ],
        "gen_ai.operation.name": "chat",
        "gen_ai.output.messages": [
          {
            "finish_reason": "tool_call",
            "parts": [
              {
                "arguments": {
                  "location": "Boston",
                },
                "id": "call_j22jbXaHUpVfl2y4CRaNQAwG",
                "name": "get_weather",
                "type": "tool_call",
              },
            ],
            "role": "assistant",
          },
        ],
        "gen_ai.provider.name": "openai",
        "gen_ai.request.max_tokens": 256,
        "gen_ai.request.model": "gpt-4.1",
        "gen_ai.request.temperature": 0.2,
        "gen_ai.response.finish_reasons": [
          "tool_call",
        ],
        "gen_ai.response.id": "resp_085f118e44b56947006b00b573908487d1bd9f263fdb521b8e",
        "gen_ai.response.model": "gpt-4.1-2025-04-14",
        "gen_ai.tool.definitions": [
          {
            "description": "Get the weather for a location",
            "name": "get_weather",
            "parameters": {
              "additionalProperties": false,
              "properties": {
                "location": {
                  "type": "string",
                },
              },
              "required": [
                "location",
              ],
              "type": "object",
            },
            "strict": true,
            "type": "function",
          },
        ],
        "gen_ai.usage.cache_creation.input_tokens": 0,
        "gen_ai.usage.cache_read.input_tokens": 0,
        "gen_ai.usage.input_tokens": 55,
        "gen_ai.usage.output_tokens": 15,
      }
    `);
  });

  // Known gap: the instrumentation does not record `llm.prompts` or
  // `llm.choices` for the legacy Completions API, so the span is labelled
  // `chat` and carries no messages. Emitting them needs matching masking rules
  // in core first, or prompts would bypass `hideInputs`.
  //
  // Not a network recording: OpenAI has retired every model the Completions
  // API served, so there is nothing left to record against and the request is
  // answered by a stub.
  it("should emit only request, response and usage attributes for a legacy completion", async () => {
    instrument({ enableGenAISemconv: true });
    const completion = {
      id: "cmpl-8fZu1H3VijJUWev9asnxaYyQvJTC9",
      object: "text_completion",
      created: 1704920149,
      model: "gpt-3.5-turbo-instruct",
      choices: [{ text: "This is a test", index: 0, logprobs: null, finish_reason: "stop" }],
      usage: { prompt_tokens: 12, completion_tokens: 5, total_tokens: 17 },
    };
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => completion));

    await openai.completions.create({
      prompt: "Say this is a test",
      model: "gpt-3.5-turbo-instruct",
    });

    expect(getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-3.5-turbo-instruct",
        "gen_ai.response.model": "gpt-3.5-turbo-instruct",
        "gen_ai.usage.input_tokens": 12,
        "gen_ai.usage.output_tokens": 5,
      }
    `);
  });

  // Not a network recording: there is no Azure OpenAI deployment to record
  // against, and the provider name comes from the client class rather than the
  // response, so the request is answered by a stub.
  it("should report the Azure OpenAI provider", async () => {
    instrument({ enableGenAISemconv: true });
    const azureOpenAI = new AzureOpenAI({
      apiKey: "fake-api-key",
      endpoint: "https://my-azure-openai.openai.azure.com",
      apiVersion: "2024-02-15-preview",
    });
    const completion = {
      id: "chatcmpl-azure",
      object: "chat.completion",
      model: "gpt-4o-2024-08-06",
      choices: [],
    };
    vi.spyOn(azureOpenAI, "post").mockImplementation(mockAPIPromise(azureOpenAI, () => completion));

    await azureOpenAI.chat.completions.create(chatRequest);

    expect(getExportedGenAIAttributes()["gen_ai.provider.name"]).toBe("azure.ai.openai");
  });

  it("should derive gen_ai attributes from masked attributes", async () => {
    instrument({ enableGenAISemconv: true, hideInputs: true, hideOutputs: true });
    const recording = await startRecording("genai-chat-completion-tool-calls");

    await openai.chat.completions.create(chatRequest);
    recording.stop();

    expect(getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.max_tokens": 256,
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.request.temperature": 0.2,
        "gen_ai.response.finish_reasons": [
          "tool_call",
        ],
        "gen_ai.response.model": "gpt-4o-2024-08-06",
        "gen_ai.usage.cache_read.input_tokens": 0,
        "gen_ai.usage.input_tokens": 57,
        "gen_ai.usage.output_tokens": 14,
      }
    `);
  });
});
