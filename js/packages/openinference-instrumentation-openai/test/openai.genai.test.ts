import type { Attributes } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import OpenAI, { AzureOpenAI } from "openai";
import type { CreateEmbeddingResponse } from "openai/resources/embeddings";
import { Stream } from "openai/streaming";
import { vi } from "vitest";

import type { TraceConfigOptions } from "@arizeai/openinference-core";

import { OpenAIInstrumentation } from "../src";
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

const chatCompletion = {
  id: "chatcmpl-8hhqZDFTRD0vzExhqWnMLE7viVl7E",
  object: "chat.completion",
  created: 1705427343,
  model: "gpt-4o-2024-08-06",
  choices: [
    {
      index: 0,
      message: {
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
      logprobs: null,
      finish_reason: "tool_calls",
    },
  ],
  usage: {
    prompt_tokens: 70,
    completion_tokens: 7,
    total_tokens: 77,
    prompt_tokens_details: { cached_tokens: 32 },
  },
};

const chatRequest = {
  model: "gpt-4o",
  temperature: 0.2,
  max_completion_tokens: 256,
  messages: [
    { role: "system" as const, content: "You are a weather assistant." },
    { role: "user" as const, content: "What is the weather in Boston?" },
  ],
  tools: [
    {
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
    },
  ],
};

/**
 * Covers the dependent side of the GenAI semantic convention export: the
 * conversion lives in `@arizeai/openinference-core`, and these tests check
 * that it reaches the spans this instrumentation creates.
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
    openai = new OpenAI({ apiKey: "fake-api-key" });
  }

  /**
   * Reads the `gen_ai.*` attributes of the only exported span.
   * @returns The `gen_ai.*` attributes of the span
   */
  function getExportedGenAIAttributes(): Record<string, unknown> {
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    return getGenAIAttributes(spans[0].attributes);
  }

  beforeEach(() => {
    memoryExporter.reset();
  });
  afterEach(() => {
    instrumentation?.disable();
    vi.clearAllMocks();
  });

  it("should not emit gen_ai attributes by default", async () => {
    instrument();
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => chatCompletion));

    await openai.chat.completions.create(chatRequest);

    expect(getExportedGenAIAttributes()).toEqual({});
  });

  it("should emit gen_ai attributes for a chat completion with tool calls", async () => {
    instrument({ enableGenAISemconv: true });
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => chatCompletion));

    await openai.chat.completions.create(chatRequest);

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
                "id": "call_5ERYvu4iTGSvDlcDQjDP3g3J",
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
        "gen_ai.response.id": "chatcmpl-8hhqZDFTRD0vzExhqWnMLE7viVl7E",
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
        "gen_ai.usage.cache_read.input_tokens": 32,
        "gen_ai.usage.input_tokens": 70,
        "gen_ai.usage.output_tokens": 7,
      }
    `);
  });

  it("should emit gen_ai attributes for a streamed chat completion", async () => {
    instrument({ enableGenAISemconv: true });
    vi.spyOn(openai, "post").mockImplementation(
      mockAPIPromise(openai, () => {
        const iterator = () =>
          (async function* () {
            const model = "gpt-4o-2024-08-06";
            yield { model, choices: [{ delta: { role: "assistant", content: "This is " } }] };
            yield { model, choices: [{ delta: { content: "a test." } }] };
            yield { model, choices: [{ delta: {}, finish_reason: "stop" }] };
            yield {
              model,
              choices: [],
              usage: { prompt_tokens: 13, completion_tokens: 7, total_tokens: 20 },
            };
          })();
        return new Stream(iterator, new AbortController());
      }),
    );

    const stream = await openai.chat.completions.create({
      messages: [{ role: "user", content: "Say this is a test" }],
      model: "gpt-4o",
      stream: true,
    });
    for await (const _chunk of stream) {
      // Drain the stream so the span ends.
    }

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
        "gen_ai.usage.input_tokens": 13,
        "gen_ai.usage.output_tokens": 7,
      }
    `);
  });

  it("should emit gen_ai attributes for embeddings", async () => {
    instrument({ enableGenAISemconv: true });
    const response: CreateEmbeddingResponse = {
      object: "list",
      data: [{ object: "embedding", index: 0, embedding: [1, 2, 3] }],
      model: "text-embedding-3-small",
      usage: { prompt_tokens: 4, total_tokens: 4 },
    };
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => response));

    await openai.embeddings.create({
      input: "A happy moment",
      model: "text-embedding-3-small",
    });

    expect(getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.embeddings.dimension.count": 3,
        "gen_ai.operation.name": "embeddings",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "text-embedding-3-small",
        "gen_ai.usage.input_tokens": 4,
      }
    `);
  });

  it("should emit gen_ai attributes for a multi-turn chat with a tool result", async () => {
    instrument({ enableGenAISemconv: true });
    const completion = {
      ...chatCompletion,
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "It is 52 degrees and rainy in Boston." },
          logprobs: null,
          finish_reason: "stop",
        },
      ],
    };
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => completion));

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
              "content": "It is 52 degrees and rainy in Boston.",
              "type": "text",
            },
          ],
          "role": "assistant",
        },
      ]
    `);
  });

  it("should emit gen_ai attributes for the Responses API", async () => {
    instrument({ enableGenAISemconv: true });
    const response = {
      id: "resp_245",
      object: "response",
      status: "completed",
      model: "gpt-4.1-2025-04-14",
      output: [
        {
          id: "fc_example",
          type: "function_call",
          status: "completed",
          call_id: "call_weather",
          name: "get_weather",
          arguments: '{"location":"Boston"}',
        },
      ],
      usage: {
        input_tokens: 40,
        output_tokens: 12,
        total_tokens: 52,
        input_tokens_details: { cached_tokens: 16 },
        output_tokens_details: { reasoning_tokens: 0 },
      },
    };
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => response));

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
          },
          strict: true,
        },
      ],
    });

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
                "id": "call_weather",
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
        "gen_ai.response.id": "resp_245",
        "gen_ai.response.model": "gpt-4.1-2025-04-14",
        "gen_ai.tool.definitions": [
          {
            "description": "Get the weather for a location",
            "name": "get_weather",
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
            "strict": true,
            "type": "function",
          },
        ],
        "gen_ai.usage.cache_read.input_tokens": 16,
        "gen_ai.usage.input_tokens": 40,
        "gen_ai.usage.output_tokens": 12,
      }
    `);
  });

  // Known gap: the instrumentation does not record `llm.prompts` or
  // `llm.choices` for the legacy Completions API, so the span is labelled
  // `chat` and carries no messages. Emitting them needs matching masking rules
  // in core first, or prompts would bypass `hideInputs`.
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

  it("should report the Azure OpenAI provider", async () => {
    instrument({ enableGenAISemconv: true });
    const azureOpenAI = new AzureOpenAI({
      apiKey: "fake-api-key",
      endpoint: "https://my-azure-openai.openai.azure.com",
      apiVersion: "2024-02-15-preview",
    });
    vi.spyOn(azureOpenAI, "post").mockImplementation(
      mockAPIPromise(azureOpenAI, () => chatCompletion),
    );

    await azureOpenAI.chat.completions.create(chatRequest);

    expect(getExportedGenAIAttributes()["gen_ai.provider.name"]).toBe("azure.ai.openai");
  });

  it("should derive gen_ai attributes from masked attributes", async () => {
    instrument({ enableGenAISemconv: true, hideInputs: true, hideOutputs: true });
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => chatCompletion));

    await openai.chat.completions.create(chatRequest);

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
        "gen_ai.usage.cache_read.input_tokens": 32,
        "gen_ai.usage.input_tokens": 70,
        "gen_ai.usage.output_tokens": 7,
      }
    `);
  });
});
